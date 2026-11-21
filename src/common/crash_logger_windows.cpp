// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Crash logger for Windows.
//
// Hook: SetUnhandledExceptionFilter (plus a SIGABRT handler for abort()/asserts).
//
// The unhandled-exception filter only runs for an exception that every first-chance vectored
// handler and every SEH frame on the stack declined, so the emulator's deliberate faults never
// reach it: fastmem accesses are answered by dynarmic's function-table handler
// (exception_handler_windows.cpp, returns ExceptionContinueExecution) and SparseLargeVector page
// faults by its vectored handler (returns EXCEPTION_CONTINUE_EXECUTION). Nothing in this file
// runs, or costs anything, until a crash.
//
// The handler is written in the async-signal-safe style: a static buffer, no heap, no locks, no
// logging backend, only CreateFileW/WriteFile and a few syscalls (VirtualQuery, GlobalMemoryStatusEx,
// NtQueryInformationThread). Modules are found by reading the loader list in the PEB without
// taking the loader lock, and the stack is walked with RtlLookupFunctionEntry/RtlVirtualUnwind.
// Symbol names are not resolved here (dbghelp allocates); the report carries module+offset and
// the build ships PDBs, so resolve offline.
//
// Not covered, because the OS terminates the process without running any filter: stack overflow
// is only caught if enough stack is left for this handler (it needs about 4 KiB), and fast-fail
// terminations (__fastfail, RaiseFailFastException, heap corruption: 0xC0000409) skip every
// handler. abort() is covered by the SIGABRT handler below.

#include <windows.h>
#include <winternl.h>
#include <dbghelp.h>
#include <intrin.h>

#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include "common/crash_logger.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/scm_rev.h"
#include "common/thread.h"

namespace Common::CrashLogger {

namespace {

using u64 = std::uint64_t;

constexpr std::size_t BufferSize = 48 * 1024;
constexpr int MaxFrames = 32;
constexpr int MaxRegions = 8;
constexpr DWORD ThreadNameInformationClass = 38;

char g_buf[BufferSize];
std::size_t g_len = 0;

wchar_t g_dir[2 * MAX_PATH];
std::size_t g_dir_len = 0; // includes the trailing backslash
wchar_t g_path[2 * MAX_PATH + 64];

std::atomic<bool> g_installed{false};
std::atomic<DWORD> g_owner{0};
LPTOP_LEVEL_EXCEPTION_FILTER g_previous_filter = nullptr;
bool g_dump_enabled = false;
TestMode g_test_mode = TestMode::None;

using NtQueryInformationThreadFn = LONG(NTAPI*)(HANDLE, DWORD, PVOID, ULONG, PULONG);
NtQueryInformationThreadFn g_query_thread = nullptr;

using MiniDumpWriteDumpFn = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, MINIDUMP_TYPE,
                                          PMINIDUMP_EXCEPTION_INFORMATION,
                                          PMINIDUMP_USER_STREAM_INFORMATION,
                                          PMINIDUMP_CALLBACK_INFORMATION);
MiniDumpWriteDumpFn g_mini_dump_write_dump = nullptr;

struct Region {
    std::atomic<std::uintptr_t> base{0};
    std::atomic<std::size_t> size{0};
    const char* name = "";
};
Region g_regions[MaxRegions];
std::mutex g_regions_mutex; // registration only; the handler never takes it

// Scratch the handler needs; static so the faulting thread's (possibly nearly exhausted) stack is
// not used for it.
CONTEXT g_ctx;
EXCEPTION_RECORD g_record;
EXCEPTION_POINTERS g_pointers;
UNWIND_HISTORY_TABLE g_history;
struct {
    UNICODE_STRING name;
    WCHAR chars[128];
} g_thread_name;

// ---- Buffer output -------------------------------------------------------------------------

void Put(const char* s) {
    while (*s != '\0' && g_len < BufferSize - 1) {
        g_buf[g_len++] = *s++;
    }
}

void PutChar(char c) {
    if (g_len < BufferSize - 1) {
        g_buf[g_len++] = c;
    }
}

void PutHex(u64 value, int digits = 0) {
    char tmp[16];
    int n = 0;
    do {
        tmp[n++] = "0123456789abcdef"[value & 0xf];
        value >>= 4;
    } while (value != 0 && n < 16);
    while (n < digits && n < 16) {
        tmp[n++] = '0';
    }
    Put("0x");
    while (n > 0) {
        PutChar(tmp[--n]);
    }
}

void PutDec(u64 value) {
    char tmp[20];
    int n = 0;
    do {
        tmp[n++] = static_cast<char>('0' + value % 10);
        value /= 10;
    } while (value != 0 && n < 20);
    while (n > 0) {
        PutChar(tmp[--n]);
    }
}

void PutDec2(unsigned value) {
    PutChar(static_cast<char>('0' + (value / 10) % 10));
    PutChar(static_cast<char>('0' + value % 10));
}

void PutWide(const wchar_t* s, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        PutChar(s[i] < 0x80 && s[i] >= 0x20 ? static_cast<char>(s[i]) : '?');
    }
}

void PutReg(const char* name, u64 value) {
    Put(name);
    Put("=");
    PutHex(value, 16);
}

// ---- Loader list ---------------------------------------------------------------------------

// Leading part of the loader's LDR_DATA_TABLE_ENTRY (64-bit layout), which winternl.h hides.
struct LdrEntry {
    LIST_ENTRY InLoadOrderLinks;
    LIST_ENTRY InMemoryOrderLinks;
    LIST_ENTRY InInitializationOrderLinks;
    PVOID DllBase;
    PVOID EntryPoint;
    ULONG SizeOfImage;
    UNICODE_STRING FullDllName;
    UNICODE_STRING BaseDllName;
};

struct ModuleInfo {
    u64 base;
    u64 size;
    const wchar_t* name;
    std::size_t name_len;
};

// Finds the module containing addr (or, with addr == 0, the main executable) by walking the PEB
// loader list. Takes no lock; the list is read under SEH in case it is being modified.
bool FindModule(u64 addr, ModuleInfo* out) {
    __try {
        const PEB* peb = reinterpret_cast<const PEB*>(NtCurrentTeb()->ProcessEnvironmentBlock);
        const LIST_ENTRY* head = &peb->Ldr->InMemoryOrderModuleList;
        int budget = 1024;
        for (const LIST_ENTRY* link = head->Flink; link != head && budget-- > 0;
             link = link->Flink) {
            const LdrEntry* entry = CONTAINING_RECORD(link, LdrEntry, InMemoryOrderLinks);
            const u64 base = reinterpret_cast<u64>(entry->DllBase);
            if (base == 0) {
                continue;
            }
            const bool hit = addr == 0 || (addr >= base && addr - base < entry->SizeOfImage);
            if (hit) {
                out->base = base;
                out->size = entry->SizeOfImage;
                const wchar_t* name = entry->BaseDllName.Buffer;
                std::size_t len = entry->BaseDllName.Length / sizeof(wchar_t);
                if (name == nullptr || len == 0) {
                    name = entry->FullDllName.Buffer;
                    len = entry->FullDllName.Length / sizeof(wchar_t);
                }
                out->name = name;
                out->name_len = len;
                return true;
            }
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
    return false;
}

void PutModuleOffset(u64 addr) {
    ModuleInfo module{};
    if (FindModule(addr, &module)) {
        __try {
            PutWide(module.name, module.name_len);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            Put("?");
        }
        Put("+");
        PutHex(addr - module.base);
        Put(" (base ");
        PutHex(module.base);
        Put(")");
    } else {
        Put("<no module>");
    }
}

void PutMainImagePath() {
    ModuleInfo module{};
    if (!FindModule(0, &module)) {
        return;
    }
    __try {
        // The first entry in memory order is the executable; BaseDllName is enough to identify it.
        PutWide(module.name, module.name_len);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
}

// ---- Report sections -----------------------------------------------------------------------

const char* ExceptionName(DWORD code) {
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION:
        return "EXCEPTION_ACCESS_VIOLATION";
    case EXCEPTION_IN_PAGE_ERROR:
        return "EXCEPTION_IN_PAGE_ERROR";
    case EXCEPTION_ILLEGAL_INSTRUCTION:
        return "EXCEPTION_ILLEGAL_INSTRUCTION";
    case EXCEPTION_PRIV_INSTRUCTION:
        return "EXCEPTION_PRIV_INSTRUCTION";
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
        return "EXCEPTION_INT_DIVIDE_BY_ZERO";
    case EXCEPTION_INT_OVERFLOW:
        return "EXCEPTION_INT_OVERFLOW";
    case EXCEPTION_FLT_DIVIDE_BY_ZERO:
        return "EXCEPTION_FLT_DIVIDE_BY_ZERO";
    case EXCEPTION_STACK_OVERFLOW:
        return "EXCEPTION_STACK_OVERFLOW";
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
        return "EXCEPTION_ARRAY_BOUNDS_EXCEEDED";
    case EXCEPTION_DATATYPE_MISALIGNMENT:
        return "EXCEPTION_DATATYPE_MISALIGNMENT";
    case EXCEPTION_BREAKPOINT:
        return "EXCEPTION_BREAKPOINT";
    case EXCEPTION_GUARD_PAGE:
        return "EXCEPTION_GUARD_PAGE";
    case EXCEPTION_NONCONTINUABLE_EXCEPTION:
        return "EXCEPTION_NONCONTINUABLE_EXCEPTION";
    case 0xC0000409:
        return "STATUS_STACK_BUFFER_OVERRUN (fast fail)";
    case 0xE06D7363:
        return "C++ exception";
    case 0:
        return "(none: abort)";
    default:
        return "";
    }
}

void PutVirtualQuery(u64 addr) {
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(reinterpret_cast<LPCVOID>(addr), &mbi, sizeof(mbi)) == 0) {
        Put("VirtualQuery failed");
        return;
    }
    Put("state=");
    Put(mbi.State == MEM_COMMIT ? "COMMIT" : mbi.State == MEM_RESERVE ? "RESERVE" : "FREE");
    if (mbi.State != MEM_FREE) {
        Put(" type=");
        Put(mbi.Type == MEM_IMAGE ? "IMAGE" : mbi.Type == MEM_MAPPED ? "MAPPED" : "PRIVATE");
        Put(" protect=");
        PutHex(mbi.Protect);
    }
    Put(" region=");
    PutHex(reinterpret_cast<u64>(mbi.BaseAddress));
    Put("+");
    PutHex(mbi.RegionSize);
}

void PutRegionMembership(u64 addr) {
    bool any = false;
    for (const Region& region : g_regions) {
        const std::uintptr_t base = region.base.load(std::memory_order_acquire);
        if (base == 0) {
            continue;
        }
        any = true;
        const std::size_t size = region.size.load(std::memory_order_relaxed);
        if (addr >= base && addr - base < size) {
            Put("INSIDE guest memory region '");
            Put(region.name);
            Put("' at offset ");
            PutHex(addr - base);
            return;
        }
    }
    Put(any ? "not inside any registered guest memory region"
            : "no guest memory region registered yet");
}

void PutRegisters(const CONTEXT& c) {
#if defined(_M_X64)
    Put("registers:\n");
    const struct {
        const char* name;
        u64 value;
    } regs[] = {
        {"RAX", c.Rax}, {"RCX", c.Rcx}, {"RDX", c.Rdx}, {"RBX", c.Rbx},
        {"RSP", c.Rsp}, {"RBP", c.Rbp}, {"RSI", c.Rsi}, {"RDI", c.Rdi},
        {"R8 ", c.R8},  {"R9 ", c.R9},  {"R10", c.R10}, {"R11", c.R11},
        {"R12", c.R12}, {"R13", c.R13}, {"R14", c.R14}, {"R15", c.R15},
        {"RIP", c.Rip},
    };
    int column = 0;
    for (const auto& reg : regs) {
        Put("  ");
        PutReg(reg.name, reg.value);
        if (++column % 4 == 0) {
            PutChar('\n');
        }
    }
    Put("  ");
    Put("EFLAGS=");
    PutHex(c.EFlags, 8);
    Put(" MXCSR=");
    PutHex(c.MxCsr, 8);
    PutChar('\n');
#elif defined(_M_ARM64)
    Put("registers:\n");
    for (int i = 0; i < 29; ++i) {
        Put(" x");
        PutDec(static_cast<u64>(i));
        Put("=");
        PutHex(c.X[i], 16);
        if ((i + 1) % 4 == 0) {
            PutChar('\n');
        }
    }
    Put("\n fp=");
    PutHex(c.Fp, 16);
    Put(" lr=");
    PutHex(c.Lr, 16);
    Put(" sp=");
    PutHex(c.Sp, 16);
    Put(" pc=");
    PutHex(c.Pc, 16);
    PutChar('\n');
#endif
}

#if defined(_M_X64)
u64 ContextPc(const CONTEXT& c) {
    return c.Rip;
}
u64 ContextSp(const CONTEXT& c) {
    return c.Rsp;
}
#elif defined(_M_ARM64)
u64 ContextPc(const CONTEXT& c) {
    return c.Pc;
}
u64 ContextSp(const CONTEXT& c) {
    return c.Sp;
}
#else
u64 ContextPc(const CONTEXT&) {
    return 0;
}
u64 ContextSp(const CONTEXT&) {
    return 0;
}
#endif

// Walks the stack of the context in g_ctx (it is consumed). Guarded: the stack may be garbage.
void WalkStack() {
#if defined(_M_X64)
    std::memset(&g_history, 0, sizeof(g_history));
    __try {
        for (int i = 0; i < MaxFrames; ++i) {
            const u64 pc = g_ctx.Rip;
            if (pc == 0) {
                break;
            }
            Put("  #");
            PutDec(static_cast<u64>(i));
            if (i < 10) {
                PutChar(' ');
            }
            Put(" ");
            PutHex(pc, 16);
            Put("  ");
            PutModuleOffset(pc);
            PutChar('\n');

            // A return address points after the call; look up the call itself.
            const u64 lookup_pc = i == 0 ? pc : pc - 1;
            DWORD64 image_base = 0;
            const RUNTIME_FUNCTION* function = RtlLookupFunctionEntry(lookup_pc, &image_base, &g_history);
            if (function != nullptr) {
                PVOID handler_data = nullptr;
                DWORD64 establisher = 0;
                RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, lookup_pc,
                                 const_cast<PRUNTIME_FUNCTION>(function), &g_ctx, &handler_data,
                                 &establisher, nullptr);
            } else {
                // Leaf function: the return address is at the top of the stack.
                g_ctx.Rip = *reinterpret_cast<const u64*>(g_ctx.Rsp);
                g_ctx.Rsp += sizeof(u64);
            }
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        Put("  (stack walk stopped: unreadable stack)\n");
    }
#else
    Put("  (stack walk not implemented for this architecture)\n");
#endif
}

void PutThreadName() {
    if (g_query_thread == nullptr) {
        Put("<unknown>");
        return;
    }
    std::memset(&g_thread_name, 0, sizeof(g_thread_name));
    ULONG returned = 0;
    const LONG status = g_query_thread(GetCurrentThread(), ThreadNameInformationClass,
                                       &g_thread_name, sizeof(g_thread_name), &returned);
    if (status < 0 || g_thread_name.name.Length == 0 || g_thread_name.name.Buffer == nullptr) {
        Put("<unnamed>");
        return;
    }
    // Buffer points into g_thread_name.chars for the in-place result; guard anyway.
    __try {
        PutWide(g_thread_name.name.Buffer, g_thread_name.name.Length / sizeof(wchar_t));
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        Put("<unreadable>");
    }
}

void BuildPath(const SYSTEMTIME& t, const wchar_t* extension) {
    std::size_t n = 0;
    for (; n < g_dir_len; ++n) {
        g_path[n] = g_dir[n];
    }
    const wchar_t* prefix = L"crash-";
    for (; *prefix != L'\0'; ++prefix) {
        g_path[n++] = *prefix;
    }
    const unsigned fields[] = {t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond};
    for (int i = 0; i < 6; ++i) {
        if (i == 3) {
            g_path[n++] = L'-';
        }
        if (i == 0) {
            g_path[n++] = static_cast<wchar_t>(L'0' + (fields[i] / 1000) % 10);
            g_path[n++] = static_cast<wchar_t>(L'0' + (fields[i] / 100) % 10);
        }
        g_path[n++] = static_cast<wchar_t>(L'0' + (fields[i] / 10) % 10);
        g_path[n++] = static_cast<wchar_t>(L'0' + fields[i] % 10);
    }
    for (; *extension != L'\0'; ++extension) {
        g_path[n++] = *extension;
    }
    g_path[n] = L'\0';
}

void WriteBufferToFile(const wchar_t* path) {
    const HANDLE file = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return;
    }
    std::size_t done = 0;
    while (done < g_len) {
        DWORD written = 0;
        if (!WriteFile(file, g_buf + done, static_cast<DWORD>(g_len - done), &written, nullptr) ||
            written == 0) {
            break;
        }
        done += written;
    }
    FlushFileBuffers(file);
    CloseHandle(file);
}

void WriteMiniDump(const SYSTEMTIME& t) {
    if (g_mini_dump_write_dump == nullptr) {
        return;
    }
    BuildPath(t, L".dmp");
    const HANDLE file = CreateFileW(g_path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return;
    }
    MINIDUMP_EXCEPTION_INFORMATION info{};
    info.ThreadId = GetCurrentThreadId();
    info.ExceptionPointers = &g_pointers;
    info.ClientPointers = FALSE;
    // Small on purpose: stacks, module list, thread info. Allocates inside dbghelp; this is the
    // opt-in best effort part (EDEN_CRASH_DUMP=1), the text report is already on disk.
    const auto type = static_cast<MINIDUMP_TYPE>(MiniDumpNormal | MiniDumpWithThreadInfo |
                                                 MiniDumpWithUnloadedModules);
    __try {
        g_mini_dump_write_dump(GetCurrentProcess(), GetCurrentProcessId(), file, type, &info,
                               nullptr, nullptr);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
    CloseHandle(file);
}

// Builds and writes the report. record/context describe the crash (g_record/g_ctx are copies).
// kind: what led here, for the first line.
void WriteReport(const char* kind) {
    SYSTEMTIME now;
    GetLocalTime(&now);

    g_len = 0;
    Put("n2den crash report\n");
    Put("trigger: ");
    Put(kind);
    PutChar('\n');
    Put("build: ");
    Put(Common::g_scm_desc);
    Put(" (");
    Put(Common::g_scm_rev);
    Put(", branch ");
    Put(Common::g_scm_branch);
    Put(")\n");
    Put("time: ");
    PutDec(now.wYear);
    PutChar('-');
    PutDec2(now.wMonth);
    PutChar('-');
    PutDec2(now.wDay);
    PutChar(' ');
    PutDec2(now.wHour);
    PutChar(':');
    PutDec2(now.wMinute);
    PutChar(':');
    PutDec2(now.wSecond);
    Put(" (local)\n");
    Put("process: ");
    PutMainImagePath();
    Put(" pid ");
    PutDec(GetCurrentProcessId());
    Put("\n");
    Put("thread: id ");
    PutDec(GetCurrentThreadId());
    Put(" name '");
    PutThreadName();
    Put("'\n");

    const EXCEPTION_RECORD& rec = g_record;
    Put("exception: ");
    PutHex(rec.ExceptionCode, 8);
    Put(" ");
    Put(ExceptionName(rec.ExceptionCode));
    Put(" flags ");
    PutHex(rec.ExceptionFlags);
    Put(" address ");
    PutHex(reinterpret_cast<u64>(rec.ExceptionAddress), 16);
    PutChar('\n');

    if ((rec.ExceptionCode == EXCEPTION_ACCESS_VIOLATION ||
         rec.ExceptionCode == EXCEPTION_IN_PAGE_ERROR) &&
        rec.NumberParameters >= 2) {
        const u64 kind_code = rec.ExceptionInformation[0];
        const u64 target = rec.ExceptionInformation[1];
        Put("access: ");
        Put(kind_code == 0 ? "READ" : kind_code == 1 ? "WRITE" : kind_code == 8 ? "EXECUTE" : "?");
        Put(" of address ");
        PutHex(target, 16);
        PutChar('\n');
        Put("target: ");
        PutVirtualQuery(target);
        PutChar('\n');
        Put("target vs guest memory: ");
        PutRegionMembership(target);
        PutChar('\n');
        if (rec.ExceptionCode == EXCEPTION_IN_PAGE_ERROR && rec.NumberParameters >= 3) {
            Put("in-page status: ");
            PutHex(rec.ExceptionInformation[2], 8);
            PutChar('\n');
        }
    } else {
        for (DWORD i = 0; i < rec.NumberParameters && i < 4; ++i) {
            Put("parameter ");
            PutDec(i);
            Put(": ");
            PutHex(rec.ExceptionInformation[i]);
            PutChar('\n');
        }
    }

    const u64 pc = ContextPc(g_ctx);
    Put("fault pc: ");
    PutHex(pc, 16);
    Put("  ");
    PutModuleOffset(pc);
    PutChar('\n');
    Put("pc memory: ");
    PutVirtualQuery(pc);
    PutChar('\n');
    Put("stack pointer: ");
    PutHex(ContextSp(g_ctx), 16);
    Put("  ");
    PutVirtualQuery(ContextSp(g_ctx));
    PutChar('\n');

    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem)) {
        Put("memory: physical free ");
        PutDec(mem.ullAvailPhys >> 20);
        Put(" MiB of ");
        PutDec(mem.ullTotalPhys >> 20);
        Put(" MiB, commit free ");
        PutDec(mem.ullAvailPageFile >> 20);
        Put(" MiB of ");
        PutDec(mem.ullTotalPageFile >> 20);
        Put(" MiB, address space free ");
        PutDec(mem.ullAvailVirtual >> 20);
        Put(" MiB\n");
    }

    PutRegisters(g_ctx);

    Put("stack (return addresses as module+offset; resolve with the matching PDB):\n");
    WalkStack();
    Put("end of report\n");

    BuildPath(now, L".txt");
    WriteBufferToFile(g_path);

    // Tell whoever started us (console, script) where the report went.
    const HANDLE err = GetStdHandle(STD_ERROR_HANDLE);
    if (err != nullptr && err != INVALID_HANDLE_VALUE) {
        static const char prefix[] = "\nn2den crashed; report written to ";
        DWORD written = 0;
        WriteFile(err, prefix, sizeof(prefix) - 1, &written, nullptr);
        for (const wchar_t* p = g_path; *p != L'\0'; ++p) {
            const char c = *p < 0x80 ? static_cast<char>(*p) : '?';
            WriteFile(err, &c, 1, &written, nullptr);
        }
        WriteFile(err, "\n", 1, &written, nullptr);
    }

    if (g_dump_enabled) {
        WriteMiniDump(now);
    }
}

// Returns true if this thread may write the report. A second crashing thread waits (bounded) for
// the first to finish; a crash inside the handler itself gives up.
bool EnterHandler() {
    const DWORD self = GetCurrentThreadId();
    DWORD expected = 0;
    if (g_owner.compare_exchange_strong(expected, self)) {
        return true;
    }
    if (expected != self) {
        Sleep(5000);
    }
    return false;
}

LONG WINAPI UnhandledFilter(EXCEPTION_POINTERS* info) {
    if (EnterHandler()) {
        g_record = *info->ExceptionRecord;
        g_ctx = *info->ContextRecord;
        g_pointers.ExceptionRecord = info->ExceptionRecord;
        g_pointers.ContextRecord = info->ContextRecord;
        WriteReport("unhandled exception");
    }
    // Hand over to whatever was installed before us (the C runtime's filter), then the default
    // handling, so exit codes and Windows Error Reporting behave exactly as without the logger.
    if (g_previous_filter != nullptr) {
        return g_previous_filter(info);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

// abort(), failed asserts and std::terminate end in raise(SIGABRT) and then a fast fail that no
// exception filter sees. The CRT calls this handler first.
void AbortHandler(int) {
    if (EnterHandler()) {
        std::memset(&g_record, 0, sizeof(g_record));
        RtlCaptureContext(&g_ctx);
        g_record.ExceptionAddress = reinterpret_cast<PVOID>(ContextPc(g_ctx));
        g_pointers.ExceptionRecord = &g_record;
        g_pointers.ContextRecord = &g_ctx;
        WriteReport("abort() / SIGABRT (frame #0 is the handler; the caller's frames follow)");
    }
    // Returning lets the CRT carry on with the normal abort.
    std::signal(SIGABRT, SIG_DFL);
}

bool EnvIs(const char* name, const char* value) {
    char buffer[16];
    const DWORD n = GetEnvironmentVariableA(name, buffer, sizeof(buffer));
    return n != 0 && n < sizeof(buffer) && std::strcmp(buffer, value) == 0;
}

__declspec(noinline) void NullWrite() {
    *static_cast<volatile int*>(nullptr) = 1;
}

__declspec(noinline) void CrashTestLevel2() {
    NullWrite();
    _ReadWriteBarrier();
}

__declspec(noinline) void CrashTestLevel1() {
    CrashTestLevel2();
    _ReadWriteBarrier();
}

} // namespace

void Install() {
    if (g_installed.exchange(true)) {
        return;
    }

    // Where to write: next to the log. Created now so the handler never has to.
    const auto& log_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir);
    void(Common::FS::CreateDirs(log_dir));
    const std::wstring dir = log_dir.wstring();
    if (dir.size() + 1 >= sizeof(g_dir) / sizeof(g_dir[0])) {
        return;
    }
    std::memcpy(g_dir, dir.c_str(), dir.size() * sizeof(wchar_t));
    g_dir_len = dir.size();
    if (g_dir_len != 0 && g_dir[g_dir_len - 1] != L'\\' && g_dir[g_dir_len - 1] != L'/') {
        g_dir[g_dir_len++] = L'\\';
    }
    g_dir[g_dir_len] = L'\0';

    if (const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll")) {
        g_query_thread = reinterpret_cast<NtQueryInformationThreadFn>(
            reinterpret_cast<void*>(GetProcAddress(ntdll, "NtQueryInformationThread")));
    }

    if (EnvIs("EDEN_CRASH_DUMP", "1")) {
        // Loaded now so the handler does not load anything.
        if (const HMODULE dbghelp =
                LoadLibraryExW(L"dbghelp.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32)) {
            g_mini_dump_write_dump = reinterpret_cast<MiniDumpWriteDumpFn>(
                reinterpret_cast<void*>(GetProcAddress(dbghelp, "MiniDumpWriteDump")));
            g_dump_enabled = g_mini_dump_write_dump != nullptr;
        }
    }

    g_previous_filter = SetUnhandledExceptionFilter(&UnhandledFilter);
    std::signal(SIGABRT, &AbortHandler);

    if (EnvIs("EDEN_CRASH_TEST", "av")) {
        g_test_mode = TestMode::Worker;
        std::thread([] {
            Common::SetCurrentThreadName("CrashTestWorker");
            Sleep(4000);
            CrashTestLevel1();
        }).detach();
    } else if (EnvIs("EDEN_CRASH_TEST", "av-main")) {
        g_test_mode = TestMode::Main;
    }
}

void Reassert() {
    if (!g_installed.load()) {
        return;
    }
    const auto current = SetUnhandledExceptionFilter(&UnhandledFilter);
    if (current != &UnhandledFilter && current != nullptr) {
        // Somebody replaced us; they become the filter we chain to.
        g_previous_filter = current;
    }
}

void RegisterRegion(const char* name, const void* base, std::size_t size) {
    std::scoped_lock lock{g_regions_mutex};
    for (Region& region : g_regions) {
        if (region.base.load() == 0) {
            region.name = name;
            region.size.store(size);
            region.base.store(reinterpret_cast<std::uintptr_t>(base), std::memory_order_release);
            return;
        }
    }
}

void UnregisterRegion(const void* base) {
    std::scoped_lock lock{g_regions_mutex};
    for (Region& region : g_regions) {
        if (region.base.load() == reinterpret_cast<std::uintptr_t>(base)) {
            region.base.store(0);
        }
    }
}

TestMode GetTestMode() {
    return g_test_mode;
}

void TriggerTestCrash() {
    CrashTestLevel1();
    std::abort();
}

} // namespace Common::CrashLogger
