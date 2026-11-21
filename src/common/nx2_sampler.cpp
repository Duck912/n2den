// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>
#include <fmt/format.h>
#include "common/nx2_sampler.h"
#include "common/scope_exit.h"
#include "common/string_util.h"

#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
#include <windows.h>
// windows.h first: dbghelp.h and tlhelp32.h use its types.
#include <dbghelp.h>
#include <tlhelp32.h>
#endif

namespace Nx2Sampler {
#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
namespace {
struct Snapshot {
    DWORD64 base{};
    SIZE_T size{};
    std::array<unsigned char, 65536> stack;
};
thread_local Snapshot* snapshot{};

BOOL CALLBACK ReadStack(HANDLE process, DWORD64 address, PVOID buffer, DWORD size, LPDWORD read) {
    *read = 0;
    if (snapshot && address >= snapshot->base && address - snapshot->base <= snapshot->size &&
        size <= snapshot->size - (address - snapshot->base)) {
        std::memcpy(buffer, snapshot->stack.data() + (address - snapshot->base), size);
        *read = size;
        return TRUE;
    }
    // Never read a live stack after its owner has resumed.
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(reinterpret_cast<void*>(address), &info, sizeof(info)) ||
        info.Type != MEM_IMAGE)
        return FALSE;
    SIZE_T amount{};
    const BOOL ok =
        ReadProcessMemory(process, reinterpret_cast<void*>(address), buffer, size, &amount);
    *read = static_cast<DWORD>(amount);
    return ok;
}
struct Thread {
    HANDLE handle;
    std::string name;
};
std::vector<Thread> OpenNamedThreads(const char* names) {
    const HANDLE list = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (list == INVALID_HANDLE_VALUE)
        throw std::runtime_error{"cannot enumerate host threads"};
    SCOPE_EXIT {
        CloseHandle(list);
    };
    const auto description = reinterpret_cast<HRESULT(WINAPI*)(HANDLE, PWSTR*)>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription"));
    if (!description)
        throw std::runtime_error{"thread descriptions are unavailable"};
    std::vector<std::string> wanted;
    std::istringstream input{names};
    for (std::string name; std::getline(input, name, ',');)
        wanted.push_back(name);
    std::vector<Thread> threads;
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    if (Thread32First(list, &entry))
        do {
            if (entry.th32OwnerProcessID != GetCurrentProcessId() ||
                entry.th32ThreadID == GetCurrentThreadId())
                continue;
            HANDLE handle =
                OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                           FALSE, entry.th32ThreadID);
            if (!handle)
                continue;
            PWSTR wide{};
            std::string name;
            if (SUCCEEDED(description(handle, &wide)) && wide) {
                name = Common::UTF16ToUTF8(wide);
                LocalFree(wide);
            }
            if (std::ranges::find(wanted, name) != wanted.end())
                threads.push_back({handle, name});
            else
                CloseHandle(handle);
        } while (Thread32Next(list, &entry));
    if (threads.empty())
        throw std::runtime_error{"no named host threads found"};
    std::ranges::sort(threads, {}, &Thread::name);
    return threads;
}
std::string Symbol(HANDLE process, DWORD64 address) {
    alignas(SYMBOL_INFO) std::array<char, sizeof(SYMBOL_INFO) + MAX_SYM_NAME> storage{};
    auto* symbol = reinterpret_cast<SYMBOL_INFO*>(storage.data());
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = MAX_SYM_NAME;
    DWORD64 displacement{};
    if (SymFromAddr(process, address, &displacement, symbol))
        return symbol->Name;
    return fmt::format("0x{:x}", address);
}
} // namespace
#endif

std::string Sample(const char* names, double seconds, const std::string& path) {
#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
    static std::mutex mutex;
    std::scoped_lock lock{mutex};
    const HANDLE process = GetCurrentProcess();
    const bool initialized = SymInitialize(process, nullptr, TRUE) != FALSE;
    SCOPE_EXIT {
        if (initialized)
            SymCleanup(process);
    };
    std::vector<Thread> threads = OpenNamedThreads(names);
    SCOPE_EXIT {
        for (const auto& thread : threads)
            CloseHandle(thread.handle);
    };
    using Stack = std::vector<DWORD64>;
    std::map<std::pair<std::string, Stack>, size_t> samples;
    const auto begin = std::chrono::steady_clock::now();
    const auto end = begin + std::chrono::duration<double>{seconds};
    Snapshot copy{};
    while (std::chrono::steady_clock::now() < end) {
        const auto next = std::chrono::steady_clock::now() + std::chrono::milliseconds{1};
        for (const auto& thread : threads) {
            CONTEXT context{};
            context.ContextFlags = CONTEXT_FULL;
            bool copied = false;
            if (SuspendThread(thread.handle) == DWORD(-1))
                continue;
            // Only copy registers and stack while suspended; unwind and allocate after resuming.
            if (GetThreadContext(thread.handle, &context)) {
                copy.base = context.Rsp;
                copy.size = 0;
                copied =
                    ReadProcessMemory(process, reinterpret_cast<void*>(copy.base),
                                      copy.stack.data(), copy.stack.size(), &copy.size) != FALSE;
                copied |= copy.size != 0;
            }
            ResumeThread(thread.handle);
            if (!copied)
                continue;
            snapshot = &copy;
            STACKFRAME64 frame{};
            frame.AddrPC = {context.Rip, 0, AddrModeFlat};
            frame.AddrStack = {context.Rsp, 0, AddrModeFlat};
            frame.AddrFrame = {context.Rbp, 0, AddrModeFlat};
            const DWORD64 rip = context.Rip;
            Stack stack{rip};
            DWORD64 last_sp = 0;
            for (size_t walked = 0; walked < 64; ++walked) {
                if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, process, thread.handle, &frame, &context,
                                 ReadStack, SymFunctionTableAccess64, SymGetModuleBase64,
                                 nullptr) ||
                    !frame.AddrPC.Offset)
                    break;
                // The first call reports the sampled frame itself, already in the stack.
                if (walked == 0 && frame.AddrPC.Offset == rip) {
                    last_sp = frame.AddrStack.Offset;
                    continue;
                }
                // Each caller sits higher on the stack; stop if the walk stops climbing.
                if (walked != 0 && frame.AddrStack.Offset <= last_sp)
                    break;
                last_sp = frame.AddrStack.Offset;
                stack.push_back(frame.AddrPC.Offset);
            }
            snapshot = nullptr;
            ++samples[{thread.name, std::move(stack)}];
        }
        std::this_thread::sleep_until(next);
    }
    std::map<std::string, size_t> self, callers, inclusive;
    std::ofstream folded{std::filesystem::u8path(path + ".folded")};
    for (const auto& [key, count] : samples) {
        std::vector<std::string> stack;
        for (const auto address : key.second)
            stack.push_back(Symbol(process, address));
        self[key.first + ": " + stack[0]] += count;
        if (stack.size() > 1)
            callers[key.first + ": " + stack[0] + " <- " + stack[1]] += count;
        auto unique = stack;
        std::ranges::sort(unique);
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        for (const auto& name : unique)
            inclusive[key.first + ": " + name] += count;
        folded << key.first;
        for (auto it = stack.rbegin(); it != stack.rend(); ++it)
            folded << ';' << *it;
        folded << ' ' << count << '\n';
    }
    std::string report = fmt::format("Sampled {} for {:.3f} seconds\n", names, seconds);
    const auto section = [&](std::string_view title, const auto& counts) {
        report += fmt::format("\n{}\n", title);
        std::vector<std::pair<size_t, std::string>> sorted;
        for (const auto& [name, count] : counts)
            sorted.emplace_back(count, name);
        std::sort(sorted.rbegin(), sorted.rend());
        for (const auto& [count, name] : sorted)
            report += fmt::format("{:8} {}\n", count, name);
    };
    section("Self", self);
    section("Self <- caller", callers);
    section("Inclusive", inclusive);
    std::ofstream output{std::filesystem::u8path(path)};
    output << report;
    output.close();
    folded.close();
    if (!output || !folded)
        throw std::runtime_error{"cannot write sampling report"};
    return report;
#else
    throw std::runtime_error{"host thread sampling requires Windows x64"};
#endif
}

RipRun SampleRips(const char* names, double seconds, std::uint32_t period_us,
                  const std::function<void(std::span<const RipSample>)>& batch,
                  const std::function<bool()>& stop) {
#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
    std::vector<Thread> threads = OpenNamedThreads(names);
    SCOPE_EXIT {
        for (const auto& thread : threads)
            CloseHandle(thread.handle);
    };
    RipRun run;
    for (const auto& thread : threads)
        run.threads.push_back(thread.name);
    // A high-resolution waitable timer paces the rounds; sleep_until would round up to the
    // system timer period.
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                          TIMER_ALL_ACCESS);
    SCOPE_EXIT {
        if (timer)
            CloseHandle(timer);
    };
    const auto period = std::chrono::microseconds{(std::max)(period_us, std::uint32_t{100})};
    std::vector<RipSample> pending;
    pending.reserve(threads.size() * 512);
    std::vector<std::uint64_t> rips(threads.size());
    std::vector<char> taken(threads.size());
    const auto begin = std::chrono::steady_clock::now();
    const auto end = begin + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                 std::chrono::duration<double>{seconds});
    auto next = begin;
    auto flush = begin + std::chrono::milliseconds{200};
    while (std::chrono::steady_clock::now() < end && !stop()) {
        for (size_t i = 0; i < threads.size(); ++i) {
            // Nothing that can allocate or lock while a thread is suspended.
            taken[i] = 0;
            if (SuspendThread(threads[i].handle) == DWORD(-1))
                continue;
            CONTEXT context{};
            context.ContextFlags = CONTEXT_CONTROL;
            if (GetThreadContext(threads[i].handle, &context)) {
                rips[i] = context.Rip;
                taken[i] = 1;
            }
            ResumeThread(threads[i].handle);
        }
        for (size_t i = 0; i < threads.size(); ++i) {
            if (taken[i])
                pending.push_back({static_cast<std::uint32_t>(i), rips[i]});
            else
                ++run.missed;
        }
        ++run.passes;
        const auto now = std::chrono::steady_clock::now();
        if (now >= flush) {
            batch(pending);
            pending.clear();
            flush = now + std::chrono::milliseconds{200};
        }
        next += period;
        if (next < now)
            next = now;
        const auto wait = std::chrono::duration_cast<std::chrono::nanoseconds>(next - now).count();
        if (timer && wait > 0) {
            LARGE_INTEGER due{};
            due.QuadPart = -static_cast<LONGLONG>(wait / 100);
            if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE))
                WaitForSingleObject(timer, INFINITE);
        } else if (wait > 0) {
            std::this_thread::sleep_until(next);
        }
    }
    batch(pending);
    run.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    return run;
#else
    (void)names, (void)seconds, (void)period_us, (void)batch, (void)stop;
    throw std::runtime_error{"host thread sampling requires Windows x64"};
#endif
}

std::string HostModuleName(std::uint64_t address) {
#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
    HMODULE module{};
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(address), &module))
        return {};
    std::array<wchar_t, MAX_PATH> path{};
    const DWORD length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0)
        return {};
    return Common::UTF16ToUTF8(
        std::filesystem::path{std::wstring{path.data(), length}}.filename().wstring());
#else
    (void)address;
    return {};
#endif
}
} // namespace Nx2Sampler
