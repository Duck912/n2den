// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "yuzu/render/system_monitor.h"

#include "common/thread.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <span>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <psapi.h>
#include <tlhelp32.h>
#elif defined(__linux__)
#include <dirent.h>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

using namespace std::chrono_literals;

namespace {

// ---------------------------------------------------------------------------------------------
// NVML (NVIDIA Management Library), loaded at runtime: it ships with the NVIDIA driver, and on
// other machines the overlay simply goes without temperature, clock and power.
// ---------------------------------------------------------------------------------------------
#if defined(_WIN32) || defined(__linux__)
class Nvml {
public:
    Nvml() {
#ifdef _WIN32
        library = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!library) {
            // Drivers before R460 kept it here instead of System32.
            library = LoadLibraryExW(L"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll",
                                     nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        }
#else
        library = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
        if (!library || !Load(init, "nvmlInit_v2") || !Load(shutdown, "nvmlShutdown") ||
            !Load(get_count, "nvmlDeviceGetCount_v2") ||
            !Load(get_handle, "nvmlDeviceGetHandleByIndex_v2") ||
            !Load(get_name, "nvmlDeviceGetName") ||
            !Load(get_utilization, "nvmlDeviceGetUtilizationRates") ||
            !Load(get_memory, "nvmlDeviceGetMemoryInfo") ||
            !Load(get_temperature, "nvmlDeviceGetTemperature") ||
            !Load(get_clock, "nvmlDeviceGetClockInfo") ||
            !Load(get_power, "nvmlDeviceGetPowerUsage") || init() != 0) {
            Unload();
            return;
        }
        initialized = true;
    }

    ~Nvml() {
        if (initialized) {
            shutdown();
        }
        Unload();
    }

    /// Picks the device whose name matches `preferred` (the adapter the OS reports), or the first.
    bool SelectDevice(const std::string& preferred) {
        if (!initialized) {
            return false;
        }
        unsigned count = 0;
        if (get_count(&count) != 0 || count == 0) {
            return false;
        }
        for (unsigned i = 0; i < count; ++i) {
            Device handle{};
            if (get_handle(i, &handle) != 0) {
                continue;
            }
            char name[96]{};
            if (get_name(handle, name, sizeof(name)) != 0) {
                continue;
            }
            if (!device || (!preferred.empty() && preferred == name)) {
                device = handle;
                device_name = name;
            }
        }
        return device != nullptr;
    }

    [[nodiscard]] const std::string& Name() const {
        return device_name;
    }

    void Sample(SystemStats& out) const {
        if (!device) {
            return;
        }
        Utilization util{};
        if (get_utilization(device, &util) == 0) {
            out.gpu_usage = util.gpu;
        }
        Memory memory{};
        if (get_memory(device, &memory) == 0) {
            out.vram_used = memory.used;
            out.vram_total = memory.total;
        }
        unsigned value = 0;
        if (get_temperature(device, 0 /* NVML_TEMPERATURE_GPU */, &value) == 0) {
            out.gpu_temp_c = value;
        }
        if (get_clock(device, 0 /* NVML_CLOCK_GRAPHICS */, &value) == 0) {
            out.gpu_clock_mhz = value;
        }
        if (get_power(device, &value) == 0) {
            out.gpu_power_w = value / 1000.0;
        }
    }

private:
    struct DeviceSt;
    using Device = DeviceSt*;
    struct Utilization {
        unsigned gpu;
        unsigned memory;
    };
    struct Memory {
        unsigned long long total;
        unsigned long long free;
        unsigned long long used;
    };

    template <typename F>
    bool Load(F& fn, const char* name) {
#ifdef _WIN32
        fn = reinterpret_cast<F>(reinterpret_cast<void*>(GetProcAddress(library, name)));
#else
        fn = reinterpret_cast<F>(dlsym(library, name));
#endif
        return fn != nullptr;
    }

    void Unload() {
        if (library) {
#ifdef _WIN32
            FreeLibrary(library);
#else
            dlclose(library);
#endif
            library = nullptr;
        }
    }

#ifdef _WIN32
    HMODULE library = nullptr;
#else
    void* library = nullptr;
#endif
    bool initialized = false;
    Device device = nullptr;
    std::string device_name;

    int (*init)() = nullptr;
    int (*shutdown)() = nullptr;
    int (*get_count)(unsigned*) = nullptr;
    int (*get_handle)(unsigned, Device*) = nullptr;
    int (*get_name)(Device, char*, unsigned) = nullptr;
    int (*get_utilization)(Device, Utilization*) = nullptr;
    int (*get_memory)(Device, Memory*) = nullptr;
    int (*get_temperature)(Device, int, unsigned*) = nullptr;
    int (*get_clock)(Device, int, unsigned*) = nullptr;
    int (*get_power)(Device, unsigned*) = nullptr;
};
#endif

#ifdef _WIN32
u64 FileTimeToU64(const FILETIME& ft) {
    return (u64{ft.dwHighDateTime} << 32) | ft.dwLowDateTime;
}

std::string WideToUtf8(const wchar_t* text) {
    const int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1) {
        return {};
    }
    std::string result(static_cast<size_t>(size - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), size, nullptr, nullptr);
    return result;
}

template <typename T>
void Release(T*& object) {
    if (object) {
        object->Release();
        object = nullptr;
    }
}
#endif

u64 CurrentThreadId() {
#ifdef _WIN32
    return GetCurrentThreadId();
#elif defined(__linux__)
    return static_cast<u64>(syscall(SYS_gettid));
#else
    return 0;
#endif
}

#ifdef __linux__
std::string ReadFirstLine(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    std::getline(file, line);
    return line;
}

std::optional<u64> ReadNumber(const std::string& path) {
    const std::string line = ReadFirstLine(path);
    if (line.empty()) {
        return std::nullopt;
    }
    try {
        return std::stoull(line);
    } catch (...) {
        return std::nullopt;
    }
}
#endif

} // namespace

#ifdef _WIN32
/// CPU time of each thread of this process between two samples.
class ThreadSampler {
public:
    ThreadSampler() {
        // Windows 10 1607 and later.
        if (const HMODULE kernel_base = GetModuleHandleW(L"KernelBase.dll")) {
            get_description = reinterpret_cast<GetDescriptionFn>(
                reinterpret_cast<void*>(GetProcAddress(kernel_base, "GetThreadDescription")));
        }
    }

    ~ThreadSampler() {
        for (auto& [id, thread] : threads) {
            CloseHandle(thread.handle);
        }
    }

    void Sample(std::vector<ThreadUsage>& out, u64 ui_thread_id) {
        const auto now = std::chrono::steady_clock::now();
        const double wall_100ns = std::chrono::duration<double>(now - last_sample).count() * 1e7;
        const bool have_last = sampled && wall_100ns > 0.0;

        const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (snapshot == INVALID_HANDLE_VALUE) {
            return;
        }
        ++generation;
        const DWORD pid = GetCurrentProcessId();
        THREADENTRY32 entry{};
        entry.dwSize = sizeof(entry);
        for (BOOL more = Thread32First(snapshot, &entry); more;
             more = Thread32Next(snapshot, &entry)) {
            if (entry.th32OwnerProcessID != pid) {
                continue;
            }
            const auto it = threads.try_emplace(entry.th32ThreadID).first;
            Thread& thread = it->second;
            if (thread.handle) {
                // The id of a thread that exited can be handed to a new one.
                DWORD exit_code = 0;
                if (!GetExitCodeThread(thread.handle, &exit_code) || exit_code != STILL_ACTIVE) {
                    CloseHandle(thread.handle);
                    thread = {};
                }
            }
            if (!thread.handle) {
                thread.handle =
                    OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
                if (!thread.handle) {
                    continue; // dropped below: its generation stays behind
                }
            }
            FILETIME create_ft, exit_ft, kernel_ft, user_ft;
            if (!GetThreadTimes(thread.handle, &create_ft, &exit_ft, &kernel_ft, &user_ft)) {
                continue;
            }
            thread.generation = generation;
            if (thread.name.empty()) {
                // Threads name themselves after they start, so keep asking until one does.
                if (entry.th32ThreadID == ui_thread_id) {
                    thread.name = "UI";
                } else if (get_description) {
                    PWSTR description = nullptr;
                    if (SUCCEEDED(get_description(thread.handle, &description)) && description) {
                        thread.name = WideToUtf8(description);
                        LocalFree(description);
                    }
                }
            }
            const u64 cpu = FileTimeToU64(kernel_ft) + FileTimeToU64(user_ft);
            if (thread.seen && have_last && cpu >= thread.last_cpu) {
                out.push_back({thread.name, 100.0 * static_cast<double>(cpu - thread.last_cpu) /
                                                wall_100ns});
            }
            thread.seen = true;
            thread.last_cpu = cpu;
        }
        CloseHandle(snapshot);

        std::erase_if(threads, [this](auto& item) {
            Thread& thread = item.second;
            if (thread.generation == generation) {
                return false;
            }
            if (thread.handle) {
                CloseHandle(thread.handle);
            }
            return true;
        });
        last_sample = now;
        sampled = true;
    }

private:
    using GetDescriptionFn = HRESULT(WINAPI*)(HANDLE, PWSTR*);

    struct Thread {
        HANDLE handle = nullptr;
        std::string name;
        u64 last_cpu = 0;
        u64 generation = 0;
        bool seen = false;
    };

    GetDescriptionFn get_description = nullptr;
    std::unordered_map<DWORD, Thread> threads;
    std::chrono::steady_clock::time_point last_sample;
    u64 generation = 0;
    bool sampled = false;
};
#elif defined(__linux__)
/// CPU time of each thread of this process between two samples.
class ThreadSampler {
public:
    void Sample(std::vector<ThreadUsage>& out, u64 ui_thread_id) {
        const auto now = std::chrono::steady_clock::now();
        const double wall_ticks = std::chrono::duration<double>(now - last_sample).count() *
                                  static_cast<double>(sysconf(_SC_CLK_TCK));
        const bool have_last = sampled && wall_ticks > 0.0;

        DIR* const tasks = opendir("/proc/self/task");
        if (!tasks) {
            return;
        }
        ++generation;
        while (const dirent* entry = readdir(tasks)) {
            if (entry->d_name[0] < '0' || entry->d_name[0] > '9') {
                continue;
            }
            // "tid (name) state ..." with utime and stime as fields 14 and 15.
            const std::string stat =
                ReadFirstLine(std::string("/proc/self/task/") + entry->d_name + "/stat");
            const size_t open = stat.find('(');
            const size_t close = stat.rfind(')');
            if (open == std::string::npos || close == std::string::npos || close < open) {
                continue;
            }
            std::istringstream fields(stat.substr(std::min(close + 2, stat.size())));
            std::string skip;
            for (int i = 3; i < 14; ++i) {
                fields >> skip;
            }
            u64 utime = 0, stime = 0;
            if (!(fields >> utime >> stime)) {
                continue;
            }
            const u64 tid = std::strtoull(entry->d_name, nullptr, 10);
            Thread& thread = threads[tid];
            thread.generation = generation;
            const u64 cpu = utime + stime;
            if (thread.seen && have_last && cpu >= thread.last_cpu) {
                out.push_back({tid == ui_thread_id ? std::string("UI")
                                                   : stat.substr(open + 1, close - open - 1),
                               100.0 * static_cast<double>(cpu - thread.last_cpu) / wall_ticks});
            }
            thread.seen = true;
            thread.last_cpu = cpu;
        }
        closedir(tasks);

        std::erase_if(threads,
                      [this](const auto& item) { return item.second.generation != generation; });
        last_sample = now;
        sampled = true;
    }

private:
    struct Thread {
        u64 last_cpu = 0;
        u64 generation = 0;
        bool seen = false;
    };

    std::unordered_map<u64, Thread> threads;
    std::chrono::steady_clock::time_point last_sample;
    u64 generation = 0;
    bool sampled = false;
};
#endif

// -------------------------------------------------------------------------------------------------
// Platform backend. Created and used only on the monitor thread.
// -------------------------------------------------------------------------------------------------
struct SystemMonitor::Backend {
#ifdef _WIN32
    Backend() {
        OpenAdapter();
        // NVML only describes NVIDIA cards; a machine with an AMD or Intel GPU next to an NVIDIA
        // one must not report the NVIDIA card's numbers.
        constexpr UINT VendorNvidia = 0x10DE;
        if ((adapter == nullptr || vendor_id == VendorNvidia) && nvml.SelectDevice(gpu_name)) {
            if (gpu_name.empty()) {
                gpu_name = nvml.Name();
            }
        } else {
            OpenGpuCounters();
        }
    }

    ~Backend() {
        if (gpu_query) {
            PdhCloseQuery(gpu_query);
        }
        Release(adapter);
    }

    void OpenAdapter() {
        IDXGIFactory1* factory = nullptr;
        if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) {
            return;
        }
        // The emulator renders on the GPU with the most dedicated memory unless told otherwise,
        // and that is the one worth watching.
        SIZE_T best_memory = 0;
        IDXGIAdapter1* candidate = nullptr;
        for (UINT i = 0; factory->EnumAdapters1(i, &candidate) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            candidate->GetDesc1(&desc);
            IDXGIAdapter3* adapter3 = nullptr;
            const bool usable = !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                                desc.DedicatedVideoMemory > best_memory &&
                                SUCCEEDED(candidate->QueryInterface(
                                    __uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter3)));
            if (usable) {
                Release(adapter);
                adapter = adapter3;
                best_memory = desc.DedicatedVideoMemory;
                gpu_name = WideToUtf8(desc.Description);
                vram_total = desc.DedicatedVideoMemory;
                vendor_id = desc.VendorId;
                luid = desc.AdapterLuid;
            }
            Release(candidate);
        }
        Release(factory);
    }

    void OpenGpuCounters() {
        // Without NVML, Windows' own per-engine counters (what Task Manager shows) give the load.
        if (!adapter || PdhOpenQueryW(nullptr, 0, &gpu_query) != ERROR_SUCCESS) {
            gpu_query = nullptr;
            return;
        }
        if (PdhAddEnglishCounterW(gpu_query, L"\\GPU Engine(*)\\Utilization Percentage", 0,
                                  &gpu_counter) != ERROR_SUCCESS) {
            PdhCloseQuery(gpu_query);
            gpu_query = nullptr;
            return;
        }
        // Whole-GPU dedicated memory in use, which NVML would otherwise provide. Optional.
        if (PdhAddEnglishCounterW(gpu_query, L"\\GPU Adapter Memory(*)\\Dedicated Usage", 0,
                                  &vram_counter) != ERROR_SUCCESS) {
            vram_counter = nullptr;
        }
        PdhCollectQueryData(gpu_query);
        wchar_t prefix[64];
        swprintf(prefix, std::size(prefix), L"luid_0x%08lX_0x%08lX",
                 static_cast<unsigned long>(luid.HighPart),
                 static_cast<unsigned long>(luid.LowPart));
        luid_prefix = prefix;
    }

    /// Every instance of a wildcard counter, or an empty span.
    std::span<const PDH_FMT_COUNTERVALUE_ITEM_W> ReadCounter(PDH_HCOUNTER counter, DWORD format,
                                                             std::vector<std::byte>& buffer) {
        DWORD size = 0;
        DWORD count = 0;
        if (!counter ||
            PdhGetFormattedCounterArrayW(counter, format, &size, &count, nullptr) !=
                static_cast<PDH_STATUS>(PDH_MORE_DATA) ||
            size == 0) {
            return {};
        }
        buffer.resize(size);
        auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buffer.data());
        if (PdhGetFormattedCounterArrayW(counter, format, &size, &count, items) != ERROR_SUCCESS) {
            return {};
        }
        return {items, count};
    }

    void SampleGpuCounters(SystemStats& out) {
        if (!gpu_query || PdhCollectQueryData(gpu_query) != ERROR_SUCCESS) {
            return;
        }
        // Instances look like "pid_1234_luid_0x0_0xC8F3_phys_0_eng_3_engtype_3D". Add up every
        // process per engine of our adapter; the busiest engine is the GPU's load.
        const auto engines_items = ReadCounter(gpu_counter, PDH_FMT_DOUBLE, counter_buffer);
        if (!engines_items.empty()) {
            std::map<std::wstring_view, double> engines;
            for (const auto& item : engines_items) {
                const std::wstring_view name{item.szName};
                const size_t at = name.find(luid_prefix);
                if (at == std::wstring_view::npos || item.FmtValue.CStatus != ERROR_SUCCESS) {
                    continue;
                }
                engines[name.substr(at)] += item.FmtValue.doubleValue;
            }
            double busiest = 0.0;
            for (const auto& [engine, load] : engines) {
                busiest = std::max(busiest, load);
            }
            out.gpu_usage = std::clamp(busiest, 0.0, 100.0);
        }
        // Instances look like "luid_0x0_0xC8F3_phys_0", one per physical adapter.
        u64 vram = 0;
        bool found = false;
        for (const auto& item : ReadCounter(vram_counter, PDH_FMT_LARGE, vram_buffer)) {
            if (std::wstring_view{item.szName}.find(luid_prefix) != std::wstring_view::npos &&
                item.FmtValue.CStatus == ERROR_SUCCESS && item.FmtValue.largeValue >= 0) {
                vram += static_cast<u64>(item.FmtValue.largeValue);
                found = true;
            }
        }
        if (found) {
            out.vram_used = vram;
        }
    }

    void Sample(SystemStats& out) {
        out.gpu_name = gpu_name;

        FILETIME idle_ft, kernel_ft, user_ft;
        FILETIME create_ft, exit_ft, proc_kernel_ft, proc_user_ft;
        if (GetSystemTimes(&idle_ft, &kernel_ft, &user_ft) &&
            GetProcessTimes(GetCurrentProcess(), &create_ft, &exit_ft, &proc_kernel_ft,
                            &proc_user_ft)) {
            // Kernel time includes idle time; all of these are summed over every logical core.
            const u64 idle = FileTimeToU64(idle_ft);
            const u64 total = FileTimeToU64(kernel_ft) + FileTimeToU64(user_ft);
            const u64 process_kernel = FileTimeToU64(proc_kernel_ft);
            const u64 process = process_kernel + FileTimeToU64(proc_user_ft);
            if (last_total != 0 && process > last_process) {
                out.cpu_kernel =
                    std::clamp(100.0 * static_cast<double>(process_kernel - last_process_kernel) /
                                   static_cast<double>(process - last_process),
                               0.0, 100.0);
            }
            last_process_kernel = process_kernel;
            if (last_total != 0 && total > last_total) {
                const double d_total = static_cast<double>(total - last_total);
                out.cpu_system =
                    std::clamp(100.0 * (d_total - static_cast<double>(idle - last_idle)) / d_total,
                               0.0, 100.0);
                out.cpu_process = std::clamp(
                    100.0 * static_cast<double>(process - last_process) / d_total, 0.0, 100.0);
            }
            last_idle = idle;
            last_total = total;
            last_process = process;
        }

        PROCESS_MEMORY_COUNTERS_EX counters{};
        if (GetProcessMemoryInfo(GetCurrentProcess(),
                                 reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                                 sizeof(counters))) {
            out.ram_process = counters.WorkingSetSize;
        }
        MEMORYSTATUSEX memory{};
        memory.dwLength = sizeof(memory);
        if (GlobalMemoryStatusEx(&memory)) {
            out.ram_total = memory.ullTotalPhys;
            out.ram_used = memory.ullTotalPhys - memory.ullAvailPhys;
        }

        if (adapter) {
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
                out.vram_process = info.CurrentUsage;
            }
            if (vram_total) {
                out.vram_total = vram_total;
            }
        }

        thread_sampler.Sample(out.threads, ui_thread_id);

        nvml.Sample(out);
        SampleGpuCounters(out); // only set up when NVML is not in use
    }

    ThreadSampler thread_sampler;
    u64 ui_thread_id = 0;
    u64 last_process_kernel = 0;

    Nvml nvml;
    IDXGIAdapter3* adapter = nullptr;
    LUID luid{};
    UINT vendor_id = 0;
    std::string gpu_name;
    u64 vram_total = 0;

    PDH_HQUERY gpu_query = nullptr;
    PDH_HCOUNTER gpu_counter = nullptr;
    PDH_HCOUNTER vram_counter = nullptr;
    std::wstring luid_prefix;
    std::vector<std::byte> counter_buffer;
    std::vector<std::byte> vram_buffer;

    u64 last_idle = 0;
    u64 last_total = 0;
    u64 last_process = 0;

#elif defined(__linux__)
    Backend() {
        if (nvml.SelectDevice({})) {
            gpu_name = nvml.Name();
        } else {
            FindAmdGpu();
        }
    }

    void FindAmdGpu() {
        // amdgpu publishes its load and VRAM use in sysfs; take the card with the most VRAM.
        DIR* dir = opendir("/sys/class/drm");
        if (!dir) {
            return;
        }
        u64 best_vram = 0;
        while (const dirent* entry = readdir(dir)) {
            const std::string_view name{entry->d_name};
            if (name.substr(0, 4) != "card" || name.find('-') != std::string_view::npos) {
                continue;
            }
            const std::string device = "/sys/class/drm/" + std::string(name) + "/device";
            const auto vram = ReadNumber(device + "/mem_info_vram_total");
            if (vram && *vram > best_vram && ReadNumber(device + "/gpu_busy_percent")) {
                best_vram = *vram;
                amd_device = device;
            }
        }
        closedir(dir);
        if (amd_device.empty()) {
            return;
        }
        if (DIR* hwmon = opendir((amd_device + "/hwmon").c_str())) {
            while (const dirent* entry = readdir(hwmon)) {
                if (std::string_view{entry->d_name}.substr(0, 5) == "hwmon") {
                    amd_hwmon = amd_device + "/hwmon/" + entry->d_name;
                    break;
                }
            }
            closedir(hwmon);
        }
    }

    void Sample(SystemStats& out) {
        out.gpu_name = gpu_name;

        {
            std::ifstream proc_stat("/proc/stat");
            std::string cpu;
            u64 user = 0, nice = 0, kernel = 0, idle = 0, iowait = 0, irq = 0, softirq = 0,
                steal = 0;
            proc_stat >> cpu >> user >> nice >> kernel >> idle >> iowait >> irq >> softirq >> steal;
            const u64 idle_all = idle + iowait;
            const u64 total = user + nice + kernel + idle + iowait + irq + softirq + steal;

            // utime and stime are fields 14 and 15, after the parenthesised command name.
            u64 process = 0;
            const std::string self = ReadFirstLine("/proc/self/stat");
            if (const size_t close = self.rfind(')'); close != std::string::npos) {
                std::istringstream fields(self.substr(close + 2));
                std::string skip;
                for (int i = 3; i < 14; ++i) {
                    fields >> skip;
                }
                u64 utime = 0, stime = 0;
                fields >> utime >> stime;
                process = utime + stime;
                if (last_total != 0 && process > last_process && stime >= last_process_kernel) {
                    out.cpu_kernel =
                        std::clamp(100.0 * static_cast<double>(stime - last_process_kernel) /
                                       static_cast<double>(process - last_process),
                                   0.0, 100.0);
                }
                last_process_kernel = stime;
            }
            if (last_total != 0 && total > last_total) {
                const double d_total = static_cast<double>(total - last_total);
                out.cpu_system = std::clamp(
                    100.0 * (d_total - static_cast<double>(idle_all - last_idle)) / d_total, 0.0,
                    100.0);
                out.cpu_process = std::clamp(
                    100.0 * static_cast<double>(process - last_process) / d_total, 0.0, 100.0);
            }
            last_idle = idle_all;
            last_total = total;
            last_process = process;
        }

        {
            std::ifstream statm("/proc/self/statm");
            u64 size = 0, resident = 0;
            if (statm >> size >> resident) {
                out.ram_process = resident * static_cast<u64>(sysconf(_SC_PAGESIZE));
            }
            std::ifstream meminfo("/proc/meminfo");
            std::string key;
            u64 value = 0;
            std::string unit;
            std::optional<u64> total, available;
            while (meminfo >> key >> value >> unit) {
                if (key == "MemTotal:") {
                    total = value * 1024;
                } else if (key == "MemAvailable:") {
                    available = value * 1024;
                }
                if (total && available) {
                    break;
                }
            }
            if (total && available) {
                out.ram_total = *total;
                out.ram_used = *total - std::min(*total, *available);
            }
        }

        thread_sampler.Sample(out.threads, ui_thread_id);

        nvml.Sample(out);
        if (!amd_device.empty()) {
            if (const auto busy = ReadNumber(amd_device + "/gpu_busy_percent")) {
                out.gpu_usage = static_cast<double>(*busy);
            }
            out.vram_used = ReadNumber(amd_device + "/mem_info_vram_used");
            out.vram_total = ReadNumber(amd_device + "/mem_info_vram_total");
            if (!amd_hwmon.empty()) {
                if (const auto temp = ReadNumber(amd_hwmon + "/temp1_input")) {
                    out.gpu_temp_c = *temp / 1000.0;
                }
                if (const auto power = ReadNumber(amd_hwmon + "/power1_average")) {
                    out.gpu_power_w = *power / 1'000'000.0;
                }
            }
        }
    }

    ThreadSampler thread_sampler;
    u64 ui_thread_id = 0;
    u64 last_process_kernel = 0;

    Nvml nvml;
    std::string gpu_name;
    std::string amd_device;
    std::string amd_hwmon;
    u64 last_idle = 0;
    u64 last_total = 0;
    u64 last_process = 0;

#else
    void Sample(SystemStats&) {}
    u64 ui_thread_id = 0;
#endif
};

SystemMonitor::SystemMonitor()
    : ui_thread_id{CurrentThreadId()}, thread{[this](std::stop_token stop) { Run(stop); }} {}

SystemMonitor::~SystemMonitor() {
    thread.request_stop();
    wake.notify_all();
}

void SystemMonitor::SetActive(bool active_) {
    {
        std::scoped_lock lock{wake_mutex};
        active = active_;
    }
    wake.notify_all();
}

SystemStats SystemMonitor::Get() const {
    std::scoped_lock lock{stats_mutex};
    return stats;
}

void SystemMonitor::Run(std::stop_token stop) {
    Common::SetCurrentThreadName("PerfMonitor");
    std::unique_ptr<Backend> backend;
    while (!stop.stop_requested()) {
        {
            std::unique_lock lock{wake_mutex};
            if (!wake.wait(lock, stop, [this] { return active; })) {
                break;
            }
        }
        if (!backend) {
            backend = std::make_unique<Backend>();
            backend->ui_thread_id = ui_thread_id;
        }
        SystemStats sample;
        backend->Sample(sample);
        {
            std::scoped_lock lock{stats_mutex};
            stats = std::move(sample);
        }
        std::unique_lock lock{wake_mutex};
        wake.wait_for(lock, stop, 1s, [] { return false; });
    }
}
