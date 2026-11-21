// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/hle/service/bcat/news/builtin_news.h"
#include "core/hle/service/bcat/news/msgpack.h"

#include "common/container/unordered_map.h"
#include "common/fs/path_util.h"
#include "common/logging.h"

#include <fmt/format.h>

#include <filesystem>
#include <functional>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>

// n2den does not fetch built-in news from any server: the release feed and the
// remote images are gone, so the news service starts with an empty local store.

namespace Service::News {
namespace {

::Common::unordered_map<std::string, std::vector<u8>> news_images_small;
::Common::unordered_map<std::string, std::vector<u8>> news_images_large;
std::mutex images_mutex;

std::filesystem::path GetNewsImagePath(std::string_view news_id, bool large) {
    const std::string filename = fmt::format("{}_{}.jpg", news_id, large ? "large" : "small");
    return Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir) / "news" / "images" / filename;
}

u32 HashToNewsId(std::string_view key) {
    return static_cast<u32>(std::hash<std::string_view>{}(key) & 0x7FFFFFFF);
}

std::vector<u8> TryLoadFromDisk(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) return {};

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};

    const auto file_size = static_cast<std::streamsize>(f.tellg());
    if (file_size <= 0 || file_size > 10 * 1024 * 1024) return {};

    f.seekg(0);
    std::vector<u8> data(static_cast<size_t>(file_size));
    if (!f.read(reinterpret_cast<char*>(data.data()), file_size)) return {};

    return data;
}

// Local cache only; nothing is downloaded.
std::vector<u8> GetNewsImage(std::string_view news_id, bool large) {
    const std::string id_str{news_id};

    {
        std::lock_guard lock{images_mutex};
        auto& cache = large ? news_images_large : news_images_small;
        if (auto it = cache.find(id_str); it != cache.end()) {
            return it->second;
        }
    }

    auto data = TryLoadFromDisk(GetNewsImagePath(news_id, large));

    {
        std::lock_guard lock{images_mutex};
        auto& cache = large ? news_images_large : news_images_small;
        cache[id_str] = data;
    }

    return data;
}

} // anonymous namespace

std::vector<u8> BuildMsgpack(std::string_view title, std::string_view body,
                             std::string_view topic_name, u64 published_at,
                             u64 pickup_limit, u32 priority,
                             const std::vector<std::string>& languages,
                             const std::string& author,
                             const std::vector<std::pair<std::string, std::string>>& /*assets*/,
                             const std::string& html_url,
                             std::optional<u32> override_id) {
    MsgPack::Writer w;

    const u32 news_id = override_id.value_or(HashToNewsId(title.empty() ? "n2den" : title));
    const std::string news_id_str = fmt::format("{}", news_id);

    const auto img_small = GetNewsImage(news_id_str, false);
    const auto img_large = GetNewsImage(news_id_str, true);

    w.WriteFixMap(23);

    // Version infos, could exist a 2?
    w.WriteKey("version");
    w.WriteFixMap(2);
    w.WriteKey("format");
    w.WriteUInt(1);
    w.WriteKey("semantics");
    w.WriteUInt(1);

    // Metadata
    w.WriteKey("news_id");
    w.WriteUInt(news_id);
    w.WriteKey("published_at");
    w.WriteUInt(published_at);
    w.WriteKey("pickup_limit");
    w.WriteUInt(pickup_limit);
    w.WriteKey("priority");
    w.WriteUInt(priority);
    w.WriteKey("deletion_priority");
    w.WriteUInt(100);

    // Language
    w.WriteKey("language");
    w.WriteString(languages.empty() ? "en" : languages.front());
    w.WriteKey("supported_languages");
    w.WriteFixArray(languages.size());
    for (const auto& lang : languages) w.WriteString(lang);

    // Display settings
    w.WriteKey("display_type");
    w.WriteString("NORMAL");
    w.WriteKey("topic_id");
    w.WriteString("n2den");

    w.WriteKey("no_photography"); // still show image
    w.WriteUInt(0);
    w.WriteKey("surprise"); // no idea
    w.WriteUInt(0);
    w.WriteKey("bashotorya"); // no idea
    w.WriteUInt(0);
    w.WriteKey("movie");
    w.WriteUInt(0); // 1 = has video, movie_url must be set but we don't support it yet

    // News Subject (Title)
    w.WriteKey("subject");
    w.WriteFixMap(2);
    w.WriteKey("caption");
    w.WriteUInt(1);
    w.WriteKey("text");
    w.WriteString(title.empty() ? "No title" : title);

    // Topic name = who wrote it
    w.WriteKey("topic_name");
    w.WriteString("n2den");

    w.WriteKey("list_image");
    w.WriteBinary(img_small);

    // Footer
    w.WriteKey("footer");
    w.WriteFixMap(1);
    w.WriteKey("text");
    w.WriteString("");

    w.WriteKey("allow_domains");
    w.WriteString("^$"); // no external domains

    // More link
    w.WriteKey("more");
    w.WriteFixMap(1);
    w.WriteKey("browser");
    w.WriteFixMap(2);
    w.WriteKey("url");
    w.WriteString(html_url);
    w.WriteKey("text");
    w.WriteString("Open link");

    // Body
    w.WriteKey("body");
    w.WriteFixMap(4);
    w.WriteKey("text");
    w.WriteString(body);
    w.WriteKey("main_image_height");
    w.WriteUInt(450);
    w.WriteKey("movie_url");
    w.WriteString("");
    w.WriteKey("main_image");
    w.WriteBinary(img_large);

    // no clue
    w.WriteKey("contents_descriptors");
    w.WriteString("");
    w.WriteKey("interactive_elements");
    w.WriteString("");

    return w.Take();
}

void EnsureBuiltinNewsLoaded() {
    static std::once_flag once;
    std::call_once(once, [] {
        LOG_INFO(Service_BCAT, "news: built-in news feed is disabled in n2den");
    });
}

} // namespace Service::News
