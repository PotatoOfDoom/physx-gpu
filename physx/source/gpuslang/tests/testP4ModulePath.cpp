// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

// The bridge's module-path helpers (bridge/PxgModulePath.cpp), which the official
// Vulkan PhysXGpu uses to find PhysXGpuShaders/ next to itself.  The UTF-8 <->
// UTF-16 converters only run on Windows in the product (PxWindowsInclude.h hides
// MultiByteToWideChar), so this is the one place they execute before a Windows
// build exists: every code point round-trips, known encodings match, and
// malformed or overlong input can neither yield NUL nor '/' nor '\'.

#include "pxgTestFramework.h"
#include "PxgModulePath.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace
{
using physx::gpuslang::pxgUtf8ToUtf16;
using physx::gpuslang::pxgUtf16ToUtf8;

std::u16string utf16Of(uint32_t cp)
{
    std::u16string s;
    if (cp >= 0x10000)
    {
        s.push_back(char16_t(0xD800 + ((cp - 0x10000) >> 10)));
        s.push_back(char16_t(0xDC00 + ((cp - 0x10000) & 0x3FF)));
    }
    else
        s.push_back(char16_t(cp));
    return s;
}

bool hasForbidden(const std::u16string& s)
{
    for (char16_t c : s)
        if (c == 0 || c == u'/' || c == u'\\')
            return true;
    return false;
}
} // namespace

PXG_TEST(p4ModulePathUtf)
{
    // Every code point, surrogates included (as lone units), survives UTF-16 -> UTF-8 -> UTF-16.
    uint32_t bad = 0;
    for (uint32_t cp = 0; cp <= 0x10FFFF; ++cp)
    {
        const std::u16string w = utf16Of(cp);
        if (pxgUtf8ToUtf16(pxgUtf16ToUtf8(w)) != w)
            ++bad;
    }
    PXG_CHECK_MSG(bad == 0, "%u code points do not round-trip", bad);

    // Known encodings.
    PXG_CHECK(pxgUtf16ToUtf8(u"\u00FC") == "\xC3\xBC");
    PXG_CHECK(pxgUtf16ToUtf8(u"\u20AC") == "\xE2\x82\xAC");
    PXG_CHECK(pxgUtf16ToUtf8(u"\U0001F600") == "\xF0\x9F\x98\x80");
    PXG_CHECK(pxgUtf8ToUtf16("C:/Programme/\xE6\x97\xA5\xE6\x9C\xAC") == u"C:/Programme/\u65E5\u672C");

    // Malformed input: never NUL / '/' / '\' out of a non-ASCII byte, always U+FFFD.
    const char* const malformed[] = {
        "\xC0\x80",             // overlong NUL
        "\xC0\xAF",             // overlong '/'
        "\xE0\x80\xAF",         // overlong '/'
        "\xF0\x80\x80\xAF",     // overlong '/'
        "\xC1\x9C",             // overlong '\'
        "\xF4\x90\x80\x80",     // > U+10FFFF
        "\xF8\x88\x80\x80\x80", // 5-byte form
        "\xC3",                 // truncated at the end
        "\xE2\x82",             // truncated at the end
        "\x80",                 // stray continuation byte
    };
    for (const char* m : malformed)
    {
        const std::u16string w = pxgUtf8ToUtf16(m);
        PXG_CHECK_MSG(!w.empty() && !hasForbidden(w) && w[0] == char16_t(0xFFFD), "malformed input decoded to %zu units", w.size());
    }
    // A truncated sequence does not swallow the next character.
    PXG_CHECK(pxgUtf8ToUtf16("\xC3z") == std::u16string(u"\uFFFDz"));
}

PXG_TEST(p4ModulePathFiles)
{
    using physx::gpuslang::pxgModuleDirectory;
    using physx::gpuslang::pxgGetEnv;
    using physx::gpuslang::pxgReadFile;

    // The directory of the binary this is linked into: gpuslang-tests itself.
    const std::string dir = pxgModuleDirectory();
    PXG_CHECK_MSG(!dir.empty() && dir.back() == '/', "module directory '%s'", dir.c_str());
    std::vector<uint8_t> self;
    PXG_CHECK_MSG(pxgReadFile(dir + "gpuslang-tests", self) && self.size() > 4 && self[0] == 0x7F && self[1] == 'E',
                  "the module directory holds the running binary");

    const std::string tmp = dir + "p4ModulePath.tmp";
    std::vector<uint8_t> data;
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    PXG_CHECK(f != nullptr);
    if (f)
        std::fclose(f);
    PXG_CHECK_MSG(!pxgReadFile(tmp, data) && data.empty(), "an empty file is refused");
    f = std::fopen(tmp.c_str(), "wb");
    if (f)
    {
        std::vector<uint8_t> big(40000);
        for (size_t i = 0; i < big.size(); ++i)
            big[i] = uint8_t(i * 7u);
        std::fwrite(big.data(), 1, big.size(), f);
        std::fclose(f);
        PXG_CHECK_MSG(pxgReadFile(tmp, data) && data == big, "a file larger than one read chunk is read whole");
    }
    std::remove(tmp.c_str());
    PXG_CHECK_MSG(!pxgReadFile(tmp, data), "a missing file is refused");

    std::string v;
    setenv("PXG_P4_MODULE_PATH_ENV", "a/b", 1);
    PXG_CHECK(pxgGetEnv("PXG_P4_MODULE_PATH_ENV", v) && v == "a/b");
    unsetenv("PXG_P4_MODULE_PATH_ENV");
    PXG_CHECK(!pxgGetEnv("PXG_P4_MODULE_PATH_ENV", v));
}

// The pipeline-cache file helpers: the per-user cache directory (XDG_CACHE_HOME, then
// HOME/.cache), nested directory creation, and replacing a file through a temporary
// so a reader never sees a half-written cache.
PXG_TEST(p4ModulePathCacheFiles)
{
    using physx::gpuslang::pxgUserCacheDirectory;
    using physx::gpuslang::pxgMakeDirectories;
    using physx::gpuslang::pxgWriteFileReplace;
    using physx::gpuslang::pxgReadFile;

    std::string savedXdg, savedHome;
    const bool hadXdg = physx::gpuslang::pxgGetEnv("XDG_CACHE_HOME", savedXdg);
    const bool hadHome = physx::gpuslang::pxgGetEnv("HOME", savedHome);

    setenv("XDG_CACHE_HOME", "/tmp/pxgXdgCache", 1);
    PXG_CHECK_MSG(pxgUserCacheDirectory() == "/tmp/pxgXdgCache/PhysXGpuVulkan/", "XDG_CACHE_HOME wins (%s)",
                  pxgUserCacheDirectory().c_str());
    setenv("XDG_CACHE_HOME", "relative/dir", 1);   // the XDG spec ignores a relative value
    setenv("HOME", "/home/pxgUser", 1);
    PXG_CHECK_MSG(pxgUserCacheDirectory() == "/home/pxgUser/.cache/PhysXGpuVulkan/", "HOME/.cache fallback (%s)",
                  pxgUserCacheDirectory().c_str());
    unsetenv("XDG_CACHE_HOME");
    unsetenv("HOME");
    PXG_CHECK_MSG(pxgUserCacheDirectory().empty(), "no base directory -> empty");
    if (hadXdg)
        setenv("XDG_CACHE_HOME", savedXdg.c_str(), 1);
    if (hadHome)
        setenv("HOME", savedHome.c_str(), 1);

    std::string base = physx::gpuslang::pxgModuleDirectory();
    if (base.empty())
        base = "./";
    const std::string dir = base + "p4CacheDir/a/b/";
    PXG_CHECK_MSG(pxgMakeDirectories(dir), "nested directories are created");
    PXG_CHECK_MSG(pxgMakeDirectories(dir), "existing directories are fine");
    const std::string file = dir + "cache.bin";
    std::vector<uint8_t> one(70000), two(123), got;
    for (size_t i = 0; i < one.size(); ++i)
        one[i] = uint8_t(i * 13u);
    for (size_t i = 0; i < two.size(); ++i)
        two[i] = uint8_t(255 - i);
    PXG_CHECK_MSG(pxgWriteFileReplace(file, one) && pxgReadFile(file, got) && got == one, "write + read back");
    PXG_CHECK_MSG(pxgWriteFileReplace(file, two) && pxgReadFile(file, got) && got == two,
                  "an existing file is replaced, not appended to or merged");
    PXG_CHECK_MSG(!pxgWriteFileReplace(base + "p4CacheDir/missing/x.bin", two), "a missing directory fails");
    PXG_CHECK_MSG(!pxgWriteFileReplace(file, std::vector<uint8_t>()), "empty data is refused");
    std::remove(file.c_str());
    std::remove((base + "p4CacheDir/a/b").c_str());
    std::remove((base + "p4CacheDir/a").c_str());
    std::remove((base + "p4CacheDir").c_str());
}
