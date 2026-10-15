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


#include "PxgModulePath.h"

#include "foundation/PxPreprocessor.h"

#if PX_WINDOWS_FAMILY
#include "foundation/windows/PxWindowsInclude.h"
#include <cstdio>
#include <cstdlib>
#include <direct.h>
#include <process.h>
#else
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace physx
{
namespace gpuslang
{
namespace
{

// Any address inside this binary: the module lookup asks the OS which loaded
// image contains it.  (A data object, not a function: casting a function pointer
// to an object pointer is only conditionally supported.)
const char gModuleAnchor = 0;

#if PX_WINDOWS_FAMILY

std::wstring toWide(const std::string& s)
{
    const std::u16string u = pxgUtf8ToUtf16(s);
    return std::wstring(u.begin(), u.end()); // wchar_t is UTF-16 on Windows
}

std::string toUtf8(const wchar_t* w, size_t len)
{
    const std::u16string u(w, w + len);
    return pxgUtf16ToUtf8(u);
}

#endif

} // anonymous namespace

// UTF-8 <-> UTF-16 by hand: PxWindowsInclude.h defines NONLS, which hides
// MultiByteToWideChar / WideCharToMultiByte, and a raw <windows.h> is not allowed.
// Malformed, overlong or truncated UTF-8 sequences become U+FFFD (one per lead
// byte); an unpaired surrogate is encoded as its own code point, both ways.
std::u16string pxgUtf8ToUtf16(const std::string& s)
{
    std::u16string w;
    w.reserve(s.size());
    size_t i = 0;
    while (i < s.size())
    {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp;
        size_t extra;
        if (c < 0x80) { cp = c; extra = 0; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1Fu; extra = 1; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0Fu; extra = 2; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07u; extra = 3; }
        else { w.push_back(char16_t(0xFFFD)); ++i; continue; }
        bool ok = i + extra < s.size();
        for (size_t k = 1; ok && k <= extra; ++k)
        {
            const unsigned char cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80)
                ok = false;
            else
                cp = (cp << 6) | (cc & 0x3Fu);
        }
        // Reject overlong forms (C0 80 for NUL, E0 80 AF for '/') -- they would smuggle
        // characters past a path check.  Encoded surrogates (ED A0..BF xx) are kept, so
        // an unpaired surrogate from pxgUtf16ToUtf8 round-trips.
        static const uint32_t minForLength[4] = { 0x0u, 0x80u, 0x800u, 0x10000u };
        if (!ok || cp > 0x10FFFF || cp < minForLength[extra])
        {
            w.push_back(char16_t(0xFFFD));
            ++i;
            continue;
        }
        i += extra + 1;
        if (cp >= 0x10000)
        {
            cp -= 0x10000;
            w.push_back(char16_t(0xD800 + (cp >> 10)));
            w.push_back(char16_t(0xDC00 + (cp & 0x3FFu)));
        }
        else
            w.push_back(char16_t(cp));
    }
    return w;
}

std::string pxgUtf16ToUtf8(const std::u16string& w)
{
    std::string s;
    s.reserve(w.size());
    for (size_t i = 0; i < w.size(); ++i)
    {
        uint32_t cp = uint32_t(w[i]);
        if (cp >= 0xD800 && cp < 0xDC00 && i + 1 < w.size() && uint32_t(w[i + 1]) >= 0xDC00 && uint32_t(w[i + 1]) < 0xE000)
        {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (uint32_t(w[i + 1]) - 0xDC00);
            ++i;
        }
        if (cp < 0x80)
            s.push_back(char(cp));
        else if (cp < 0x800)
        {
            s.push_back(char(0xC0 | (cp >> 6)));
            s.push_back(char(0x80 | (cp & 0x3F)));
        }
        else if (cp < 0x10000)
        {
            s.push_back(char(0xE0 | (cp >> 12)));
            s.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(char(0x80 | (cp & 0x3F)));
        }
        else
        {
            s.push_back(char(0xF0 | (cp >> 18)));
            s.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
            s.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(char(0x80 | (cp & 0x3F)));
        }
    }
    return s;
}

std::string pxgModuleDirectory()
{
#if PX_WINDOWS_FAMILY
    HMODULE module = NULL;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&gModuleAnchor), &module))
        return std::string();
    // GetModuleFileNameW truncates silently (returning the buffer size) when the
    // path is longer than the buffer; grow until it fits.
    std::vector<wchar_t> buf(MAX_PATH);
    for (;;)
    {
        const DWORD n = GetModuleFileNameW(module, buf.data(), DWORD(buf.size()));
        if (n == 0)
            return std::string();
        if (n < buf.size())
        {
            size_t end = n;
            while (end > 0 && buf[end - 1] != L'\\' && buf[end - 1] != L'/')
                --end;
            return toUtf8(buf.data(), end);
        }
        if (buf.size() >= 32768) // the longest path Windows supports
            return std::string();
        buf.resize(buf.size() * 2);
    }
#else
    Dl_info info;
    if (!dladdr(&gModuleAnchor, &info) || !info.dli_fname)
        return std::string();
    const std::string path(info.dli_fname);
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? std::string() : path.substr(0, slash + 1);
#endif
}

bool pxgGetEnv(const char* name, std::string& value)
{
#if PX_WINDOWS_FAMILY
    const std::wstring wname = toWide(name);
    const wchar_t* w = _wgetenv(wname.c_str());
    if (!w)
        return false;
    size_t len = 0;
    while (w[len])
        ++len;
    value = toUtf8(w, len);
    return true;
#else
    const char* v = std::getenv(name);
    if (!v)
        return false;
    value = v;
    return true;
#endif
}

bool pxgReadFile(const std::string& path, std::vector<uint8_t>& data)
{
    data.clear();
#if PX_WINDOWS_FAMILY
    FILE* f = _wfopen(toWide(path).c_str(), L"rb");
#else
    FILE* f = std::fopen(path.c_str(), "rb");
#endif
    if (!f)
        return false;
    uint8_t chunk[16384];
    for (;;)
    {
        const size_t n = std::fread(chunk, 1, sizeof(chunk), f);
        data.insert(data.end(), chunk, chunk + n);
        if (n < sizeof(chunk))
            break;
    }
    const bool ok = !std::ferror(f) && !data.empty();
    std::fclose(f);
    if (!ok)
        data.clear();
    return ok;
}

std::string pxgUserCacheDirectory()
{
    std::string base;
#if PX_WINDOWS_FAMILY
    if (!pxgGetEnv("LOCALAPPDATA", base) || base.empty())
        return std::string();
    if (base[base.size() - 1] != '\\' && base[base.size() - 1] != '/')
        base += '\\';
    return base + "PhysXGpuVulkan\\";
#else
    if (pxgGetEnv("XDG_CACHE_HOME", base) && !base.empty() && base[0] == '/')
    {
        if (base[base.size() - 1] != '/')
            base += '/';
    }
    else if (pxgGetEnv("HOME", base) && !base.empty())
    {
        if (base[base.size() - 1] != '/')
            base += '/';
        base += ".cache/";
    }
    else
        return std::string();
    return base + "PhysXGpuVulkan/";
#endif
}

bool pxgMakeDirectories(const std::string& dir)
{
    if (dir.empty())
        return false;
    // Create each prefix that ends at a separator, then the whole path; "already
    // exists" is fine at every step, the final check decides.
    for (size_t i = 1; i <= dir.size(); ++i)
    {
        if (i < dir.size() && dir[i] != '/' && dir[i] != '\\')
            continue;
        const std::string prefix = dir.substr(0, i);
#if PX_WINDOWS_FAMILY
        if (prefix.size() == 2 && prefix[1] == ':')
            continue;   // a drive ("C:")
        _wmkdir(toWide(prefix).c_str());
#else
        mkdir(prefix.c_str(), 0755);
#endif
    }
#if PX_WINDOWS_FAMILY
    struct _stat64 st;
    std::string probe = dir;
    while (probe.size() > 3 && (probe[probe.size() - 1] == '\\' || probe[probe.size() - 1] == '/'))
        probe.erase(probe.size() - 1);
    return _wstat64(toWide(probe).c_str(), &st) == 0 && (st.st_mode & _S_IFDIR) != 0;
#else
    struct stat st;
    return stat(dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

bool pxgWriteFileReplace(const std::string& path, const std::vector<uint8_t>& data)
{
    if (path.empty() || data.empty())
        return false;
    char suffix[48];
#if PX_WINDOWS_FAMILY
    std::snprintf(suffix, sizeof(suffix), ".tmp%d", _getpid());
#else
    std::snprintf(suffix, sizeof(suffix), ".tmp%d", int(getpid()));
#endif
    const std::string tmp = path + suffix;
#if PX_WINDOWS_FAMILY
    FILE* f = _wfopen(toWide(tmp).c_str(), L"wb");
#else
    FILE* f = std::fopen(tmp.c_str(), "wb");
#endif
    if (!f)
        return false;
    const bool written = std::fwrite(data.data(), 1, data.size(), f) == data.size();
    const bool closed = std::fclose(f) == 0;
    bool ok = written && closed;
#if PX_WINDOWS_FAMILY
    if (ok)
    {
        // _wrename does not replace an existing file: remove it first.  Another process
        // may slip its own file in between; then this rename fails and its file stays.
        _wremove(toWide(path).c_str());
        ok = _wrename(toWide(tmp).c_str(), toWide(path).c_str()) == 0;
    }
    if (!ok)
        _wremove(toWide(tmp).c_str());
#else
    if (ok)
        ok = std::rename(tmp.c_str(), path.c_str()) == 0;   // atomic replace on POSIX
    if (!ok)
        std::remove(tmp.c_str());
#endif
    return ok;
}

} // namespace gpuslang
} // namespace physx
