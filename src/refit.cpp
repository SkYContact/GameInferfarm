// refit.cpp — RW1 blob 解析（ai_infer.cpp ParseRw1 的同源抽取）。
// 逐字节校验：magic/版本/越界/截断/尾部残余全部 fail fast。
#include "inferfarm/refit.h"
#include <chrono>
#include <cstdio>
#include <cstring>

namespace inferfarm {

static double RefitNowMs() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

size_t Rw1DtypeSize(uint8_t dtype) {
    static const uint32_t kDtypeSize[4] = {1, 2, 4, 8};   // int8/f16/f32/int64
    return dtype < 4 ? kDtypeSize[dtype] : 0;
}

bool ParseRw1(const std::string& path, std::vector<char>& blob,
              std::vector<Rw1Entry>& out) {
    double t0 = RefitNowMs();
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "[refit] 打不开 refit blob: %s\n", path.c_str());
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) {
        fclose(f);
        return false;
    }
    blob.resize((size_t)sz);
    size_t got = sz > 0 ? fread(blob.data(), 1, (size_t)sz, f) : 0;
    fclose(f);
    if (got != (size_t)sz || sz < 12) {
        std::fprintf(stderr, "[refit] blob 读取不完整/过短: %s（%zu/%ld 字节）\n",
                     path.c_str(), got, sz);
        return false;
    }
    const char* p = blob.data();
    const char* end = p + blob.size();
    if (std::memcmp(p, "RW1\x00", 4) != 0) {
        std::fprintf(stderr, "[refit] blob magic 不对（须 RW1\\0）: %s\n", path.c_str());
        return false;
    }
    uint32_t ver = 0, n = 0;
    memcpy(&ver, p + 4, 4);
    memcpy(&n, p + 8, 4);
    p += 12;
    if (ver != 1) {
        std::fprintf(stderr, "[refit] blob 版本 %u ≠ 1: %s\n", ver, path.c_str());
        return false;
    }
    out.reserve(n);
    bool bad = false;
    uint32_t i = 0;
    for (; i < n; i++) {
        uint16_t nl = 0;
        if (p + 2 > end) { bad = true; break; }
        memcpy(&nl, p, 2);
        p += 2;
        if (p + nl + 5 > end) { bad = true; break; }
        Rw1Entry e;
        e.name.assign(p, (size_t)nl);
        p += nl;
        e.dtype = (uint8_t)*p++;
        memcpy(&e.numel, p, 4);
        p += 4;
        if (e.dtype > 3) {
            std::fprintf(stderr, "[refit] 条目 %s dtype=%u ∉ {0,1,2,3}: %s\n",
                         e.name.c_str(), e.dtype, path.c_str());
            return false;
        }
        size_t ds = Rw1DtypeSize(e.dtype);
        e.bytes = (size_t)e.numel * ds;
        if (p + e.bytes > end) { bad = true; break; }
        e.data = p;
        p += e.bytes;
        out.push_back(std::move(e));
    }
    if (bad) {
        std::fprintf(stderr, "[refit] 条目 #%u 越界（文件截断/格式错）: %s\n", i, path.c_str());
        return false;
    }
    if (p != end) {
        std::fprintf(stderr, "[refit] 尾部残余 %td 字节（格式错？）: %s\n",
                     end - p, path.c_str());
        return false;
    }
    std::fprintf(stderr, "[refit] blob 解析: %u 项 %.1fMB（%.1fms）: %s\n",
                 n, blob.size() / 1e6, RefitNowMs() - t0, path.c_str());
    return true;
}

} // namespace inferfarm
