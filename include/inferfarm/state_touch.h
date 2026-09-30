// state_touch.h — 状态池触池注册表（跨后端公共面，2026-10-01 自 trt_backend
// 上收）。key=池 dev 指针——跨 backend 实例/跨后端类型（trt↔ort）同一把钥匙：
// 池共享形态下 ShareStatePool 只拷池指针，他组会话不在本实例会话表里，跨流
// 序在跨组形态从未兑现（DATA17 同源洞）。注册表把"读/写同一池"的会话连进
// 同一 wait 面，洞闭合。
//
// 纯主机数据面（无 CUDA 调用）——wait/record 的驱动语义归各后端（g_cu 可选
// 符号面各自守卫）。锁只护表结构；快照出的条目生命期由停机序保证（调度台/
// 收割腿全部 join 后才 DestroySession——与旧裸遍历同一不变量）。
#pragma once
#include <map>
#include <mutex>
#include <vector>

namespace inferfarm {

struct StTouchEntry {
    void* sess;      // 会话身份（去重/摘除键；后端各自解释）
    void* ev_state;  // 该会话的散射后 record 事件（wait 面）
};

inline std::mutex& StTouchMx() {
    static std::mutex mx;
    return mx;
}
inline std::map<void*, std::vector<StTouchEntry>>& StTouchMap_() {
    static std::map<void*, std::vector<StTouchEntry>> m;
    return m;
}

inline void StTouchAdd(void* pool_dev, void* sess, void* ev_state) {
    if (!pool_dev) return;
    std::lock_guard<std::mutex> lk(StTouchMx());
    auto& v = StTouchMap_()[pool_dev];
    for (auto& e : v)
        if (e.sess == sess) { e.ev_state = ev_state; return; }
    v.push_back({sess, ev_state});
}

inline void StTouchRemove(void* sess) {
    std::lock_guard<std::mutex> lk(StTouchMx());
    for (auto it = StTouchMap_().begin(); it != StTouchMap_().end();) {
        auto& v = it->second;
        for (size_t i = 0; i < v.size();)
            if (v[i].sess == sess) v.erase(v.begin() + (long)i);
            else i++;
        if (v.empty()) it = StTouchMap_().erase(it);
        else ++it;
    }
}

// 快照本会话触池的他席 wait 面（去重；调用方持快照逐个 StreamWaitEvent，
// EventQuery 探完成跳过——见 trt_backend SgWaitOthers 注）。返回条目数。
inline int StTouchOthers(void* sess, void* pool_dev,
                         StTouchEntry* out, int cap) {
    std::lock_guard<std::mutex> lk(StTouchMx());
    auto it = StTouchMap_().find(pool_dev);
    if (it == StTouchMap_().end()) return 0;
    int n = 0;
    for (auto& e : it->second) {
        if (e.sess == sess) continue;
        bool dup = false;
        for (int k = 0; k < n; k++)
            if (out[k].sess == e.sess) { dup = true; break; }
        if (!dup && n < cap) out[n++] = e;
    }
    return n;
}

} // namespace inferfarm
