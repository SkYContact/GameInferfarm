# -*- coding: utf-8 -*-
"""dbg_sample.py — poor man's sampling profiler（Windows）

用法:
  python dbg_sample.py <exe路径> <map路径> [采样秒数=12] [exe参数...]
  python dbg_sample.py --pid <pid> <map路径> [采样秒数=12] [主模块名]
  python dbg_sample.py --ygo-env ...   # 前置旗标：注入 09-30 战役环境档（见下）

spawn exe→每轮对全部线程 Suspend/GetThreadContext(RIP)/Resume→按 map 定符号→
热线程/热符号 top 汇总。出身=胜率坍缩案 09-30（工人 2.2ms/行盲区归因）的战役
仪器；采样内核通用，机器/乘客绑定的 env 注入收编为 --ygo-env 显式旗标（缺省
不注入=中立采样，可采任意 inferfarm 系 exe）。--pid 模式只读不杀：结束时
detach，附着进程保持运行。

map 生成：CMakeLists 对应目标临时加 target_link_options(/MAP) 重链即得；
模块枚举失败才回退 map 首选基址 0x140000000（ASLR 重定位则符号错位）。
"""
import ctypes, ctypes.wintypes as wt, re, sys, time, subprocess, os, bisect
from collections import Counter

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
TH32CS_SNAPTHREAD = 0x4
THREAD_GET_CONTEXT = 0x0008
THREAD_SUSPEND_RESUME = 0x0002
THREAD_QUERY_INFORMATION = 0x0040

class THREADENTRY32(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD),
                ("th32ThreadID", wt.DWORD), ("th32OwnerProcessID", wt.DWORD),
                ("tpBasePri", ctypes.c_long), ("tpDeltaPri", ctypes.c_long),
                ("dwFlags", wt.DWORD)]

def list_threads(pid):
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    if snap in (-1, 0):
        print("[sampler] 线程快照失败 gle=", k32.GetLastError(), flush=True)
        return []
    te = THREADENTRY32(); te.dwSize = ctypes.sizeof(te)
    tids = []
    if k32.Thread32First(snap, ctypes.byref(te)):
        while True:
            if te.th32OwnerProcessID == pid:
                tids.append(te.th32ThreadID)
            if not k32.Thread32Next(snap, ctypes.byref(te)):
                break
    k32.CloseHandle(snap)
    return tids

def module_base(pid, exe_name):
    """主模块基址。exe_name=None→取首模块（=主 exe）；给名→按 basename 匹配。"""
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    h = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h:
        print("[sampler] OpenProcess 失败 gle=", k32.GetLastError(), flush=True)
        return None
    arr = (ctypes.c_void_p * 256)()
    needed = wt.DWORD()
    if not psapi.EnumProcessModules(h, ctypes.byref(arr), ctypes.sizeof(arr), ctypes.byref(needed)):
        print("[sampler] EnumProcessModules 失败 gle=", k32.GetLastError(), flush=True)
        k32.CloseHandle(h); return None
    cnt = needed.value // ctypes.sizeof(wt.HMODULE)
    name = ctypes.create_unicode_buffer(512)
    base = None
    for i in range(cnt):
        psapi.GetModuleFileNameExW(h, wt.HMODULE(arr[i]), name, 512)
        if exe_name is None or os.path.basename(name.value).lower() == exe_name.lower():
            base = arr[i]
            break
    k32.CloseHandle(h)
    return base

# x64 CONTEXT=1232B；Rip@0xF8、ContextFlags@0x30；GetThreadContext 要求结构
# 16B 对齐——手工对齐缓冲 + from_buffer 视图（旧版靠 ctypes 堆分配恰好 16 对齐
# 的隐性侥幸，已显式接线）。
class CONTEXT(ctypes.Structure):
    _fields_ = [("buf", ctypes.c_uint64 * 156)]   # 1248B ≥ 1232B
    def set_flags(self):
        self.buf[6] = 0x10000B   # CONTEXT_FULL(amd64)=CONTROL|INTEGER|SEGMENTS
    def rip(self):
        return self.buf[0xF8 // 8]

def make_ctx():
    buf = ctypes.create_string_buffer(1280)
    off = (16 - (ctypes.addressof(buf) & 15)) & 15
    ctx = CONTEXT.from_buffer(buf, off)
    ctx.aligned_buf = buf   # from_buffer 已持引用；显式留一份防误回收
    return ctx

def sample_thread(tid, ctx):
    h = k32.OpenThread(THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME |
                       THREAD_QUERY_INFORMATION, False, tid)
    if not h: return None, "open"
    rip = None
    err = ""
    if k32.SuspendThread(h) != 0xFFFFFFFF:
        ctx.set_flags()
        if k32.GetThreadContext(h, ctypes.byref(ctx)):
            rip = ctx.rip()
            if rip == 0:
                err = "rip=0"
        else:
            err = "gtc gle=" + str(k32.GetLastError())
        k32.ResumeThread(h)
    else:
        err = "suspend"
    k32.CloseHandle(h)
    return rip, err

def load_map(mapf):
    syms = []
    for line in open(mapf, encoding="utf-8", errors="replace"):
        m = re.match(r"\s*0001:([0-9a-f]{8})\s+(\S+)\s+([0-9a-f]{16})", line)
        if m:
            syms.append((int(m.group(1), 16), m.group(2)))
    syms.sort()
    return syms

def sym_of(syms, rva):
    i = bisect.bisect_right(syms, (rva, "\x7f")) - 1
    return syms[i][1] if i >= 0 else "?"

def main():
    argv = sys.argv[1:]
    ygo_env = "--ygo-env" in argv
    if ygo_env:
        argv.remove("--ygo-env")
    if not argv:
        print(__doc__)
        return 1
    pid_mode = argv[0] == "--pid"
    proc = None
    if pid_mode:
        pid = int(argv[1]); mapf = argv[2]
        secs = float(argv[3]) if len(argv) > 3 else 12.0
        module = argv[4] if len(argv) > 4 else None   # 缺省=首模块（主 exe）
        time.sleep(0.2)
    else:
        exe = argv[0]; mapf = argv[1]
        secs = float(argv[2]) if len(argv) > 2 else 12.0
        module = os.path.basename(exe)
        exe_args = argv[3:]

    env = dict(os.environ)
    if ygo_env:
        # —— 09-30 胜率坍缩案战役档（机器绑定：q35 conda + CUDA13/CUDNN v9.12）——
        # YGO 腿环境（缺 cudnn/ORT PATH=DLL 入口点缺失秒死 0xC0000139）
        env.setdefault("FARM_TRT_DIR", "C:/Users/41601/Miniconda3/envs/q35/Lib/site-packages/tensorrt_libs")
        env.setdefault("FARM_CUDART_DLL", "cudart64_13.dll")
        env.setdefault("FARM_ORT_ASYNC", "3")
        env.setdefault("FARM_FIBER_WORKERS", "32")
        env.setdefault("YGO_OPP_SEED", "1")
        env.setdefault("YGO_ORT_THREADS", "1")
        env.setdefault("YGO_INFER_PIPE", "1")
        env.setdefault("OMP_NUM_THREADS", "8")
        env.setdefault("FARM_ORT_DIR", "C:/Users/41601/Miniconda3/envs/q35/Lib/site-packages/onnxruntime/capi")
        env["PATH"] = ("C:/Program Files/NVIDIA/CUDNN/v9.12/bin/13.0;"
                       "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64;") + env.get("PATH", "")

    if pid_mode:
        pass
    else:
        proc = subprocess.Popen([exe] + exe_args, env=env,
                                cwd=os.path.dirname(os.path.abspath(exe)),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pid = proc.pid
        time.sleep(3.0)   # 等过引擎/腿形起跑

    syms = load_map(mapf)
    print(f"[sampler] map 符号 {len(syms)} 条", flush=True)
    base = module_base(pid, module)
    if not base:
        base = 0x140000000
        print("[sampler] 模块枚举失败，回退 map 首选基址（ASLR 重定位则符号错位）", flush=True)
    print(f"[sampler] pid={pid} base={hex(base)}", flush=True)
    ctx = make_ctx()
    for _ in range(600):   # 等线程出现才开始计时
        if list_threads(pid): break
        time.sleep(0.1)
    t_end = time.time() + secs
    per_tid_rip = {}
    cyc = 0
    nfail = 0
    tids = []
    while time.time() < t_end:
        tids = list_threads(pid)
        for tid in tids:
            rip, err = sample_thread(tid, ctx)
            if rip:
                rva = rip - base
                if 0 <= rva < 0x1000000:   # 出界（内核/已卸载）样本丢弃
                    per_tid_rip.setdefault(tid, Counter())[rva] += 1
            elif err:
                if nfail < 3:
                    print(f"[sampler] tid={tid} 采样失败: {err}", flush=True)
                nfail += 1
        cyc += 1
    print(f"[sampler] 末轮 tids={len(tids)} 总失败={nfail}", flush=True)
    if proc:
        proc.kill()   # spawn 模式收尸
    else:
        print("[sampler] --pid 模式：只读不杀，附着进程保持运行", flush=True)
    # 汇总：按 tid 样本数排，top tid 各自 top 符号 + 全局 top
    g = Counter()
    rows = []
    for tid, c in per_tid_rip.items():
        n = sum(c.values())
        g.update(c)
        rows.append((n, tid, c))
    rows.sort(reverse=True)
    print(f"[sampler] 采样 {cyc} 轮, 线程 {len(per_tid_rip)} 个", flush=True)
    print("=== 热线程 top6（tid 样本数 | top 符号）===", flush=True)
    for n, tid, c in rows[:6]:
        tops = c.most_common(4)
        ss = " | ".join(f"{hex(rva)}={sym_of(syms, rva)} {cnt}" for rva, cnt in tops)
        print(f"tid={tid} n={n}: {ss}", flush=True)
    print("=== 全局 top20 符号 ===", flush=True)
    for rva, cnt in g.most_common(20):
        print(f"{cnt:6d}  {sym_of(syms, rva)}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
