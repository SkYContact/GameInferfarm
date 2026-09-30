# -*- coding: utf-8 -*-
"""dbg_stack.py — 迷你崩溃取栈器（临时诊断工具，不入 CI）

CreateProcess(DEBUG_ONLY_THIS_PROCESS) 跑目标，首个异常时从 RSP 扫栈，
过滤出落在主模块映像内的返回地址，对照 .map 反查最近符号。
"""
import ctypes, ctypes.wintypes as wt, sys, re, struct, time

k32 = ctypes.windll.kernel32
DEBUG_ONLY_THIS_PROCESS = 0x2
EXCEPTION_DEBUG_EVENT = 1
DBG_CONTINUE = 0x00010002
DBG_EXCEPTION_NOT_HANDLED = 0x80010001
EXCEPTION_CODE = 0xC00000FD  # 栈溢出

class STARTUPINFO(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("lpReserved", wt.LPWSTR), ("lpDesktop", wt.LPWSTR),
                ("lpTitle", wt.LPWSTR), ("dwX", wt.DWORD), ("dwY", wt.DWORD),
                ("dwXSize", wt.DWORD), ("dwYSize", wt.DWORD), ("dwXCountChars", wt.DWORD),
                ("dwYCountChars", wt.DWORD), ("dwFillAttribute", wt.DWORD),
                ("dwFlags", wt.DWORD), ("wShowWindow", wt.WORD), ("cbReserved2", wt.WORD),
                ("lpReserved2", ctypes.c_void_p), ("hStdInput", wt.HANDLE),
                ("hStdOutput", wt.HANDLE), ("hStdError", wt.HANDLE)]

class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", wt.HANDLE), ("hThread", wt.HANDLE),
                ("dwProcessId", wt.DWORD), ("dwThreadId", wt.DWORD)]

class EXCEPTION_RECORD64(ctypes.Structure):
    _fields_ = [("ExceptionCode", wt.DWORD), ("ExceptionFlags", wt.DWORD),
                ("ExceptionRecord", ctypes.c_uint64), ("ExceptionAddress", ctypes.c_uint64),
                ("NumberParameters", wt.DWORD), ("ExceptionInformation", ctypes.c_uint64 * 15)]

class DEBUG_EVENT(ctypes.Structure):
    class U(ctypes.Union):
        _fields_ = [("Exception", EXCEPTION_RECORD64), ("pad", ctypes.c_byte * 160)]
    _fields_ = [("dwDebugEventCode", wt.DWORD), ("dwProcessId", wt.DWORD),
                ("dwThreadId", wt.DWORD), ("u", U)]

class CONTEXT(ctypes.Structure):
    _fields_ = [("P1Home", ctypes.c_uint64), ("P2Home", ctypes.c_uint64),
                ("P3Home", ctypes.c_uint64), ("P4Home", ctypes.c_uint64),
                ("P5Home", ctypes.c_uint64), ("P6Home", ctypes.c_uint64),
                ("ContextFlags", wt.DWORD), ("MxCsr", wt.DWORD),
                ("SegCs", wt.WORD), ("SegDs", wt.WORD), ("SegEs", wt.WORD),
                ("SegFs", wt.WORD), ("SegGs", wt.WORD), ("SegSs", wt.WORD),
                ("EFlags", wt.DWORD), ("Dr0", ctypes.c_uint64), ("Dr1", ctypes.c_uint64),
                ("Dr2", ctypes.c_uint64), ("Dr3", ctypes.c_uint64), ("Dr6", ctypes.c_uint64),
                ("Dr7", ctypes.c_uint64), ("Rax", ctypes.c_uint64), ("Rcx", ctypes.c_uint64),
                ("Rdx", ctypes.c_uint64), ("Rbx", ctypes.c_uint64), ("Rsp", ctypes.c_uint64),
                ("Rbp", ctypes.c_uint64), ("Rsi", ctypes.c_uint64), ("Rdi", ctypes.c_uint64),
                ("R8", ctypes.c_uint64), ("R9", ctypes.c_uint64), ("R10", ctypes.c_uint64),
                ("R11", ctypes.c_uint64), ("R12", ctypes.c_uint64), ("R13", ctypes.c_uint64),
                ("R14", ctypes.c_uint64), ("R15", ctypes.c_uint64), ("Rip", ctypes.c_uint64),
                ("rest", ctypes.c_byte * 512)]

def main():
    # `--` 之后原样作为目标命令行（dbg_stack.py <exe> <map> [flags] -- <target args...>）
    argv = sys.argv[1:]
    tgt = []
    if "--" in argv:
        i = argv.index("--")
        tgt = argv[i + 1:]
        argv = argv[:i]
    exe = argv[0]
    mapf = sys.argv[2] if len(sys.argv) > 2 else None
    syms = []
    if mapf:
        for line in open(mapf, encoding="utf-8", errors="replace"):
            m = re.match(r"\s*0001:([0-9a-f]{8})\s+(\S+)\s+([0-9a-f]{16})", line)
            if m:
                syms.append((int(m.group(1), 16), m.group(2)))
        syms.sort()
    si = STARTUPINFO(); si.cb = ctypes.sizeof(si)
    pi = PROCESS_INFORMATION()
    cmdline = exe + (" " + " ".join('"%s"' % a for a in tgt) if tgt else "")
    if not k32.CreateProcessW(exe, cmdline, None, None, False,
                              DEBUG_ONLY_THIS_PROCESS, None, None,
                              ctypes.byref(si), ctypes.byref(pi)):
        print("CreateProcess fail", ctypes.get_last_error()); return 1
    ev = DEBUG_EVENT()
    image_base = None
    hproc = pi.hProcess; hthr = pi.hThread
    t0 = time.time()
    while True:
        if not k32.WaitForDebugEvent(ctypes.byref(ev), 5000):
            if time.time() - t0 > 150:
                print("timeout 150s, no crash"); break
            continue
        code = ev.dwDebugEventCode
        status = DBG_CONTINUE
        if code == EXCEPTION_DEBUG_EVENT:
            rec = ev.u.Exception
            if rec.ExceptionCode in (0xC0000005, 0xC0000409) and rec.ExceptionCode != EXCEPTION_CODE:
                print("EXC 0x%X at 0x%X (tid=%d)" % (rec.ExceptionCode, rec.ExceptionAddress, ev.dwThreadId))
                # 落入扫描分支（复用溢出路径的现场转储）
                ev.u.Exception.ExceptionCode = EXCEPTION_CODE
                rec = ev.u.Exception
            if rec.ExceptionCode == 0xE06D7363 and "--cpp" in sys.argv:
                print("CPP THROW at 0x%X (tid=%d)" % (rec.ExceptionAddress, ev.dwThreadId))
                k32.ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, DBG_CONTINUE)
                break
            if rec.ExceptionCode == EXCEPTION_CODE:
                print("STACK OVERFLOW at 0x%X (tid=%d)" % (rec.ExceptionAddress, ev.dwThreadId))
                # 关键：抓【出错线程】的上下文（fiber 跑在工人线程上，
                # 主线程 pi.hThread 的现场与本崩溃无关）
                THREAD_GET_CONTEXT = 0x0008
                THREAD_QUERY_INFORMATION = 0x0040
                ftid = ev.dwThreadId
                k32.OpenThread.restype = wt.HANDLE
                hthr = None
                for _ in range(50):
                    hthr = k32.OpenThread(THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                                          False, ftid)
                    if hthr:
                        break
                    time.sleep(0.01)
                ctx = CONTEXT(); ctx.ContextFlags = 0x100003  # FULL
                if not hthr or not k32.GetThreadContext(hthr, ctypes.byref(ctx)):
                    print("GetThreadContext fail err=%d hthr=%r" % (k32.GetLastError(), hthr))
                    k32.ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, DBG_CONTINUE)
                    break
                if not ctx.Rip:
                    print("ctx.Rip=0, retry"); k32.ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, DBG_CONTINUE); continue
                print("RSP=0x%X RIP=0x%X" % (ctx.Rsp, ctx.Rip))
                # RSP 落点定性：VirtualQueryEx 看分配区（线程栈 vs fiber 堆块）
                class MBI(ctypes.Structure):
                    _fields_ = [("BaseAddress", ctypes.c_uint64), ("AllocationBase", ctypes.c_uint64),
                                ("AllocationProtect", wt.DWORD), ("pad", wt.DWORD),
                                ("RegionSize", ctypes.c_uint64), ("State", wt.DWORD),
                                ("Protect", wt.DWORD), ("Type", wt.DWORD)]
                k32.VirtualQueryEx.restype = ctypes.c_size_t
                mbi = MBI()
                if k32.VirtualQueryEx(hproc, ctypes.c_void_p(ctx.Rsp), ctypes.byref(mbi), ctypes.sizeof(mbi)):
                    print("region: Base=0x%X AllocBase=0x%X Size=0x%X State=0x%X Type=0x%X depth_from_AllocBase=0x%X"
                          % (mbi.BaseAddress, mbi.AllocationBase, mbi.RegionSize, mbi.State, mbi.Type,
                             ctx.Rsp - mbi.AllocationBase))
                # RIP 现场机器码（第一现场字节，绕过 map 符号歧义）
                cbuf = (ctypes.c_char * 64)()
                cg = ctypes.c_size_t()
                if k32.ReadProcessMemory(hproc, ctypes.c_void_p(ctx.Rip - 32), cbuf, 64, ctypes.byref(cg)):
                    code = bytes(cbuf)
                    print("code @RIP-32:", code[:32].hex(' '))
                    print("code @RIP   :", code[32:].hex(' '))
                buf0 = (ctypes.c_char * 320)()
                got0 = ctypes.c_size_t()
                if k32.ReadProcessMemory(hproc, ctypes.c_void_p(ctx.Rsp), buf0, 320, ctypes.byref(got0)):
                    vals0 = struct.unpack("<40Q", bytes(buf0)[: 320])
                    for i, v in enumerate(vals0[:14]):
                        print("   raw[%d] 0x%X" % (i, v))
                # 读映像范围
                if image_base:
                    lo, hi = image_base, image_base + 0x1000000
                    got = ctypes.c_size_t()
                    found = 0
                    base = ctx.Rsp & ~0xFFF
                    for page in range(0, 0x800000, 0x1000):  # 向上最多 8MB
                        addr = base + page
                        buf = (ctypes.c_char * 0x1000)()
                        if not k32.ReadProcessMemory(hproc, ctypes.c_void_p(addr),
                                                     buf, 0x1000, ctypes.byref(got)):
                            continue
                        vals = struct.unpack("<512Q", bytes(buf)[: 0x1000])
                        for v in vals:
                            if lo <= v < hi:
                                off = v - image_base
                                name = ""
                                for a, s in syms:
                                    if a > off: break
                                    name = s
                                print("  +0x%-6X ret 0x%X (+0x%X) %s"
                                      % (addr + 8 * vals.index(v) - base, v, off, name[:100]))
                                found += 1
                                if found > 60: break
                        if found > 60: break
                    print("total ret hits:", found)
                k32.ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, DBG_CONTINUE)
                break
            status = DBG_EXCEPTION_NOT_HANDLED
        elif code == 3:  # CREATE_PROCESS_DEBUG_EVENT
            # CreateProcessInfo: hFile(8)+hProcess(8)+hThread(8)+lpBaseOfImage(8)
            image_base = ctypes.cast(ctypes.byref(ev.u, 24), ctypes.POINTER(ctypes.c_uint64)).contents.value
            print("image base 0x%X" % image_base)
        if code == 5:  # EXIT_PROCESS_DEBUG_EVENT
            ec = ctypes.cast(ctypes.byref(ev.u, 8), ctypes.POINTER(ctypes.c_uint64)).contents.value
            print("child exit code=0x%X" % ec)
            break
        k32.ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, status)
    k32.TerminateProcess(hproc, 1)
    return 0

if __name__ == "__main__":
    sys.exit(main())
