# -*- coding: utf-8 -*-
"""dbg_stack.py — 迷你崩溃取栈器（临时诊断工具，不入 CI）

CreateProcess(DEBUG_ONLY_THIS_PROCESS) 跑目标，首个异常时从 RSP 扫栈，
过滤出落在主模块映像内的返回地址，对照 .map 反查最近符号。
"""
import ctypes, ctypes.wintypes as wt, sys, re, struct

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
    exe = sys.argv[1]
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
    if not k32.CreateProcessW(exe, None, None, None, False,
                              DEBUG_ONLY_THIS_PROCESS, None, None,
                              ctypes.byref(si), ctypes.byref(pi)):
        print("CreateProcess fail", ctypes.get_last_error()); return 1
    ev = DEBUG_EVENT()
    image_base = None
    hproc = pi.hProcess; hthr = pi.hThread
    while True:
        if not k32.WaitForDebugEvent(ctypes.byref(ev), 10000):
            print("timeout, no crash"); break
        code = ev.dwDebugEventCode
        status = DBG_CONTINUE
        if code == EXCEPTION_DEBUG_EVENT:
            rec = ev.u.Exception
            if rec.ExceptionCode == EXCEPTION_CODE:
                print("STACK OVERFLOW at 0x%X" % rec.ExceptionAddress)
                ctx = CONTEXT(); ctx.ContextFlags = 0x100003  # FULL
                k32.GetThreadContext(hthr, ctypes.byref(ctx))
                print("RSP=0x%X RIP=0x%X" % (ctx.Rsp, ctx.Rip))
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
        k32.ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, status)
    k32.TerminateProcess(hproc, 1)
    return 0

if __name__ == "__main__":
    sys.exit(main())
