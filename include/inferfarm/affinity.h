// ============================================================
//  inferfarm/affinity.h — 绑核（CPU affinity 旋钮，2026-09-24）
//
//  env（值格式共用）：
//    FARM_WORKER_AFFINITY   工人线程绑核列表（fiber 工人与线程模式链线程
//                           共用；FiberPool::RunLeg 开头解析）
//    FARM_SCHED_AFFINITY    银行调度台线程绑核列表（BankScheduler Init 解析）
//
//  值格式："0,1,4-7,phys"——逗号分隔的核号/闭区间（a<=b）；"phys"=物理核
//  代表号列表（GetLogicalProcessorInformationEx 每物理核取最低逻辑位——
//  SMT 负资产判决 2 的机器面：默认工人由 OS 自由落位，两个工人可挤同一
//  物理核）。线程按 id % 列表长 模分配（列表长==线程数即一一对应）。
//
//  边界：单处理器组语义，核号=全局逻辑核号（64×组号+组内位；本机 32T 单
//  组即朴素编号；>64 逻辑核的分组机器不适用=路标）。空/未设/整段全废=
//  不绑（默认零行为差）。绑核只改调度落位，不改任何算术与收割顺序——
//  逐位门全绿是必然（判决 17 spin 同款论证）。
// ============================================================
#pragma once
#include <vector>

namespace inferfarm {

// 解析列表；返回去重保序核号。段非法（非数字/区间反序）整段跳过（stderr
// 警告一条），合法段照收——实验旋钮软失败纪律，不因一处笔误弃整串。
std::vector<int> ParseCpuList(const char* spec);

// 当前线程绑到 cpus[id % cpus.size()]；空表=不动返回 false。失败=stderr
// 警告+false（没绑=默认调度，非危险态；核号越界/不在进程亲和集/系统拒绝）。
bool PinThread(const std::vector<int>& cpus, int id, const char* role);

// PinThread 成功总次数（探针；G15 门断言"真绑上"——防解析了没绑上的空过，
// R6 空过防线教训同源）。
int AffinityPinnedCount();

} // namespace inferfarm
