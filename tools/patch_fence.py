# patch_fence.py — 给 fb onnx 图尾追加 InferfarmFence 节点（FARM_ORT_ASYNC=3 用）。
#
# 后端约定（src/backends/ort_backend.cpp 的 InferfarmFence custom op）：
#   - domain "inferfarm"、op 名 "InferfarmFence"、无输入
#   - 输出 __fence_out（float [slots]）挂 graph output——防图优化器剪枝
#     （哑输出，收割按请求目标名驱动，无人读它）
#   - 与后端输出校验兼容（f32 + dim0==slots），slots 取自原图任一 graph
#     output 的 dim0（fb 图全输出同 dim0）
#
# 用法：
#   python tools/patch_fence.py models/gomoku_mlp.fb8.onnx \
#       --out models/gomoku_mlp.fb8_fence.onnx
#
# 数学零变化：只加哑节点，与原模型逐位对拍（R6 门）是强验收。
# 导出后跑一次即可（模型级补丁，非每会话）。
import argparse
import sys

import onnx
from onnx import TensorProto, helper


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="原 fb onnx")
    ap.add_argument("--out", required=True, help="补丁后输出路径")
    ap.add_argument("--name", default="__fence_out", help="fence 输出名")
    args = ap.parse_args()

    m = onnx.load(args.src)

    # 幂等：已有 fence 节点即退出（防重复 patch）
    if any(n.op_type == "InferfarmFence" for n in m.graph.node):
        print(f"[patch_fence] {args.src} 已含 InferfarmFence 节点（幂等跳过）")
        return 0

    # slots 取自输入 dim0（fb 图输入必钉死；torch export 的输出 ValueInfo 常
    # 留符号 'batch'，但 ORT 建会话时按图推断具体化——fence 输出钉死数字与
    # 之同形）
    slots = None
    for i in m.graph.input:
        dims = i.type.tensor_type.shape.dim
        if dims and dims[0].dim_value:
            slots = dims[0].dim_value
            break
    if not slots:
        for o in m.graph.output:
            dims = o.type.tensor_type.shape.dim
            if dims and dims[0].dim_value:
                slots = dims[0].dim_value
                break
    if not slots:
        print("[patch_fence] 找不到 dim0（输入/输出均符号化）——非 fb 图？",
              file=sys.stderr)
        return 1
    if any(o.name == args.name for o in m.graph.output):
        print(f"[patch_fence] 输出名 {args.name} 已被占用", file=sys.stderr)
        return 1

    node = helper.make_node(
        "InferfarmFence", [], [args.name], name="__fence", domain="inferfarm")
    m.graph.node.append(node)
    m.graph.output.append(
        helper.make_tensor_value_info(args.name, TensorProto.FLOAT, [slots]))
    # custom op 域须在 opset_import 声明（查重后追加）
    if not any(i.domain == "inferfarm" for i in m.opset_import):
        m.opset_import.append(helper.make_opsetid("inferfarm", 1))

    onnx.checker.check_model(m)
    onnx.save(m, args.out)
    print(f"[patch_fence] 完成 slots={slots}：{args.src} -> {args.out}"
          f"（fence 输出={args.name}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
