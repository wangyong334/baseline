"""导出原论文 EV-SpSegNet（离线 3D 稀疏卷积）的逐事件预测，用于和流式 V1 做独立核验。

只使用原仓库的模型 model/evspsegnet.py、数据集 dataset/ev_uav.py、体素化 collate 与 utils/eval.py，
不导入任何流式 V1 的代码。推理流程与原 test.py 逐行对应：
    net(voxel) -> preds[p2v_map] -> 逐事件概率
每个序列保存一个 NPZ（字段与流式 V1 的 --dump-dir 相同）：
    locs [batch,x,y,t] int64、labels float32、probabilities float32、target_id float64
并在最后用原 evalute 计算 IoU/ACC/Pd/Fa，数值应与原 test.py 的输出一致（seed37 test 集为 IoU 0.5894）。

用法:
    CUDA_VISIBLE_DEVICES=0 python tools/dump_baseline_predictions.py \
        --config configs/evisseg_evuav_baseline_v2_repolr.yaml \
        --checkpoint log/baseline_v2_repolr_seed37/best_iou_seed37.pt \
        --split test --out-dir log/verify/baseline_seed37_test
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    """解析本脚本自己的参数。必须在导入原仓库模块之前完成，因为 configs/configs.py 在导入时会解析命令行。"""
    parser = argparse.ArgumentParser(description="导出原 EV-SpSegNet 的逐事件预测")
    parser.add_argument("--config", required=True, help="原基线的 YAML 配置（提供数据根目录与网络宽度）")
    parser.add_argument("--checkpoint", required=True, help="原基线的权重文件")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main():
    """入口：加载原网络与权重 -> 按原 test.py 流程逐序列推理 -> 保存 NPZ -> 用原评估函数打印指标。"""
    args = parse_args()
    # 原 configs/configs.py 在 import 时调用 parse_args()，只认 --config，所以这里改写 sys.argv
    sys.argv = [sys.argv[0], "--config", args.config]

    import numpy as np
    import torch
    from configs.configs import cfg
    from dataset.ev_uav import EvUAV
    from model.evspsegnet import evspsegnet
    from utils.eval import evalute

    os.makedirs(args.out_dir, exist_ok=True)
    net = evspsegnet(cfg).eval().cuda()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cuda"))
    dataset = EvUAV(cfg, mode=args.split)
    if args.split != "train":
        assert dataset.mode != "train"          # 非训练模式不会随机降采样，事件与文件一一对应
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.custom_collate)
    evaluator = evalute(cfg)
    with torch.no_grad():
        for i, ev in enumerate(loader):
            name = dataset.file_list[i]
            label = ev["seg_label"].float()
            locs = ev["locs"]
            p2v_map = ev["p2v_map"].long().cuda()
            preds, _ = net(ev["voxel_ev"])
            preds = preds[p2v_map].squeeze(-1).cpu()
            np.savez(os.path.join(args.out_dir, name),
                     locs=locs.numpy().astype(np.int64),
                     labels=label.numpy().astype(np.float32),
                     probabilities=preds.numpy().astype(np.float32),
                     target_id=np.asarray(ev["idx_label"], dtype=np.float64))
            evaluator.matches[str(i)] = {"seg_pred": preds, "seg_gt": label.cuda()}
            ev_locs = locs.float()
            evaluator.roc_update(ev_locs[:, 3], preds, ev["idx_label"], label, ev_locs)
            print("[%d/%d] %s  事件数 %d" % (i + 1, len(dataset), name, label.numel()), flush=True)
    iou = evaluator.evaluate_semantic_segmantation_miou()
    acc = evaluator.evaluate_semantic_segmantation_accuracy()
    pd, fa = evaluator.cal_roc()
    print("原评估函数（与 test.py 相同调用）: iou:%s,seg_acc:%s,pd:%s,fa:%s" % (iou, acc, pd, fa))
    print("DUMP FINISHED:", args.out_dir)


if __name__ == "__main__":
    main()
