"""交叉拟合的样本外骨干输出（RCL v4.1；堆叠泛化 Wolpert 1992 / Breiman 1996：第二级模型要用第一级模型没见过的数据上的
输出来训练）。

为什么：模块三（判决层）叠在冻结的 V5 上，若用 V5 在自己训练录像上的输出训练，学到的是 V5 的样本内错误——普通录像
召回 0.998（val 0.991）、漏检里整个目标没检出的只占 15%（val 47%）、极密录像召回 0.93（val 0.48）——于是"未来帮助救回"
在训练里几乎用不上，越训越不救（本地对照见 Stream-SNN-Innovations.md 13.84）。

做法：train_stream_v1.py --cv-folds K --cv-fold k --cv-block b 训练 K 个骨干（与 V5 同一配方，各自没见过第 k 折）；
本工具用第 k 个骨干对第 k 折录像逐层流式推理（与 train_track_head.BaseOutputs 同一代码），存每个事件的 z0 与 dec1 特征。
所有训练录像各出现一次。判决层训练时由配置 train_outputs 读入；验证 / 测试仍用完整骨干（base_checkpoint）。

用法：python tools/export_oof_outputs.py --config configs/changepoint_v41.yaml \
          --checkpoints log/v5_cv4_f0_seed37/best_val_iou_seed37.pt ... log/v5_cv4_f3_seed37/best_val_iou_seed37.pt \
          --out run/oof_v5_cv4.pt
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import train_stream_v1 as T  # noqa: E402
import train_track_head as TH  # noqa: E402
from utils.stream_common import load_flat_config  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="判决层配置（数据根目录、完整骨干 base_checkpoint）")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="按折号 0..K-1 排列的折骨干 checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    cfg = load_flat_config(args.config)
    _, _, full_base, full_q99, full_w, _ = TH.load_base(cfg, device)
    names = T.train_file_names(dict(full_base, cv_folds=0))
    table, manifest = {}, {"folds": [], "full_checkpoint": cfg["base_checkpoint"]}
    folds = None
    for k, path in enumerate(args.checkpoints):
        net, tap, base, q99, w, ckpt = TH.load_base(dict(cfg, base_checkpoint=path), device)
        cv = ckpt["config"]
        K, fold, block = int(cv.get("cv_folds", 0) or 0), int(cv.get("cv_fold", -1)), int(cv.get("cv_block", 1))
        if K != len(args.checkpoints) or fold != k:
            raise RuntimeError("%s 是第 %d / %d 折的骨干，但按位置应是第 %d / %d 折" % (path, fold, K, k, len(args.checkpoints)))
        if not np.allclose(q99, full_q99) or abs(w - full_w) > 1e-9:
            raise RuntimeError("%s 的输入编码 q99 / 正样本权重与完整骨干不同：z0 的含义不一致" % path)
        assign = T.crossfit_folds(names, K, block)
        if folds is None:
            folds = assign
        elif assign != folds:
            raise RuntimeError("各折骨干的分折规则不一致（cv_block 不同）")
        held = [n for n in names if assign[n] == k]
        ds = TH.EvUAVStream(base["root"], "train", base, names=held)
        out = TH.BaseOutputs(net, tap, base, q99, device, ds, keep_all=False)
        t0 = time.perf_counter()
        for i in range(len(ds)):
            seq, z0, feat = out.get(i)
            table[ds.names[i]] = (z0.float().cpu(), feat.float().cpu())
        manifest["folds"].append({"fold": k, "checkpoint": path, "sha256": sha256(path), "held_out": held})
        print("第 %d 折：%s 推理留出的 %d 条录像（%.0f s）" % (k, path, len(held), time.perf_counter() - t0), flush=True)
        del net, tap, out
        if device.type == "cuda":
            torch.cuda.empty_cache()
    missing = sorted(set(names) - set(table))
    if missing:
        raise RuntimeError("这些训练录像没有样本外输出：%s" % missing[:5])
    manifest.update(n_recordings=len(table), n_events=int(sum(int(v[0].shape[0]) for v in table.values())))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"outputs": table, "manifest": manifest}, args.out)
    print("样本外输出：%d 条录像、%d 个事件 -> %s" % (manifest["n_recordings"], manifest["n_events"], args.out), flush=True)


if __name__ == "__main__":
    main()
