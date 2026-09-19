"""在验证集上校准轨迹证据融合分数的权重，再固定用于测试集。

融合分数 = mark_logit + w * F(d) + b，mark_logit 与 F(d) 取自 `train_stream_v2.py --mode eval --dump-dir` 的导出
（字段 logit_net、evidence_d{d}）。每个延迟 d 单独拟合 (w, b)：在 val 全部事件上最小化逐事件 BCE（不加权），
w 限制为非负（证据只能按同一方向使用）。拟合完成后把 test 的校准概率写成新的导出目录（probabilities 字段），
可直接交给 tools/sweep_threshold.py 或 tools/verify_predictions.py；同时打印 val/test 在阈值 0.9 下的 IoU、召回、精确率。

用法:
    python tools/calibrate_fusion.py --val-dump log/verify/stream_v2_s37_val --test-dump log/verify/stream_v2_s37_test \
        --delays 1 2 5 --out-root log/verify/stream_v2_s37_test_cal
"""
import argparse
import json
import os

import numpy as np
import torch


def load(directory, delay):
    """读取一个导出目录：返回 [(文件名, 字典)]，字典含 logit、evidence、labels 与其余原字段。"""
    out = []
    for name in sorted(n for n in os.listdir(directory) if n.endswith(".npz")):
        with np.load(os.path.join(directory, name)) as data:
            item = {key: np.asarray(data[key]) for key in data.files}
        if "logit_net" not in item or ("evidence_d%d" % delay) not in item:
            raise SystemExit("%s 缺少 logit_net / evidence_d%d（请用新版 train_stream_v2.py 重新导出）" % (name, delay))
        out.append((name, item))
    return out


def fit(logit, evidence, labels, steps=200):
    """最小化 BCE(logit + w*F + b) 求 (w, b)，w >= 0（用 softplus 参数化）。"""
    x = torch.from_numpy(logit.astype(np.float64))
    f = torch.from_numpy(evidence.astype(np.float64))
    y = torch.from_numpy(labels.astype(np.float64))
    raw_w = torch.tensor(0.5413, dtype=torch.float64, requires_grad=True)     # softplus(0.5413) = 1
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw_w, b], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            x + torch.nn.functional.softplus(raw_w) * f + b, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.nn.functional.softplus(raw_w)), float(b)


def metrics(prob, labels, threshold=0.9):
    """阈值下的 IoU / 召回 / 精确率（全部事件拼接，与原口径一致）。"""
    pred, pos = prob >= threshold, labels == 1
    tp, fp, fn = int((pred & pos).sum()), int((pred & ~pos).sum()), int((~pred & pos).sum())
    return {"iou": tp / max(tp + fp + fn, 1), "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1)}


def main():
    parser = argparse.ArgumentParser(description="在 val 上校准融合权重并应用到 test")
    parser.add_argument("--val-dump", required=True)
    parser.add_argument("--test-dump", required=True)
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--out-root", required=True, help="每个延迟写一个目录 <out-root>_d{d}")
    args = parser.parse_args()
    report = {}
    for d in args.delays:
        val, test = load(args.val_dump, d), load(args.test_dump, d)
        cat = lambda items, key: np.concatenate([it[key] for _, it in items])  # noqa: E731
        w, b = fit(cat(val, "logit_net"), cat(val, "evidence_d%d" % d), cat(val, "labels"))
        row = {"weight": w, "bias": b}
        for split, items in (("val", val), ("test", test)):
            logit, evidence, labels = cat(items, "logit_net"), cat(items, "evidence_d%d" % d), cat(items, "labels")
            row[split + "_net"] = metrics(1.0 / (1.0 + np.exp(-logit)), labels)
            row[split + "_fused_w1"] = metrics(1.0 / (1.0 + np.exp(-(logit + evidence))), labels)
            row[split + "_fused_cal"] = metrics(1.0 / (1.0 + np.exp(-(logit + w * evidence + b))), labels)
        out_dir = "%s_d%d" % (args.out_root, d)
        os.makedirs(out_dir, exist_ok=True)
        for name, item in test:
            prob = 1.0 / (1.0 + np.exp(-(item["logit_net"] + w * item["evidence_d%d" % d] + b)))
            np.savez(os.path.join(out_dir, name), locs=item["locs"], labels=item["labels"],
                     probabilities=prob.astype(np.float32), target_id=item["target_id"])
        report["d%d" % d] = row
        print("d=%d  w=%.3f b=%.3f | val IoU net %.4f / 融合(w=1) %.4f / 校准 %.4f | test IoU net %.4f / 融合(w=1) %.4f / 校准 %.4f" % (
            d, w, b, row["val_net"]["iou"], row["val_fused_w1"]["iou"], row["val_fused_cal"]["iou"],
            row["test_net"]["iou"], row["test_fused_w1"]["iou"], row["test_fused_cal"]["iou"]), flush=True)
    with open(args.out_root + "_report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
