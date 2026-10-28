"""E2：分环境恒虚警（CFAR）约束训练（Inn 13.49）。

问题：E1（噪声增强）让网络在密噪声下能分开目标与噪声（AUC 大幅上升），但在固定阈值下，每个负事件的误报率
随环境上升（test_023 0.33% -> 0.55%）。逐事件 BCE 是 Bayes 准则（固定先验与代价），虚警率会随噪声变化；
环境会变时，正确的准则是 Neyman-Pearson / CFAR：在每一种环境里，虚警率都不超过同一个水平。

做法（只改训练目标，推理不变，表示层保持标准 12 通道）：
    环境量  每窗、每条流的随机巧合负荷 L = 25 x（本窗事件数 / 画面像素数），即 5x5 邻域内每窗的期望事件数；
            只用事件计数、不用标签。按 cfar_load_edges（默认 0.1 / 0.3 / 1.0，负荷 1 是巧合的物理分界）分层，
            第 0 层是安静环境。
    约束    对每个 s >= 1：FPR_s <= FPR_0（FPR = 负事件中 p >= 阈值 的比例，阈值取评估阈值 0.9）。
            即"各环境的虚警率都不超过安静环境"，不引入额外的虚警率常数。
    惩罚    sum_s mu_s * sum_{负事件 i in s} sigmoid((logit_i - logit(阈值)) / T) / 本次更新事件数
            （与 BCE 同一个"每事件"尺度；T 为平滑温度）。
    对偶    每次更新后：mu_s <- max(0, mu_s + eta * (FPR_s / FPR_0 - 1))，FPR 用硬计数的指数滑动平均。
cfar_dual_lr 未配置（或为 null）时返回 None：训练与原来逐位相同。
"""
import math

import numpy as np
import torch

DEFAULT_EDGES = (0.1, 0.3, 1.0)


class CfarConstraint(object):
    def __init__(self, dual_lr, threshold, height, width, edges=DEFAULT_EDGES, temperature=1.0, ema=0.99,
                 min_negatives=2000):
        if not (math.isfinite(dual_lr) and dual_lr > 0):
            raise ValueError("cfar_dual_lr 必须是有限正数")
        if not (0.0 < threshold < 1.0) or temperature <= 0 or not (0.0 < ema < 1.0):
            raise ValueError("cfar 参数不合法")
        edges = tuple(float(e) for e in edges)
        if list(edges) != sorted(edges) or not edges or edges[0] <= 0:
            raise ValueError("cfar_load_edges 必须是递增的正数")
        self.dual_lr, self.temperature, self.ema = float(dual_lr), float(temperature), float(ema)
        self.logit_thr = math.log(threshold / (1.0 - threshold))
        self.pixels = float(height * width)
        self.edges = edges
        self.n_strata = len(edges) + 1
        self.min_negatives = int(min_negatives)
        self.mu = np.zeros(self.n_strata)
        self.fp_ema = np.zeros(self.n_strata)
        self.neg_ema = np.zeros(self.n_strata)
        self.start_epoch()

    @classmethod
    def from_config(cls, cfg):
        lr = cfg.get("cfar_dual_lr")
        if lr is None:
            return None
        if cfg.get("trainer", "sequence") != "streams":
            raise ValueError("CFAR 约束训练只实现了 trainer: streams")
        return cls(float(lr), float(cfg["threshold"]), int(cfg["height"]), int(cfg["width_px"]),
                   edges=cfg.get("cfar_load_edges", DEFAULT_EDGES),
                   temperature=float(cfg.get("cfar_temperature", 1.0)),
                   ema=float(cfg.get("cfar_ema", 0.99)))

    def start_epoch(self):
        self._fp = np.zeros(self.n_strata)
        self._neg = np.zeros(self.n_strata)
        self._mu_hist, self._fpr_hist, self._mu_start = [], [], self.mu.copy()

    def strata(self, events, steps, batch):
        """每个事件所在窗的环境层号（long [N]）：按 (窗, 流) 统计事件数，换算成 5x5 巧合负荷后分层。"""
        key = events["t"].long() * int(batch) + events["b"].long()
        counts = torch.bincount(key, minlength=int(steps) * int(batch)).to(torch.float64)
        load = 25.0 * counts / self.pixels
        edges = torch.tensor(self.edges, dtype=torch.float64, device=load.device)
        return torch.bucketize(load, edges, right=True)[key]

    def chunk_loss(self, logits, labels, events, steps, batch, block_events):
        """一个片段的惩罚项（全部 mu = 0 时返回 None）；同时累计本次更新的硬计数。"""
        if logits.numel() == 0:
            return None
        stratum = self.strata(events, steps, batch)
        neg = labels < 0.5
        hard = (logits.detach() >= self.logit_thr) & neg
        self._neg += torch.bincount(stratum[neg], minlength=self.n_strata).cpu().numpy()
        self._fp += torch.bincount(stratum[hard], minlength=self.n_strata).cpu().numpy()
        if not np.any(self.mu > 0):
            return None
        soft = torch.sigmoid((logits - self.logit_thr) / self.temperature)
        weight = torch.as_tensor(self.mu, dtype=logits.dtype, device=logits.device)[stratum]
        return (weight * soft * neg.to(logits.dtype)).sum() / float(block_events)

    def end_update(self):
        """一次参数更新之后：更新硬计数的滑动平均，再对 s >= 1 做对偶上升。"""
        self.fp_ema = self.ema * self.fp_ema + (1.0 - self.ema) * self._fp
        self.neg_ema = self.ema * self.neg_ema + (1.0 - self.ema) * self._neg
        self._fp[:] = 0
        self._neg[:] = 0
        fpr = self.fpr()
        ref = fpr[0]
        if np.isfinite(ref) and ref > 0:
            for s in range(1, self.n_strata):
                if np.isfinite(fpr[s]) and self.neg_ema[s] * 100 >= self.min_negatives:   # 滑动平均约 100 次更新
                    self.mu[s] = max(0.0, self.mu[s] + self.dual_lr * (fpr[s] / ref - 1.0))
        self._mu_hist.append(self.mu.copy())
        self._fpr_hist.append(fpr)

    def fpr(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.neg_ema > 0, self.fp_ema / np.maximum(self.neg_ema, 1e-12), np.nan)

    def summary(self):
        """本轮统计（写入 metrics.jsonl 的 cfar 项）。"""
        fpr = self.fpr()
        return {"edges": list(self.edges), "temperature": self.temperature, "dual_lr": self.dual_lr,
                "fpr_end": [None if not np.isfinite(v) else float(v) for v in fpr],
                "mu_start": self._mu_start.tolist(), "mu_end": self.mu.tolist(),
                "mu_mean": np.mean(self._mu_hist, axis=0).tolist() if self._mu_hist else None,
                "neg_share": (self.neg_ema / max(self.neg_ema.sum(), 1e-12)).tolist()}
