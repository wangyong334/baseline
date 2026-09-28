"""流式 SNN V2（方案一）的损失函数（都不使用 pos_weight：类别不平衡由事件率本身表达）。

mark_loss       逐事件标签的二元交叉熵（给定事件位置时"是不是目标"的条件似然），训练 mark 头
boundary_mark_loss  V3 归属训练的边界负样本项（目标附近的背景事件；near_target_field 给出"目标附近"）
intensity_loss  当前窗目标强度场的泊松负对数似然 sum[ g - m*log g ]，m 为目标计数在 (s x s) 邻域内的平均
                （把"这一窗恰好落在哪几个像素"平滑成强度场；泊松负对数似然对非整数目标仍是均值的恰当评分，
                最优解 g = E[m | 特征]）。训练 log_g 头，CUSUM 把它按速度假设平移，作为下一窗的"可预测"目标强度。
两者都返回求和值，由调用者除以整条序列的事件数（与 V1 相同的归一化）。
"""
import torch.nn.functional as F


def mark_loss(mark_logits, labels):
    """逐事件 BCE（求和）。mark_logits、labels 均为 [N]；N=0 时返回 None。"""
    if labels.numel() == 0:
        return None
    return F.binary_cross_entropy_with_logits(mark_logits, labels, reduction="sum")


def near_target_field(target_counts, radius):
    """同一窗里离任何目标事件不超过 radius 像素（切比雪夫距离）的位置为 1。target_counts [T,B,1,H,W] -> 同形状 0/1。"""
    present = (target_counts > 0).to(target_counts.dtype)
    r = int(radius)
    if r <= 0:
        return present
    shape = present.shape
    flat = present.reshape((-1, 1) + tuple(shape[-2:]))
    return F.max_pool2d(flat, 2 * r + 1, stride=1, padding=r).view(shape)


def boundary_mark_loss(mark_logits, labels, near):
    """V3 归属训练的边界负样本项（求和口径，与 mark_loss 同量纲）。

    V3 的发布层里，mark（归属）决定一个事件能不能从等待中得到加分；实测等待新增的虚警与零等待的误检都集中在
    目标附近的背景事件上（"存在"的证据被分给了不属于目标的事件）。这些事件在全部背景里占比极小，普通 BCE 几乎不管它们，
    所以单列一项：目标附近背景事件的平均 BCE × 本片段的目标事件数——与正样本等总权重，不依赖数据集的事件比例。
    输入: mark_logits / labels [N]，near [N] 布尔（事件是否在同窗目标附近）。没有这类事件或没有目标事件时返回 None。
    """
    positive = labels > 0.5
    selected = near & ~positive
    n_positive = int(positive.sum())
    if n_positive == 0 or not bool(selected.any()):
        return None
    return F.binary_cross_entropy_with_logits(mark_logits[selected], labels[selected], reduction="mean") * n_positive


def smoothed_target(target_counts, smooth):
    """目标计数 [T,B,1,H,W] 在 (smooth x smooth) 邻域内的平均（边界按有效像素平均）；smooth=1 时原样返回。"""
    s = int(smooth)
    if s <= 1:
        return target_counts
    if s % 2 == 0:
        raise ValueError("intensity 平滑窗口必须是奇数")
    shape = target_counts.shape
    flat = target_counts.reshape((-1, 1) + tuple(shape[-2:]))
    return F.avg_pool2d(flat, s, stride=1, padding=s // 2, count_include_pad=False).view(shape)


def intensity_loss(log_g, target_counts, smooth=3):
    """当前窗目标强度场的泊松负对数似然（求和）。log_g 与 target_counts 形状相同 [T,B,1,H,W]。"""
    m = smoothed_target(target_counts, smooth)
    return (log_g.exp() - m * log_g).sum()
