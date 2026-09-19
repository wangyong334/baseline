"""流式 SNN V2（方案一）的事件来源与证据前端（纯 PyTorch，兼容 torch 1.9；无可学习参数）。

EventChunkSource   整条序列的事件一次性放到设备上，按 TBPTT 片段累加出每窗的
                   正/负极性计数、目标事件计数（强度头的训练目标）、各时间尺度的"新事件"指数权重和（时间矩的递推）。
EvidenceFrontEnd   逐窗递推的前端，输出网络输入特征、只用过去估计的背景强度 mu0 与本窗事件数：
    背景强度（每像素每窗期望事件数，只用第 k 窗之前的计数，供 CUSUM 做"可预测"的补偿）：
        带先验伪计数的指数平均  mu = (S + n0*prior) / (W + n0)，S <- b*S + N，W <- b*W + 1
        mu0 = max( floor, 慢速逐像素估计, 快速估计的 (2r+1)^2 邻域平均 )
        慢速逐像素项负责热像素；快速项取大邻域平均，跟得上成片杂波，又不会被几个像素大小的目标自己抬高
    时间矩（每个时间尺度 tau、每个极性，事件时间精确到毫秒，不分箱）：
        A = sum_e exp(-age_e/tau)，B = sum_e exp(-age_e/tau)*age_e，age 为事件到窗末的时长（ms）
        A(k) = d*A(k-1) + a_new，B(k) = d*(B(k-1) + dt*A(k-1)) + b_new，d = exp(-dt/tau)
    特征（全部与背景强度成比例归一化，没有事件的位置为 0）：
        count    log1p( 本窗该极性计数 / (mu0/2) )                                  2 通道
        ratio    log1p( A / (mu0*tau/(2*dt)) )  背景下的期望值为 1，即"比平时多几倍"     2K 通道
        age      min( B/(A+eps)/tau, 3 )         事件的平均年龄（以 tau 为单位）          2K 通道
        dipole   ON 质心 - OFF 质心（局部 (2R+1)^2 邻域，按两极性事件量加权），/R          2*len(dipole_taus) 通道
                 运动的小目标前沿为一种极性、后沿为另一种，偶极方向就是运动轴；噪声与闪烁的偶极期望为 0
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EventChunkSource(object):
    """事件常驻设备的数据来源（V2）。

    chunk(start, end) 返回字典:
        counts        [T,1,2,H,W]   正(0)/负(1)极性计数
        target_counts [T,1,1,H,W]   目标事件计数（标签为 1 的事件）
        moment_a/b    [T,1,2K,H,W]  各时间尺度 exp(-age/tau) 与 exp(-age/tau)*age 的新事件和
        events        {t,b,y,x,p,t_local}，t 为片段内的窗序号
        labels        [N]           逐事件标签
        idx           numpy [N]     原始事件下标（回填用）
    事件按窗口先后、窗内按时间稳定排序排列（与 V1 的 TorchWindowSource 相同）。
    """

    def __init__(self, seq, cfg, device, dtype=torch.float32):
        self.seq, self.device, self.dtype = seq, device, dtype
        self.height, self.width = int(cfg["pad_height"]), int(cfg["pad_width"])
        self.window_ms = float(cfg["window_ms"])
        self.taus = [float(t) for t in cfg["fe_taus_ms"]]
        self.plane = self.height * self.width
        self.order, self.bounds = seq.order, seq.bounds
        order = seq.order
        window = np.repeat(np.arange(seq.n_windows, dtype=np.int64), np.diff(seq.bounds))
        x, y = seq.x[order].astype(np.int64), seq.y[order].astype(np.int64)
        negative = (seq.p[order] == 0).astype(np.int64)                   # 正极性 -> 0，负极性 -> 1
        age = (window + 1).astype(np.float64) * self.window_ms - seq.t[order].astype(np.float64)

        def upload(array, tensor_dtype):
            return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=tensor_dtype)

        self.window_of = upload(window, torch.long)
        self.pixel = upload(y * self.width + x, torch.long)
        self.negative = upload(negative, torch.long)
        self.x, self.y = upload(x, torch.long), upload(y, torch.long)
        self.p = upload(seq.p[order].astype(np.float32), torch.float32)
        self.t_local = upload(seq.t_local[order], torch.float32)
        self.label = upload(seq.label[order], torch.float32)
        self.age = upload(age, dtype)

    def _accumulate(self, keys, values, size):
        """index_put_(accumulate=True) 累加（torch 1.9 确定性模式下可用，不用 scatter_add_/bincount）。"""
        out = torch.zeros(size, dtype=self.dtype, device=self.device)
        if keys.numel():
            out.index_put_((keys,), values.to(self.dtype), accumulate=True)
        return out

    def chunk(self, start, end):
        start, end = int(start), int(end)
        steps = end - start
        if steps <= 0:
            raise ValueError("片段至少需要 1 个窗口")
        a, b = int(self.bounds[start]), int(self.bounds[end])
        plane, H, W = self.plane, self.height, self.width
        rel = self.window_of[a:b] - start
        pix, neg = self.pixel[a:b], self.negative[a:b]
        ones = torch.ones(b - a, dtype=self.dtype, device=self.device)
        counts = self._accumulate((rel * 2 + neg) * plane + pix, ones, steps * 2 * plane)
        target_counts = self._accumulate(rel * plane + pix, self.label[a:b], steps * plane).view(steps, 1, 1, H, W)
        K = len(self.taus)
        keys, wa, wb = [], [], []
        age = self.age[a:b]
        for j, tau in enumerate(self.taus):
            w = torch.exp(-age / tau)
            keys.append((rel * (2 * K) + 2 * j + neg) * plane + pix)
            wa.append(w)
            wb.append(w * age)
        keys = torch.cat(keys)
        moment_a = self._accumulate(keys, torch.cat(wa), steps * 2 * K * plane).view(steps, 1, 2 * K, H, W)
        moment_b = self._accumulate(keys, torch.cat(wb), steps * 2 * K * plane).view(steps, 1, 2 * K, H, W)
        events = {"t": rel, "b": torch.zeros(b - a, dtype=torch.long, device=self.device),
                  "y": self.y[a:b], "x": self.x[a:b], "p": self.p[a:b].to(self.dtype),
                  "t_local": self.t_local[a:b].to(self.dtype)}
        return {"counts": counts.view(steps, 1, 2, H, W), "target_counts": target_counts,
                "moment_a": moment_a, "moment_b": moment_b, "events": events,
                "labels": self.label[a:b].to(self.dtype), "idx": self.order[a:b]}


def offset_kernels(radius, dtype=torch.float32):
    """局部邻域的三个固定卷积核 [3,1,k,k]：求和、x 偏移加权和、y 偏移加权和（相对中心，单位像素）。"""
    k = 2 * int(radius) + 1
    offsets = torch.arange(k, dtype=dtype) - int(radius)
    ones = torch.ones(k, k, dtype=dtype)
    kx = offsets.view(1, k).expand(k, k)
    ky = offsets.view(k, 1).expand(k, k)
    return torch.stack([ones, kx, ky]).unsqueeze(1).contiguous()


class EvidenceFrontEnd(nn.Module):
    """逐窗递推的证据前端（无可学习参数，见模块说明）。

    参数:
        taus_ms           时间矩的时间尺度（毫秒），例如 [20, 100, 500, 2000]
        window_ms         窗长 dt
        dipole_taus_ms    计算极性偶极所用的时间尺度（必须出现在 taus_ms 中）
        dipole_radius     偶极邻域半径 R
        bg_fast_ms/bg_slow_ms  背景快、慢两个指数平均的时间常数
        bg_prior          背景先验（每像素每窗期望事件数），bg_prior_windows 为先验的伪窗数 n0
        bg_floor          mu0 的下限
        bg_smooth_radius  快速背景估计的邻域平均半径 r
    状态（字典）：A, B [B,2K,H,W]；S_fast, S_slow [B,1,H,W]；W_fast, W_slow 浮点数；k 已处理窗数。
    """

    def __init__(self, taus_ms, window_ms, dipole_taus_ms, dipole_radius=3, bg_fast_ms=500.0,
                 bg_slow_ms=5000.0, bg_prior=0.01, bg_prior_windows=2.0, bg_floor=1e-3, bg_smooth_radius=7):
        super(EvidenceFrontEnd, self).__init__()
        self.taus = [float(t) for t in taus_ms]
        self.window_ms = float(window_ms)
        self.dipole_index = []
        for tau in dipole_taus_ms:
            if float(tau) not in self.taus:
                raise ValueError("偶极时间尺度 %s 不在 taus_ms 中" % tau)
            self.dipole_index.append(self.taus.index(float(tau)))
        self.dipole_radius = int(dipole_radius)
        self.beta_fast = math.exp(-self.window_ms / float(bg_fast_ms))
        self.beta_slow = math.exp(-self.window_ms / float(bg_slow_ms))
        self.bg_prior, self.bg_prior_windows = float(bg_prior), float(bg_prior_windows)
        self.bg_floor, self.bg_smooth_radius = float(bg_floor), int(bg_smooth_radius)
        if min(self.taus) <= 0 or self.bg_floor <= 0 or self.bg_prior <= 0 or self.bg_prior_windows <= 0:
            raise ValueError("时间尺度、背景先验、背景下限都必须为正")
        K = len(self.taus)
        # 常数缓冲区保存为 float64，使用时再转到输入精度（float64 核对时不会带入 float32 舍入）
        per_channel = torch.tensor([t for t in self.taus for _ in (0, 1)], dtype=torch.float64)
        self.register_buffer("decay", torch.exp(-self.window_ms / per_channel).view(1, 2 * K, 1, 1),
                             persistent=False)
        self.register_buffer("tau_channel", per_channel.view(1, 2 * K, 1, 1), persistent=False)
        self.register_buffer("kernels", offset_kernels(self.dipole_radius, torch.float64), persistent=False)

    @property
    def n_features(self):
        """输出特征通道数 = 2 + 4K + 2*len(dipole_taus)。"""
        return 2 + 4 * len(self.taus) + 2 * len(self.dipole_index)

    def feature_names(self):
        """每个特征通道的名称（写进配置记录，方便核对通道顺序）。"""
        names = ["count_pos", "count_neg"]
        for kind in ("ratio", "age"):
            for tau in self.taus:
                names += ["%s%g_pos" % (kind, tau), "%s%g_neg" % (kind, tau)]
        for j in self.dipole_index:
            names += ["dipole%g_x" % self.taus[j], "dipole%g_y" % self.taus[j]]
        return names

    def init_state(self, batch, height, width, device, dtype=torch.float32):
        """序列开头的状态：没有历史，背景强度等于先验。"""
        K2 = 2 * len(self.taus)
        zeros = lambda c: torch.zeros(batch, c, height, width, device=device, dtype=dtype)  # noqa: E731
        return {"A": zeros(K2), "B": zeros(K2), "S_fast": zeros(1), "S_slow": zeros(1),
                "W_fast": 0.0, "W_slow": 0.0, "k": 0}

    def background(self, state):
        """只用已处理窗口的计数估计本窗的背景强度 mu0 [B,1,H,W]（每像素每窗期望事件数）。"""
        n0, prior = self.bg_prior_windows, self.bg_prior
        fast = (state["S_fast"] + n0 * prior) / (state["W_fast"] + n0)
        slow = (state["S_slow"] + n0 * prior) / (state["W_slow"] + n0)
        r = self.bg_smooth_radius
        if r > 0:
            fast = F.avg_pool2d(fast, 2 * r + 1, stride=1, padding=r, count_include_pad=False)
        return torch.clamp(torch.maximum(fast, slow), min=self.bg_floor)

    def dipole(self, A_pos, A_neg):
        """局部极性偶极 [B,2,H,W]：(ON 质心 - OFF 质心)/R，按 n_on*n_off/(n_on*n_off+1) 加权（单极性时趋于 0）。"""
        batch = int(A_pos.shape[0])
        stacked = torch.cat([A_pos, A_neg], 0)                                   # [2B,1,H,W]
        sums = F.conv2d(stacked, self.kernels.to(stacked.dtype), padding=self.dipole_radius)
        pos, neg = sums[:batch], sums[batch:]
        eps = 1e-6
        centroid_pos = pos[:, 1:3] / (pos[:, 0:1] + eps)
        centroid_neg = neg[:, 1:3] / (neg[:, 0:1] + eps)
        both = pos[:, 0:1] * neg[:, 0:1]
        weight = both / (both + 1.0)
        return weight * (centroid_pos - centroid_neg) / float(self.dipole_radius)

    def step(self, state, counts, moment_a, moment_b):
        """处理一个窗口。

        输入: counts [B,2,H,W]；moment_a/moment_b [B,2K,H,W]（EventChunkSource 的一窗）。
        输出: (新状态, features [B,C,H,W], mu0 [B,1,H,W]（只用过去）, total [B,1,H,W] 本窗事件数)
        """
        mu0 = self.background(state)
        dt = self.window_ms
        decay = self.decay.to(counts.dtype)
        A_old, B_old = state["A"], state["B"]
        A = decay * A_old + moment_a
        B = decay * (B_old + dt * A_old) + moment_b
        total = counts.sum(1, keepdim=True)
        new_state = {"A": A, "B": B,
                     "S_fast": self.beta_fast * state["S_fast"] + total,
                     "S_slow": self.beta_slow * state["S_slow"] + total,
                     "W_fast": self.beta_fast * state["W_fast"] + 1.0,
                     "W_slow": self.beta_slow * state["W_slow"] + 1.0,
                     "k": int(state["k"]) + 1}
        tau = self.tau_channel.to(counts.dtype)
        count_feat = torch.log1p(counts / (0.5 * mu0))
        ratio = torch.log1p(A / (mu0 * tau / (2.0 * dt)))
        age = torch.clamp(B / (A + 1e-3) / tau, max=3.0)
        parts = [count_feat, ratio, age]
        for j in self.dipole_index:
            parts.append(self.dipole(A[:, 2 * j:2 * j + 1], A[:, 2 * j + 1:2 * j + 2]))
        return new_state, torch.cat(parts, 1), mu0, total

    def run_chunk(self, state, chunk):
        """把 EventChunkSource.chunk 的 T 个窗口依次送入 step。

        输出: (新状态, features [T,B,C,H,W], mu0 [T,B,1,H,W], total [T,B,1,H,W])
        """
        feats, mus, totals = [], [], []
        for t in range(int(chunk["counts"].shape[0])):
            state, f, mu0, total = self.step(state, chunk["counts"][t], chunk["moment_a"][t], chunk["moment_b"][t])
            feats.append(f)
            mus.append(mu0)
            totals.append(total)
        return state, torch.stack(feats), torch.stack(mus), torch.stack(totals)


def detach_frontend_state(state):
    """截断前端状态的计算图（前端无参数，通常在 no_grad 下运行；这里只为保险）。"""
    return {key: (value.detach() if torch.is_tensor(value) else value) for key, value in state.items()}
