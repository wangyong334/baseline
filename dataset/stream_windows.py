"""流式 SNN V1 的数据核心（只依赖 numpy，可在无 GPU 环境单独测试）。

职责：读取 NPZ -> 校验事件 -> 按 50 ms 切窗 -> 构造 12 通道输入 -> 训练集统计。
本文件不导入 torch / HAIS_OP / configs.configs，保证单元测试可以在本地运行。
"""
import numpy as np


# ---------------------------------------------------------------------------
# 通道布局（固定约定，训练集统计、网络输入、单元测试都依赖它）
#   通道 0      : 正极性(p=1) 整窗事件计数
#   通道 1      : 负极性(p=0) 整窗事件计数
#   通道 2+2b   : 第 b 个时间 bin 的正极性计数
#   通道 3+2b   : 第 b 个时间 bin 的负极性计数
# ---------------------------------------------------------------------------


def num_input_channels(time_bins):
    """返回输入通道数。

    输入: time_bins 窗内时间分箱数（V1 为 5）。
    输出: 2 + 2*time_bins（V1 为 12）。
    """
    return 2 + 2 * int(time_bins)


def channel_names(time_bins):
    """返回每个输入通道的可读名称，写进统计 JSON，方便人工核对通道顺序。"""
    names = ["pos_count", "neg_count"]
    for b in range(int(time_bins)):
        names += ["bin%d_pos" % b, "bin%d_neg" % b]
    return names


class StreamSequence(object):
    """一个 8 秒序列的全部事件（numpy 数组），以及切窗所需的索引。

    属性（全部长度为 N，保持 NPZ 文件中的原始事件顺序）：
        name       文件名
        x, y, t    int64 像素坐标与毫秒时间戳
        p          int8  极性 0/1
        label      float32 事件标签 0/1
        target_id  float64 目标编号（与原评估代码 roc_update 的 idx 类型一致）
        inner_bin  int64 窗内时间 bin 编号 0..time_bins-1
        t_local    float32 窗内相对时间 (t % window_ms) / window_ms，范围 [0,1)
        order      int64 按时间稳定排序后的原始下标
        bounds     int64 长度 n_windows+1，第 k 窗的事件为 order[bounds[k]:bounds[k+1]]
    """

    def __init__(self, name, x, y, t, p, label, target_id,
                 inner_bin, t_local, order, bounds):
        self.name = name
        self.x, self.y, self.t = x, y, t
        self.p, self.label, self.target_id = p, label, target_id
        self.inner_bin, self.t_local = inner_bin, t_local
        self.order, self.bounds = order, bounds

    @property
    def n_events(self):
        """序列事件总数。"""
        return int(self.t.shape[0])

    @property
    def n_windows(self):
        """序列窗口数（V1 为 160）。"""
        return int(self.bounds.shape[0] - 1)

    def window_index(self, k):
        """返回第 k 个窗口内事件的原始下标（已按时间稳定排序）。"""
        return self.order[self.bounds[k]:self.bounds[k + 1]]


def validate_events(x, y, t, p, label, height, width, total_ms):
    """校验事件取值范围；任何一项不合法都抛出 ValueError，并给出违规数量。

    检查项：0<=t<total_ms，0<=x<width，0<=y<height，p∈{0,1}，label∈{0,1}，
    以及所有数组长度一致。宁可在数据阶段报错，也不要让越界坐标静默写错像素。
    """
    n = t.shape[0]
    for name, arr in (("x", x), ("y", y), ("p", p), ("label", label)):
        if arr.shape[0] != n:
            raise ValueError("长度不一致: %s=%d, t=%d" % (name, arr.shape[0], n))
    problems = []
    checks = (
        ("t<0", t < 0), ("t>=%d" % total_ms, t >= total_ms),
        ("x<0", x < 0), ("x>=%d" % width, x >= width),
        ("y<0", y < 0), ("y>=%d" % height, y >= height),
        ("p不在{0,1}", ~np.isin(p, (0, 1))),
        ("label不在{0,1}", ~np.isin(label, (0, 1))),
    )
    for desc, mask in checks:
        count = int(np.count_nonzero(mask))
        if count:
            problems.append("%s: %d 个事件" % (desc, count))
    if problems:
        raise ValueError("事件校验失败 -> " + "; ".join(problems))


def split_windows(t, window_ms, n_windows):
    """按时间戳把事件划分到连续窗口。

    输入: t 毫秒时间戳（int64，已通过校验），window_ms 窗长，n_windows 窗数。
    输出: (order, bounds)
        order  按 t 稳定排序的原始下标（同一时间戳保持文件原顺序）
        bounds 长度 n_windows+1，第 k 窗事件为 order[bounds[k]:bounds[k+1]]
    说明: 空窗口的 bounds[k]==bounds[k+1]，窗口依然存在（网络仍需前向以衰减状态）。
    """
    order = np.argsort(t, kind="stable").astype(np.int64)
    window_of_sorted = t[order] // int(window_ms)
    bounds = np.searchsorted(window_of_sorted, np.arange(n_windows + 1), side="left")
    return order, bounds.astype(np.int64)


def time_bin_and_local(t, window_ms, time_bins):
    """计算每个事件的窗内时间 bin 与窗内相对时间（整数运算，无浮点边界误差）。

    输入: t 毫秒时间戳 int64。
    输出: (inner_bin, t_local)
        inner_bin = ((t % window_ms) * time_bins) // window_ms，取值 0..time_bins-1
        t_local   = (t % window_ms) / window_ms，float32，范围 [0,1)
    例: window_ms=50, time_bins=5 时 t=49 -> bin 4, t_local 0.98；t=50 -> bin 0, t_local 0。
    """
    local = t % int(window_ms)
    inner_bin = (local * int(time_bins)) // int(window_ms)
    t_local = (local.astype(np.float64) / float(window_ms)).astype(np.float32)
    return inner_bin.astype(np.int64), t_local


def load_npz_events(path, height, width, window_ms, n_windows, time_bins):
    """读取一个 EV-UAV NPZ 文件并构造 StreamSequence。

    数据来源（与原仓库 dataset/ev_uav.py 一致）：
        ev_loc[:,0:3]      -> x, y, t（整数坐标与毫秒时间）
        evs_norm[:,3]      -> p
        evs_norm[:,4]      -> label
        evs_norm[:,5]      -> target_id
    不做任何降采样；所有事件都会被处理且只处理一次。
    """
    with np.load(path) as data:
        ev_loc = np.asarray(data["ev_loc"])
        evs_norm = np.asarray(data["evs_norm"])
    if ev_loc.ndim != 2 or ev_loc.shape[1] < 3:
        raise ValueError("ev_loc 形状异常: %s" % (ev_loc.shape,))
    if evs_norm.ndim != 2 or evs_norm.shape[1] < 6:
        raise ValueError("evs_norm 形状异常: %s" % (evs_norm.shape,))
    if ev_loc.shape[0] != evs_norm.shape[0]:
        raise ValueError("ev_loc 与 evs_norm 行数不一致")
    xyt = ev_loc[:, :3]
    if not np.all(np.equal(xyt, np.round(xyt))):
        raise ValueError("ev_loc 含非整数坐标或时间")
    xyt = xyt.astype(np.int64)
    x, y, t = xyt[:, 0], xyt[:, 1], xyt[:, 2]
    p_raw, label_raw = evs_norm[:, 3], evs_norm[:, 4]
    total_ms = int(window_ms) * int(n_windows)
    validate_events(x, y, t, p_raw, label_raw, height, width, total_ms)
    order, bounds = split_windows(t, window_ms, n_windows)
    inner_bin, t_local = time_bin_and_local(t, window_ms, time_bins)
    return StreamSequence(
        name=str(path).replace("\\", "/").split("/")[-1],
        x=x, y=y, t=t,
        p=p_raw.astype(np.int8), label=label_raw.astype(np.float32),
        target_id=evs_norm[:, 5].astype(np.float64),
        inner_bin=inner_bin, t_local=t_local, order=order, bounds=bounds)


def count_channels(x, y, p, inner_bin, time_bins, pad_height, pad_width):
    """把一个窗口的事件累计成 [通道, pad_height, pad_width] 的整数计数图。

    像素 (x,y) 直接写到画布的 (y,x) 位置：只在右侧和下侧补零，
    因此原始坐标不需要任何偏移，读出头也直接用原始 (x,y) 取特征。
    输出 int64 数组；空窗口返回全零。
    """
    n_ch = num_input_channels(time_bins)
    plane = int(pad_height) * int(pad_width)
    if x.shape[0] == 0:
        return np.zeros((n_ch, pad_height, pad_width), dtype=np.int64)
    pixel = y.astype(np.int64) * int(pad_width) + x.astype(np.int64)
    is_negative = (p == 0).astype(np.int64)            # 正极性 -> 0，负极性 -> 1
    whole_channel = is_negative                         # 通道 0 / 1
    bin_channel = 2 + 2 * inner_bin.astype(np.int64) + is_negative
    flat = np.concatenate([whole_channel * plane + pixel, bin_channel * plane + pixel])
    counts = np.bincount(flat, minlength=n_ch * plane)
    return counts.reshape(n_ch, pad_height, pad_width)


def normalize_counts(counts, q99, clip_max):
    """固定尺度归一化: X_c = clip(log1p(C_c) / q99_c, 0, clip_max)。

    q99 只能来自训练集统计（见 update_count_histogram / quantile_from_histogram），
    不能逐窗口重新归一化，否则会抹掉窗口之间真实的事件密度差异。
    计数为 0 的像素输出严格为 0（零输入 -> 零电流）。
    """
    q = np.asarray(q99, dtype=np.float32).reshape(-1, 1, 1)
    if q.shape[0] != counts.shape[0]:
        raise ValueError("q99 长度 %d 与通道数 %d 不一致" % (q.shape[0], counts.shape[0]))
    if np.any(q <= 0):
        raise ValueError("q99 必须全部为正")
    out = np.log1p(counts.astype(np.float32)) / q
    return np.clip(out, 0.0, float(clip_max)).astype(np.float32)


def normalization_table(q99, clip_max, max_entries=1 << 22):
    """把 normalize_counts 预先算成查找表，供 GPU 构造输入时查表（与 CPU 输入逐位相同）。

    输出: (table, n_sat)
        table  float32 [C, n_sat+1]，table[c, n] 就是 normalize_counts 对通道 c、计数 n 的输出
        n_sat  饱和计数：计数 >= n_sat 时输出恒为 clip_max，查表时把计数截断到 n_sat 即可
    表由 normalize_counts 本身算出，不重写公式；log1p 单调，超过 n_sat 的计数必然被截断到 clip_max。
    """
    q = np.asarray(q99, dtype=np.float32)
    # log1p(n)/q >= clip_max  <=>  n >= expm1(clip_max*q)；多留 2 个余量吸收 float32 舍入
    n_sat = int(np.ceil(np.expm1(float(clip_max) * float(q.max())))) + 2
    if n_sat + 1 > int(max_entries):
        raise ValueError("饱和计数 %d 过大（q99 最大 %.3f），不适合查表" % (n_sat, float(q.max())))
    counts = np.tile(np.arange(n_sat + 1, dtype=np.int64), (q.shape[0], 1, 1))   # [C, 1, n_sat+1]
    table = normalize_counts(counts, q, clip_max)[:, 0, :]
    if not np.all(table[:, -1] == np.float32(clip_max)):
        raise AssertionError("查找表末项没有饱和到 clip_max")
    return np.ascontiguousarray(table), n_sat


def build_window_input(seq, k, q99, time_bins, pad_height, pad_width, clip_max):
    """构造第 k 个窗口的网络输入与读出所需的逐事件数据。

    输出: (inp, idx)
        inp  float32 [C, pad_height, pad_width] 归一化后的 12 通道输入
        idx  该窗口事件的原始下标（按时间稳定排序），用于逐事件读出和回填
    """
    idx = seq.window_index(k)
    counts = count_channels(seq.x[idx], seq.y[idx], seq.p[idx], seq.inner_bin[idx],
                            time_bins, pad_height, pad_width)
    return normalize_counts(counts, q99, clip_max), idx


def check_window_partition(seq, window_ms):
    """断言所有窗口恰好覆盖全部事件一次，并且每个事件落在正确的窗口里。

    检查三件事：
      1. np.sort(拼接后的下标) == arange(N)：同时发现重复和遗漏
      2. 第 k 窗内所有事件满足 t // window_ms == k
      3. 窗内事件按时间非降序排列
    """
    parts = [seq.window_index(k) for k in range(seq.n_windows)]
    joined = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    if not np.array_equal(np.sort(joined), np.arange(seq.n_events)):
        raise AssertionError("窗口划分存在重复或遗漏事件")
    for k, part in enumerate(parts):
        if part.size == 0:
            continue
        if np.any(seq.t[part] // int(window_ms) != k):
            raise AssertionError("第 %d 窗含有不属于该窗的事件" % k)
        if np.any(np.diff(seq.t[part]) < 0):
            raise AssertionError("第 %d 窗内事件未按时间排序" % k)


def refill_by_index(n_events, index_parts, value_parts, dtype=np.float32):
    """把逐窗口的逐事件结果按原始下标回填为文件顺序。

    输入: n_events 事件总数；index_parts / value_parts 每个窗口的下标与数值列表。
    输出: 长度 n_events 的数组。若有事件未被写入或被写入多次则抛出 AssertionError。
    """
    out = np.zeros(n_events, dtype=dtype)
    hits = np.zeros(n_events, dtype=np.int64)
    for idx, val in zip(index_parts, value_parts):
        val = np.asarray(val)
        if idx.shape[0] != val.shape[0]:
            raise AssertionError("下标与数值长度不一致")
        out[idx] = val
        np.add.at(hits, idx, 1)
    if not np.all(hits == 1):
        raise AssertionError("回填失败: %d 个事件未写入, %d 个事件重复写入"
                             % (int(np.sum(hits == 0)), int(np.sum(hits > 1))))
    return out


# ---------------------------------------------------------------------------
# 训练集统计（q99 与 pos_weight）
# ---------------------------------------------------------------------------


def new_count_histogram(n_channels):
    """创建每个通道一个空的"计数值直方图"，hist[c][v] 表示通道 c 中计数为 v 的非零像素数。"""
    return [np.zeros(1, dtype=np.int64) for _ in range(int(n_channels))]


def update_count_histogram(hist, counts):
    """把一个窗口的计数图累加进直方图（只统计非零像素）。

    计数是整数，所以直方图可以精确求分位数，不需要保存全部像素值。
    """
    n_ch = counts.shape[0]
    flat = counts.reshape(n_ch, -1)
    ch_idx, pix_idx = np.nonzero(flat > 0)
    if ch_idx.size == 0:
        return hist
    values = flat[ch_idx, pix_idx].astype(np.int64)
    key = values * n_ch + ch_idx
    # minlength 取 (最大计数+1)*通道数，保证可以整形成 [计数值, 通道]
    table = np.bincount(key, minlength=(int(values.max()) + 1) * n_ch).reshape(-1, n_ch)
    for c in range(n_ch):
        col = table[:, c]
        if hist[c].shape[0] < col.shape[0]:
            grown = np.zeros(col.shape[0], dtype=np.int64)
            grown[:hist[c].shape[0]] = hist[c]
            hist[c] = grown
        hist[c][:col.shape[0]] += col
    return hist


def quantile_from_histogram(hist_c, q):
    """由单通道计数直方图求非零像素上 log1p(计数) 的 q 分位数（逆 CDF 定义）。

    定义: 取最小的计数值 v，使得 "计数<=v 的非零像素占比 >= q"，返回 log1p(v)。
    由于 log1p 单调，对计数求分位数再取 log1p 与直接对 log1p 值求分位数等价。
    若该通道在训练集中完全没有非零像素则抛出 ValueError。
    """
    hist_c = np.asarray(hist_c, dtype=np.int64).copy()
    hist_c[0] = 0                                   # 只统计非零像素
    total = int(hist_c.sum())
    if total == 0:
        raise ValueError("该通道没有非零像素，无法计算分位数")
    cdf = np.cumsum(hist_c)
    need = int(np.ceil(float(q) * total))
    need = min(max(need, 1), total)
    value = int(np.searchsorted(cdf, need, side="left"))
    return float(np.log1p(value))


def pos_weight_from_counts(n_pos, n_neg, cap):
    """正样本权重规则: min(训练集负样本数 / 正样本数, cap)。训练前只计算一次，不做搜索。"""
    if n_pos <= 0:
        raise ValueError("训练集没有正样本")
    return float(min(float(n_neg) / float(n_pos), float(cap)))
