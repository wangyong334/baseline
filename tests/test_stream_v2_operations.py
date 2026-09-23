"""前端与判决层运算量估计（件 2：能耗口径）的单元测试。

这些数字进论文，所以验证的不是"能跑"，而是**缩放规律**：
    前端随像素数、时间尺度数、偶极半径平方增长；关掉某组特征就该少掉对应的那一项
    判决层随假设数 V、足迹 f^2、告警阈值份数增长；延迟读出随事件数与 d 增长
另外验证"判决层完全稠密"这一条——事件数变化不影响它的逐像素部分（论文里要据此说明方案三的必要性）。
"""
import unittest

from dataset.stream_features import EvidenceFrontEnd
from model.evidence_neuron import DriftCUSUM, velocity_grid

H, W = 64, 80
P = H * W
KEYS = ("mac", "elementwise", "transcendental")


def frontend(taus=(20.0, 100.0), dipole=(100.0,), radius=3, features=None, bg_mode="adaptive", bg_radius=7):
    return EvidenceFrontEnd(list(taus), 50.0, list(dipole), radius, bg_smooth_radius=bg_radius,
                            features=features or ("count", "ratio", "age", "dipole"), bg_mode=bg_mode)


def cusum(velocities=(-1.0, 0.0, 1.0), footprint=3, aggregate="lme", track_decay=0.8, compensator="poisson"):
    return DriftCUSUM(velocity_grid(list(velocities)), footprint, compensator, None, aggregate, track_decay)


class FrontEndOperationsTests(unittest.TestCase):
    def test_scales_with_pixels(self):
        """像素数翻倍，逐像素部分翻倍（逐事件部分不变，所以总数介于 1 倍和 2 倍之间但逼近 2 倍）。"""
        fe = frontend()
        small = fe.estimate_operations(H, W, 0.0)          # 没有事件 -> 纯逐像素
        big = fe.estimate_operations(2 * H, W, 0.0)
        for key in KEYS:
            self.assertAlmostEqual(big[key], 2.0 * small[key], delta=1e-6 * max(big[key], 1.0), msg=key)

    def test_scales_with_events(self):
        """事件数翻倍，只有逐事件那一项翻倍。"""
        fe = frontend()
        a = fe.estimate_operations(H, W, 1000.0)
        b = fe.estimate_operations(H, W, 2000.0)
        first_a = [p for p in a["per_part"] if p["part"].startswith("事件累加")][0]
        first_b = [p for p in b["per_part"] if p["part"].startswith("事件累加")][0]
        for key in KEYS:
            self.assertAlmostEqual(first_b[key], 2.0 * first_a[key], delta=1e-6, msg=key)
            other_a = a[key] - first_a[key]
            other_b = b[key] - first_b[key]
            self.assertAlmostEqual(other_a, other_b, delta=1e-6, msg=key)

    def test_feature_groups_add_up(self):
        """逐组打开特征，总量应当逐项递增；关掉的组对应的 per_part 条目消失。"""
        chain = [("count",), ("count", "ratio"), ("count", "ratio", "age"),
                 ("count", "ratio", "age", "dipole")]
        totals = []
        for features in chain:
            report = frontend(features=features).estimate_operations(H, W, 500.0)
            names = " ".join(p["part"] for p in report["per_part"])
            self.assertEqual("极性偶极" in names, "dipole" in features)
            self.assertEqual("年龄特征" in names, "age" in features)
            totals.append(report["mac"] + report["elementwise"] + report["transcendental"])
        self.assertEqual(totals, sorted(totals))
        self.assertGreater(totals[-1], totals[0])

    def test_constant_background_is_cheaper(self):
        """bg_mode=constant 去掉两个 EMA 与盒式平滑，应当明显便宜。"""
        adaptive = frontend(bg_mode="adaptive").estimate_operations(H, W, 500.0)
        constant = frontend(bg_mode="constant").estimate_operations(H, W, 500.0)
        self.assertLess(constant["elementwise"], adaptive["elementwise"])
        self.assertLess(constant["mac"], adaptive["mac"])
        self.assertEqual(constant["reducible_ops"],
                         adaptive["reducible_ops"] - (15 * 15 - 2 * 15) * P)

    def test_dipole_cost_grows_with_radius_squared(self):
        """偶极卷积按 (2R+1)^2 增长，这是前端里最贵的一项。"""
        r1 = frontend(radius=3).estimate_operations(H, W, 0.0)
        r2 = frontend(radius=5).estimate_operations(H, W, 0.0)
        part1 = [p for p in r1["per_part"] if p["part"].startswith("极性偶极")][0]
        part2 = [p for p in r2["per_part"] if p["part"].startswith("极性偶极")][0]
        ratio = (part2["mac"] - 11 * P) / (part1["mac"] - 11 * P)
        self.assertAlmostEqual(ratio, (11.0 ** 2) / (7.0 ** 2), places=6)
        self.assertGreater(part1["mac"], 0.5 * r1["mac"])      # 确实是大头

    def test_more_scales_cost_more(self):
        """时间尺度数 K 线性放大时间矩与比值/年龄特征。"""
        two = frontend(taus=(20.0, 100.0)).estimate_operations(H, W, 0.0)
        four = frontend(taus=(20.0, 100.0, 500.0, 2000.0)).estimate_operations(H, W, 0.0)
        self.assertGreater(four["mac"], two["mac"])
        self.assertLess(four["mac"], 2.5 * two["mac"])

    def test_state_elements_positive_and_note_present(self):
        report = frontend().estimate_operations(H, W, 500.0)
        self.assertGreater(report["state_elements"], 0)
        self.assertIn("reducible", report["note"])


class DecisionOperationsTests(unittest.TestCase):
    def test_dense_regardless_of_events(self):
        """判决层的逐像素部分与事件数无关——这是"必须做块稀疏"的论据。"""
        layer = cusum()
        a = layer.estimate_operations(H, W, 0.0, readout_delays=(), alarm_thetas=0)
        b = layer.estimate_operations(H, W, 1e6, readout_delays=(), alarm_thetas=0)
        for key in KEYS:
            self.assertAlmostEqual(a[key], b[key], delta=1e-6, msg=key)

    def test_scales_with_hypotheses(self):
        """假设数 V 从 3 到 7，逐像素部分近似线性增长。"""
        three = cusum(velocities=(-1.0, 0.0, 1.0)).estimate_operations(H, W)
        seven = cusum(velocities=(-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)).estimate_operations(H, W)
        # velocity_grid 取单轴取值的外积，所以 3 个取值 -> 9 个假设、7 个取值 -> 49 个假设
        self.assertEqual(three["hypotheses"], 9)
        self.assertEqual(seven["hypotheses"], 49)
        for key in ("mac", "elementwise"):
            ratio = seven[key] / three[key]
            self.assertGreater(ratio, 4.5, key)
            self.assertLess(ratio, 5.5, key)

    def test_footprint_dominates_elementwise(self):
        """足迹 f 从 3 到 5，lme 聚合的加法按 f^2 增长。"""
        f3 = cusum(footprint=3).estimate_operations(H, W)
        f5 = cusum(footprint=5).estimate_operations(H, W)
        p3 = [p for p in f3["per_part"] if p["part"].startswith("足迹聚合")][0]
        p5 = [p for p in f5["per_part"] if p["part"].startswith("足迹聚合")][0]
        v = float(f3["hypotheses"])
        self.assertAlmostEqual((p5["elementwise"] - 2 * v * P) / (p3["elementwise"] - 2 * v * P),
                               25.0 / 9.0, places=6)

    def test_alarm_thetas_add_membrane_copies(self):
        """每个告警阈值各一份膜电位；3 个阈值时膜电位那一项应当是 4 份。"""
        none = cusum().estimate_operations(H, W, alarm_thetas=0)
        three = cusum().estimate_operations(H, W, alarm_thetas=3)
        p0 = [p for p in none["per_part"] if p["part"].startswith("膜电位累加")][0]
        p3 = [p for p in three["per_part"] if p["part"].startswith("膜电位累加")][0]
        self.assertAlmostEqual(p3["transcendental"] / p0["transcendental"], 4.0, places=6)
        self.assertGreater(p3["elementwise"], 4.0 * p0["elementwise"] * 0.99)

    def test_readout_delays_scale_with_events_and_delay(self):
        """延迟读出按事件数与 d 线性增长。"""
        one = cusum().estimate_operations(H, W, 1000.0, readout_delays=(1,))
        five = cusum().estimate_operations(H, W, 1000.0, readout_delays=(5,))
        p1 = [p for p in one["per_part"] if p["part"].startswith("延迟读出")][0]
        p5 = [p for p in five["per_part"] if p["part"].startswith("延迟读出")][0]
        v = float(one["hypotheses"])
        self.assertAlmostEqual((p5["elementwise"] - 1000.0 * v) / (p1["elementwise"] - 1000.0 * v),
                               5.0, places=6)
        self.assertAlmostEqual(p5["transcendental"], p1["transcendental"], places=6)

    def test_memory_switch_and_aggregate_switch(self):
        """关掉管道记忆省掉一次乘加与两次平移；sum 聚合不需要 exp/log。"""
        with_mem = cusum(track_decay=0.8).estimate_operations(H, W)
        no_mem = cusum(track_decay=0.0).estimate_operations(H, W)
        self.assertLess(no_mem["mac"], with_mem["mac"])
        self.assertLess(no_mem["elementwise"], with_mem["elementwise"])
        lme = cusum(aggregate="lme").estimate_operations(H, W)
        s = cusum(aggregate="sum").estimate_operations(H, W)
        self.assertLess(s["transcendental"], lme["transcendental"])

    def test_negbin_compensator_costs_more(self):
        """负二项补偿多一次 log1p。"""
        poisson = cusum(compensator="poisson").estimate_operations(H, W)
        negbin = DriftCUSUM(velocity_grid([-1.0, 0.0, 1.0]), 3, "negbin", 5.0, "lme", 0.8)
        self.assertGreater(negbin.estimate_operations(H, W)["transcendental"], poisson["transcendental"])


if __name__ == "__main__":
    unittest.main()
