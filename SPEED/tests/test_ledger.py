import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speed.core.build import load_config  # noqa: E402

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
CATEGORIES = {"physical", "requirement", "design", "init", "protocol", "numerical", "evuav"}


def leaves(node, path=""):
    """Numeric (or null / numeric-list) leaves; readouts addressed by kind."""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            out += leaves(v, "%s.%s" % (path, k) if path else k)
    elif isinstance(node, list) and node and all(isinstance(v, dict) for v in node):
        for v in node:
            out += leaves({k: x for k, x in v.items() if k != "kind"}, "%s.%s" % (path, v["kind"]))
    elif node is None or isinstance(node, (int, float)) and not isinstance(node, bool) or (
            isinstance(node, list) and all(isinstance(v, (int, float)) for v in node)):
        out.append(path)
    return out


class LedgerTests(unittest.TestCase):
    def test_every_constant_has_a_provenance(self):
        ledger = load_config(os.path.join(CONFIGS, "ledger.yaml"))
        for key, (category, note) in ledger.items():
            self.assertIn(category, CATEGORIES, key)
            self.assertTrue(note, key)
        for name in sorted(n for n in os.listdir(CONFIGS) if n.startswith("base_") and n.endswith(".yaml")):
            missing = [p for p in leaves(load_config(os.path.join(CONFIGS, name))) if p not in ledger]
            self.assertEqual(missing, [], name)
        v4 = os.path.join(CONFIGS, "v4")
        for name in sorted(n for n in os.listdir(v4) if n.endswith(".yaml")):
            missing = [p for p in leaves(load_config(os.path.join(v4, name))) if p not in ledger]
            self.assertEqual(missing, [], name)


if __name__ == "__main__":
    unittest.main()
