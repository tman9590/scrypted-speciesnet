import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class ConfigTests(unittest.TestCase):
    def test_backend_projections_match_root(self):
        root = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(root["model"], "resnet")
        self.assertEqual(root["input_shape"], [1, 3, 480, 480])
        self.assertEqual(len(root["labels"]), 2498)
        self.assertEqual(len(set(root["labels"].values())), 2498)
        for backend, backend_info in root["backends"].items():
            projection = json.loads((ROOT / "models" / backend / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(projection["files"], backend_info["files"])
            for key in ("model", "input_shape", "labels"):
                self.assertEqual(projection[key], root[key])


if __name__ == "__main__":
    unittest.main()

