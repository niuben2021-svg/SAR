"""Inspect class/size distribution in official trainval and test partitions."""
import argparse
import json
from collections import defaultdict

import numpy as np

from data import find_partition_samples, parse_txt, sample_paths


def summarize(root, samples):
    records = defaultdict(list)
    for rec in samples:
        _, txt = sample_paths(root, rec)
        for obj in parse_txt(txt):
            x1, y1, x2, y2 = obj["box"]
            records[obj["label"]].append([x2 - x1, y2 - y1])
    results = {"num_images": len(samples), "by_class": {}}
    for name, values in sorted(records.items()):
        arr = np.asarray(values)
        areas = arr[:, 0] * arr[:, 1]
        results["by_class"][name] = {
            "n": len(values),
            "median_width_px": round(float(np.median(arr[:, 0])), 2),
            "median_height_px": round(float(np.median(arr[:, 1])), 2),
            "median_area_px2": round(float(np.median(areas)), 2),
            "tiny_area_lt_256": int((areas < 256).sum()),
            "small_area_256_to_1023": int(((areas >= 256) & (areas < 1024)).sum()),
            "larger_area_ge_1024": int((areas >= 1024).sum()),
        }
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    args = p.parse_args()
    results = {part: summarize(args.data_root, find_partition_samples(args.data_root, part))
               for part in ("trainval", "test")}
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
