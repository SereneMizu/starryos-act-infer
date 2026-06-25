#!/usr/bin/env python3
"""从全量推理结果中提取 2 帧测试参考帧."""
import json
import sys
from pathlib import Path

TEST_FRAMES = {"frame_000000.jpg", "frame_000227.jpg"}

def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/infer_results_torch.json")
    dst_dir = Path("output/test")
    dst_dir.mkdir(parents=True, exist_ok=True)

    with open(src) as f:
        all_results = json.load(f)

    ref = [r for r in all_results if r["frame"] in TEST_FRAMES]
    found = {r["frame"] for r in ref}
    missing = TEST_FRAMES - found
    if missing:
        print(f"Warning: frames not found in {src}: {missing}", file=sys.stderr)

    dst = dst_dir / "reference.json"
    with open(dst, "w") as f:
        json.dump(ref, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(ref)} frames to {dst}")


if __name__ == "__main__":
    main()
