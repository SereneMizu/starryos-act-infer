import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TORCH_RESULTS = PROJECT_ROOT / "output" / "infer_results_torch.json"
ONNX_RESULTS = PROJECT_ROOT / "output" / "infer_results_onnx.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, default=TORCH_RESULTS, help="Reference results JSON")
    parser.add_argument("--b", type=Path, default=ONNX_RESULTS, help="Test results JSON")
    args = parser.parse_args()

    with open(args.a) as f:
        a_data = json.load(f)
    with open(args.b) as f:
        b_data = json.load(f)

    assert len(a_data) == len(b_data), f"length mismatch: {len(a_data)} vs {len(b_data)}"

    total = len(a_data)
    same_turn = 0
    diff_entries = []
    max_left_diff = 0.0
    max_right_diff = 0.0

    for t, o in zip(a_data, b_data):
        assert t["frame"] == o["frame"], f"frame mismatch: {t['frame']} vs {o['frame']}"
        left_diff = abs(t["left_vel"] - o["left_vel"])
        right_diff = abs(t["right_vel"] - o["right_vel"])
        max_left_diff = max(max_left_diff, left_diff)
        max_right_diff = max(max_right_diff, right_diff)
        turn_match = t["turn"] == o["turn"]
        if turn_match:
            same_turn += 1
        else:
            diff_entries.append({
                "frame": t["frame"],
                "a": {"left_vel": t["left_vel"], "right_vel": t["right_vel"], "turn": t["turn"]},
                "b": {"left_vel": o["left_vel"], "right_vel": o["right_vel"], "turn": o["turn"]},
            })

    a_name = args.a.stem.replace("infer_results_", "")
    b_name = args.b.stem.replace("infer_results_", "")
    print(f"Comparing: {a_name} vs {b_name}")
    print(f"Total: {total} frames")
    print(f"Turn match: {same_turn}/{total} ({same_turn/total*100:.1f}%)")
    print(f"Turn differ: {total - same_turn}/{total}")
    print(f"Max left_vel diff:  {max_left_diff:.6f}")
    print(f"Max right_vel diff: {max_right_diff:.6f}")

    if diff_entries:
        print(f"\n--- Differing frames ({len(diff_entries)}) ---")
        for e in diff_entries:
            print(f"  {e['frame']}: {a_name}={e['a']['turn']:<5} (L={e['a']['left_vel']:+.6f} R={e['a']['right_vel']:+.6f})"
                  f"  {b_name}={e['b']['turn']:<5} (L={e['b']['left_vel']:+.6f} R={e['b']['right_vel']:+.6f})")
    else:
        print("\nAll turn directions match.")


if __name__ == "__main__":
    main()
