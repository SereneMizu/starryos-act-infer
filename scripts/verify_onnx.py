import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TORCH_RESULTS = PROJECT_ROOT / "output" / "infer_results_torch.json"
ONNX_RESULTS = PROJECT_ROOT / "output" / "infer_results_onnx.json"


def main():
    with open(TORCH_RESULTS) as f:
        torch_data = json.load(f)
    with open(ONNX_RESULTS) as f:
        onnx_data = json.load(f)

    assert len(torch_data) == len(onnx_data), f"length mismatch: {len(torch_data)} vs {len(onnx_data)}"

    total = len(torch_data)
    same_turn = 0
    diff_entries = []
    max_left_diff = 0.0
    max_right_diff = 0.0

    for t, o in zip(torch_data, onnx_data):
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
                "torch": {"left_vel": t["left_vel"], "right_vel": t["right_vel"], "turn": t["turn"]},
                "onnx": {"left_vel": o["left_vel"], "right_vel": o["right_vel"], "turn": o["turn"]},
            })

    print(f"Total: {total} frames")
    print(f"Turn match: {same_turn}/{total} ({same_turn/total*100:.1f}%)")
    print(f"Turn differ: {total - same_turn}/{total}")
    print(f"Max left_vel diff:  {max_left_diff:.6f}")
    print(f"Max right_vel diff: {max_right_diff:.6f}")

    if diff_entries:
        print(f"\n--- Differing frames ({len(diff_entries)}) ---")
        for e in diff_entries:
            print(f"  {e['frame']}: torch={e['torch']['turn']} (L={e['torch']['left_vel']:+.6f} R={e['torch']['right_vel']:+.6f})"
                  f"  onnx={e['onnx']['turn']} (L={e['onnx']['left_vel']:+.6f} R={e['onnx']['right_vel']:+.6f})")
    else:
        print("\nAll turn directions match.")


if __name__ == "__main__":
    main()
