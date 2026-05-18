from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

from check_h5_integrity import run_validation


CONFIGS = [
    ("one_node_branching", 5.7, "one_branch"),
    ("one_node_branching", 5.7, "multiple_branches"),
    ("one_node_branching", 2.5, "one_branch"),
    ("one_node_branching", 2.5, "multiple_branches"),
]


def main():
    worker = Path(__file__).resolve().parent / "final_results_worker.py"
    for topology, diameter, scenario in CONFIGS:
        env = dict(os.environ)
        env["TEST_MODE"] = "1"
        subprocess.run(
            [sys.executable, str(worker), topology, str(diameter), scenario],
            check=True,
            env=env,
        )

    ok, failures = run_validation(Path(__file__).resolve().parents[1] / "final_result")
    if ok:
        print("ACCEPTANCE TEST PASSED")
    else:
        print("ACCEPTANCE TEST FAILED")
        for item in failures:
            print(item)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
