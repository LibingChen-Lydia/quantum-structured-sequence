"""Add the shared-encoder record to run JSONs written before it was logged.

The encoder fit is deterministic given (config, seed, n_train), so the record is recomputed
exactly as ``run_module_a.load_data`` produces it. Records that already have it are left alone.

    python experiments/synthetic_periodic/backfill_encoder_record.py s1_encoding s2_topology_4q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_module_a import OUT_ROOT, load_data


def main(exps: list[str]) -> None:
    for exp in exps:
        d = OUT_ROOT / exp
        cfg = json.loads((d / "config.json").read_text())
        done = 0
        for p in sorted(d.glob("*_s*.json")):
            rec = json.loads(p.read_text())
            if "encoder" in rec:
                continue
            *_, enc_rec = load_data(cfg, rec["seed"], rec["n_train"])
            rec["encoder"], rec["encoder_backfilled"] = enc_rec, True
            rec.setdefault("data", cfg.get("data", {}))
            rec.setdefault("task_type", "regression")
            p.write_text(json.dumps(rec, indent=2))
            done += 1
        print(f"{exp}: backfilled {done} records")


if __name__ == "__main__":
    main(sys.argv[1:])
