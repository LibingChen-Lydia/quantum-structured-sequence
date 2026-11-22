"""Config-driven Module A runner: (model x train_size x seed) grid, one JSON per run.

    python experiments/synthetic_periodic/run_module_a.py --config configs/module_a_s1.yaml
    python experiments/synthetic_periodic/run_module_a.py --config configs/module_a_synthetic.yaml --quick
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import yaml

from qss.classical.baselines import ExplicitInteraction, FourierMLP, MatchedMLP, RawMLP
from qss.classical.patch_features import encode_items, fit_amp_max
from qss.classical.synthetic import make_task
from qss.evaluation.metrics import mae, mse
from qss.evaluation.run_log import write_record
from qss.fusion.gated import FusedForecaster, QuantumRegressor, phase_head_features
from qss.quantum.circuits import CircuitSpec
from qss.quantum.interaction import PennyLaneInteractionBackend, QuantumInteractionLayer, resources
from qss.training import predict, train

ROOT = Path(__file__).resolve().parents[2]


def model_tag(m: dict) -> str:
    keys = ("encoding", "topology", "depth", "readout", "l1")
    parts = [m["name"]] + [f"l1={m[k]}" if k == "l1" else f"{m[k]}" for k in keys if k in m]
    return "-".join(parts)


def load_data(cfg: dict, seed: int, n_train: int):
    d = dict(cfg.get("data", {}))
    n_val, n_test = d.pop("n_val", 512), d.pop("n_test", 2000)
    n_max = max(cfg["train_sizes"])
    data = make_task(cfg["task"], n_max + n_val + n_test, seed, **d)
    tr = slice(0, n_train)
    va = slice(n_max, n_max + n_val)
    te = slice(n_max + n_val, None)
    # shared upstream encoder, fitted on the training split only
    amp, phase, _ = encode_items(data.series[tr], data.series, data.period, data.patch_len)
    amp_max = fit_amp_max(amp[tr])
    y = data.y
    return (data, (amp[tr], phase[tr], y[tr]), (amp[va], phase[va], y[va]),
            (amp[te], phase[te], y[te]), amp_max)


def spec_from(m: dict, n_items: int) -> CircuitSpec:
    return CircuitSpec(n_items, m.get("encoding", "hybrid"), m.get("topology", "cross_period"),
                       m.get("depth", 2), m.get("readout", "z"))


def shots_eval(model: QuantumRegressor, test, shots_list, n_eval: int) -> dict:
    """Re-evaluate a trained quantum model with finite shots through the backend interface."""
    amp, phase, y = (a[:n_eval] for a in test)
    theta = np.arcsin(np.clip(amp / model.amp_max, 0, 1))
    out = {}
    for shots in shots_list:
        ent = None if model.layer.ent is None else model.layer.ent.detach().numpy()
        backend = PennyLaneInteractionBackend(model.layer.spec, model.layer.weights.detach().numpy(),
                                              shots=shots, ent=ent)
        t0 = time.perf_counter()
        q = backend.run_interaction(np.stack([theta, phase], -1))
        secs = time.perf_counter() - t0
        if model.layer.spec.encoding == "phase":
            q = phase_head_features(torch.as_tensor(q), torch.as_tensor(amp)).numpy()
        with torch.no_grad():
            p = model.head(torch.as_tensor(q)).squeeze(-1).numpy()
        out[str(shots)] = {"mse": mse(y, p), "seconds_per_sample": secs / len(y)}
    return out


def run_job(job: dict) -> str:
    torch.set_num_threads(1)
    cfg, m, n_train, seed, out_path = job["cfg"], job["model"], job["n_train"], job["seed"], Path(job["out"])
    if out_path.exists():
        return f"skip {out_path.name}"
    data, tr, va, te, amp_max = load_data(cfg, seed, n_train)
    n_items = tr[0].shape[1]
    tc = cfg.get("train", {})
    rec = {"experiment": cfg["name"], "task": cfg["task"], "model": m, "tag": model_tag(m),
           "n_train": n_train, "seed": seed, "noise_floor_mse": data.noise_y ** 2,
           "backend": "default.qubit", "shots_train": None,
           "features": "qss.classical.encoder.StructuredSequenceEncoder (fit on train split)",
           "amp_max": amp_max}
    t0 = time.perf_counter()
    name = m["name"]
    if name == "explicit":
        est = ExplicitInteraction().fit(*tr)
        pred_fn = est.predict
        rec.update(n_params=est.n_params, train_info={})
    elif name == "quantum_fixed":
        # untrained random circuit, only a ridge readout is fit (reservoir / fixed feature map)
        from sklearn.linear_model import RidgeCV
        spec = spec_from(m, n_items)
        layer = QuantumInteractionLayer(spec, seed=seed)

        def feats(a, p):
            with torch.no_grad():
                th = torch.arcsin(torch.clamp(torch.as_tensor(a) / amp_max, 0, 1))
                q = layer(th, torch.as_tensor(p)).numpy()
            if spec.encoding == "phase":
                return phase_head_features(torch.as_tensor(q), torch.as_tensor(a)).numpy()
            return q
        ridge = RidgeCV(alphas=np.logspace(-6, 2, 17)).fit(feats(tr[0], tr[1]), tr[2])
        pred_fn = lambda a, p: ridge.predict(feats(a, p))
        rec.update(n_params=ridge.coef_.size + 1, train_info={}, resources=resources(spec))
    else:
        spec = spec_from(m, n_items) if name in ("quantum", "fused", "matched_mlp") else None
        if name == "raw_mlp":
            model, lr = RawMLP(n_items, m.get("hidden", 32)), tc.get("lr_classical", 0.01)
        elif name == "fourier_mlp":
            model, lr = FourierMLP(n_items, m.get("hidden", 32)), tc.get("lr_classical", 0.01)
        elif name == "matched_mlp":
            q_params = int(np.prod(spec.weight_shape)) + len(spec.observables()) + 1
            model, lr = MatchedMLP(n_items, q_params), tc.get("lr_classical", 0.01)
        elif name in ("quantum", "fused"):
            model = None
        else:
            raise ValueError(name)
        kw = {"epochs": tc.get("epochs", 300), "batch": tc.get("batch", 128),
              "patience": tc.get("patience", 30)}
        if model is not None:
            info = train(model, tr, va, seed=seed, lr=lr, **kw)
        else:
            # quantum models: several random restarts, keep the best on validation (improvement 1)
            restarts = m.get("restarts", tc.get("restarts", 1))
            best, calls, val_per_restart = None, 0, []
            for r in range(restarts):
                init_seed = seed * 1000 + r
                layer = QuantumInteractionLayer(spec, seed=init_seed)
                if name == "quantum":
                    cand, lr = QuantumRegressor(layer, amp_max), tc.get("lr_quantum", 0.05)
                else:
                    cand, lr = FusedForecaster(layer, amp_max), tc.get("lr_classical", 0.01)
                cinfo = train(cand, tr, va, seed=init_seed, lr=lr, l1=m.get("l1", 0.0), **kw)
                calls += layer.calls
                val_per_restart.append(cinfo["best_val_mse"])
                if best is None or cinfo["best_val_mse"] < best[1]["best_val_mse"]:
                    best = (cand, cinfo, r)
            model, info, chosen = best
            info = {**info, "restarts": restarts, "chosen_restart": chosen,
                    "val_mse_per_restart": val_per_restart}
            model.layer.calls = calls
        pred_fn = lambda a, p: predict(model, a, p)
        rec.update(n_params=sum(p.numel() for p in model.parameters()), train_info=info)
        if name in ("quantum", "fused"):
            rec["resources"] = resources(spec)
            # samples pushed through the circuit during training (incl. per-epoch validation),
            # times measurement settings = circuit executions on hardware-like accounting
            rec["train_samples_through_circuit"] = model.layer.calls
            rec["train_circuit_executions"] = model.layer.calls * rec["resources"]["measurement_settings"]
            rec["circuit_executions_per_prediction"] = rec["resources"]["measurement_settings"]
            if spec.learned:
                rec["edges"] = spec.edges
                rec["edge_strength"] = model.layer.edge_strength().tolist()
                rec["ent_weights"] = model.layer.ent.detach().numpy().tolist()
            if name == "quantum" and cfg.get("shots_eval"):
                rec["shots_eval"] = shots_eval(model, te, cfg["shots_eval"], cfg.get("shots_eval_n", 500))
                rec["weights"] = model.layer.weights.detach().numpy().tolist()
    rec["wall_seconds"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    p_te = pred_fn(te[0], te[1])
    rec["inference_seconds_per_sample"] = (time.perf_counter() - t1) / len(te[2])
    rec.update(test_mse=mse(te[2], p_te), test_mae=mae(te[2], p_te),
               train_mse=mse(tr[2], pred_fn(tr[0], tr[1])))
    write_record(out_path, rec)
    return f"done {out_path.name}  test_mse={rec['test_mse']:.5f}  {rec['wall_seconds']:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--quick", action="store_true", help="1 seed, smallest size, 20 epochs")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--only", default=None, help="substring filter on model tag")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.quick:
        cfg["seeds"], cfg["train_sizes"] = cfg["seeds"][:1], cfg["train_sizes"][:1]
        cfg.setdefault("train", {})["epochs"] = 20
        cfg["name"] = cfg["name"] + "_quick"
    out_dir = ROOT / "outputs" / "module_a" / cfg["name"]
    jobs = []
    for m, n, s in itertools.product(cfg["models"], cfg["train_sizes"], cfg["seeds"]):
        if args.only and args.only not in model_tag(m):
            continue
        jobs.append({"cfg": cfg, "model": m, "n_train": n, "seed": s,
                     "out": str(out_dir / f"{model_tag(m)}_n{n}_s{s}.json")})
    # slow jobs first for better packing
    jobs.sort(key=lambda j: (j["model"]["name"] not in ("quantum", "fused"), -j["n_train"]))
    print(f"{len(jobs)} jobs -> {out_dir}  ({args.workers} workers)", flush=True)
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(run_job, jobs), 1):
            print(f"[{i}/{len(jobs)} {time.perf_counter() - t0:6.0f}s] {msg}", flush=True)


if __name__ == "__main__":
    main()
