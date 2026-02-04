import os, glob
import numpy as np
import pandas as pd

RESULTS_DIR, FINAL_DIR = "results", "final"
os.makedirs(FINAL_DIR, exist_ok=True)


def _to_py(x):
    return x.item() if isinstance(x, np.generic) else x

def pick_best_setup(df: pd.DataFrame):
    run_cols = {"seed", "acc", "correct", "total", "elapsed_sec"}
    setup_cols = [c for c in df.columns if c not in run_cols]
    best = {"mean": -np.inf, "std": np.inf, "setup_key": None, "setup_params": None, "top3_acc": None, "top3_times": None}

    for key, g in df.groupby(setup_cols, dropna=False):
        if g["seed"].nunique() != 4:
            continue
        top3 = g.sort_values("acc", ascending=False).head(3)
        t3_acc, t3_times = top3["acc"].to_numpy(), top3["elapsed_sec"].to_numpy()
        m, s = float(np.mean(t3_acc)), float(np.std(t3_acc, ddof=0))
        if m > best["mean"] or (np.isclose(m, best["mean"]) and s < best["std"]):
            kv = key if isinstance(key, tuple) else (key,)
            best.update({"mean": m, "std": s, "setup_key": key,
                         "setup_params": {k: _to_py(v) for k, v in zip(setup_cols, kv)},
                         "top3_acc": t3_acc, "top3_times": t3_times})
    return best if best["setup_key"] is not None else None

def latex_from_percent(mean_dec: float, std_dec: float) -> str:
    return f'{round(mean_dec * 100, 2)}' + r'\textsubscript{\textcolor{gray}{±' + f'{round(std_dec * 100, 2)}' + r'}}'

def process_file(csv_path: str):
    df = pd.read_csv(csv_path)
    best = pick_best_setup(df)
    base = os.path.basename(csv_path)
    out = os.path.join(FINAL_DIR, os.path.splitext(base)[0] + ".txt")

    if best is None:
        content = f"------- Best setup ----------\n\nNo valid setup with four seeds found in {base}.\n"
    else:
        a1, a2, a3 = best["top3_acc"]
        m_pct, s_pct = float(np.mean(best["top3_acc"]) * 100), float(np.std(best["top3_acc"], ddof=0) * 100)
        t_m, t_s = float(np.mean(best["top3_times"])), float(np.std(best["top3_times"], ddof=0))
        lines = ["------- Best setup ----------\n"] + [f"{k}: {v}" for k, v in best["setup_params"].items()]
        lines += ["", f"acc1: {a1:.10f}", f"acc2: {a2:.10f}", f"acc3: {a3:.10f}", f"std_acc: {s_pct:.2f}",
                  f"average accuracy: {m_pct:.2f}", f"std_time: {t_s:.2f}", f"average time: {t_m:.2f}",
                  "", f"latex: {latex_from_percent(best['mean'], best['std'])}"]
        content = "\n".join(lines) + "\n"

    with open(out, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    csvs = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    if not csvs:
        print(f"No CSV files found in '{RESULTS_DIR}'.")
        return
    for c in csvs:
        process_file(c)
        print(f"Processed: {c}")

if __name__ == "__main__":
    main()
