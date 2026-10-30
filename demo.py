# Self-contained demo: a simulated AML-like cohort with bulk RNA, ex-vivo drug survival and a scRNA reference.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deconv import signature_nnls, cell_level_nnls, cell_type_survival_em, bootstrap_survival

TYPES = ["stem-like", "progenitor", "monocytic", "granulocytic", "erythroid", "T cell"]
MEAN_SHARE = np.array([0.12, 0.30, 0.25, 0.15, 0.05, 0.13])


def simulate(n_genes=300, n_patients=80, cells_per_state=30, seed=3):
    rng = np.random.default_rng(seed)
    K = len(TYPES)
    base = rng.lognormal(0, 1, n_genes)
    # each type has two sub-states (e.g. quiescent vs cycling) with partly different genes
    states = np.zeros((K, 2, n_genes))
    for k in range(K):
        prof = base * np.where(rng.uniform(size=n_genes) < 0.12, rng.lognormal(1.5, 0.5, n_genes), 1)
        for s in range(2):
            states[k, s] = prof * np.where(rng.uniform(size=n_genes) < 0.08, rng.lognormal(1.2, 0.4, n_genes), 1)

    # reference scRNA-seq: cells from both sub-states, reference donors use mostly sub-state 0
    ref_cells, ref_types = [], []
    for k in range(K):
        for s, n in [(0, 2 * cells_per_state), (1, cells_per_state)]:
            mu = states[k, s] * 2.0
            ref_cells.append(rng.poisson(mu * rng.lognormal(0, 0.2, (n, 1))))
            ref_types += [k] * n
    ref_cells, ref_types = np.vstack(ref_cells).astype(float), np.array(ref_types)
    ref_norm = ref_cells / ref_cells.sum(1, keepdims=True)
    profiles = np.array([ref_norm[ref_types == k].mean(0) for k in range(K)])

    # patients: varied composition and varied sub-state mix; bulk = mixture + per-gene platform noise
    W = rng.dirichlet(MEAN_SHARE * 12, n_patients)
    mix = rng.uniform(0, 1, (n_patients, K))
    bulk = np.einsum("nk,nkg->ng", W, mix[:, :, None] * states[:, 1] + (1 - mix[:, :, None]) * states[:, 0])
    bulk *= rng.lognormal(0, 0.15, n_genes)
    bulk = bulk / bulk.sum(1, keepdims=True)

    # ex-vivo drug screen: survival per cell type, mixed by composition, plus assay noise
    drugs = {"drug A (hits stem-like)": [0.15, 0.80, 0.85, 0.85, 0.90, 0.95],
             "drug B (hits monocytic)": [0.85, 0.80, 0.20, 0.60, 0.90, 0.95],
             "drug C (broad cytotoxic)": [0.50, 0.35, 0.40, 0.40, 0.45, 0.60],
             "drug D (hits erythroid)": [0.90, 0.85, 0.90, 0.85, 0.10, 0.95]}
    PI = np.array(list(drugs.values()))
    Y = np.clip(W @ PI.T + rng.normal(0, 0.04, (n_patients, len(drugs))), 0.001, 0.999)
    return bulk, W, ref_norm, ref_types, profiles, PI, Y, list(drugs)


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    bulk, W, ref_cells, ref_types, profiles, PI, Y, drugs = simulate()
    K = len(TYPES)
    W_sig = signature_nnls(bulk, profiles)
    W_cell = cell_level_nnls(bulk, ref_cells, ref_types, K)

    rows = []
    for name, What in [("mean-signature NNLS", W_sig), ("cell-level NNLS (CLIMB-style)", W_cell)]:
        rows.append({"step": "1. composition", "method": name, "drug": "",
                     "rmse": np.sqrt(((What - W) ** 2).mean()), "pearson_r": np.corrcoef(What.ravel(), W.ravel())[0, 1]})
    est = {}
    for name, What in [("true proportions", W), ("cell-level NNLS (CLIMB-style)", W_cell)]:
        est[name] = np.array([cell_type_survival_em(What, Y[:, d]) for d in range(len(drugs))])
        for d, drug in enumerate(drugs):
            rows.append({"step": "2. drug survival per type", "method": f"EM on {name}", "drug": drug,
                         "rmse": np.sqrt(((est[name][d] - PI[d]) ** 2).mean()),
                         "pearson_r": np.corrcoef(est[name][d], PI[d])[0, 1]})
    table = pd.DataFrame(rows).round(3)
    table.to_csv("results/summary.csv", index=False)
    pd.DataFrame(est["cell-level NNLS (CLIMB-style)"].round(3), index=drugs, columns=TYPES).to_csv(
        "results/estimated_survival_by_cell_type.csv")

    boot = bootstrap_survival(W_cell, Y[:, 3])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for What, c, lab in [(W_sig, "#c44e52", "mean-signature NNLS"), (W_cell, "#4c72b0", "cell-level NNLS")]:
        r = np.sqrt(((What - W) ** 2).mean())
        axes[0].scatter(W.ravel(), What.ravel(), s=6, alpha=0.5, color=c, label=f"{lab} (RMSE {r:.3f})")
    axes[0].plot([0, 0.8], [0, 0.8], "k--", lw=0.8)
    axes[0].set_xlabel("true cell-type proportion")
    axes[0].set_ylabel("estimated proportion")
    axes[0].set_title("Step 1: mapping bulk onto single cells beats\nthe mean profile when sub-states vary")
    axes[0].legend(fontsize=8)

    E = est["cell-level NNLS (CLIMB-style)"]
    markers = ["o", "s", "^", "D"]
    for d, drug in enumerate(drugs):
        axes[1].scatter(PI[d], E[d], marker=markers[d], s=45, label=drug)
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].set_xlabel("true survival of the cell type under the drug")
    axes[1].set_ylabel("estimated from bulk survival only")
    axes[1].set_title("Step 2: EM recovers which cell type each drug kills")
    axes[1].legend(fontsize=8)

    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    pd.DataFrame({"cell_type": TYPES, "mean_share": W.mean(0).round(3), "ci_width_drug_D": (hi - lo).round(3)}).to_csv(
        "results/identifiability_drug_D.csv", index=False)
    x = np.arange(K)
    axes[2].errorbar(x, E[3], yerr=[E[3] - lo, hi - E[3]], fmt="o", color="#4c72b0", capsize=4, label="estimate, 95% bootstrap CI")
    axes[2].scatter(x, PI[3], marker="x", color="k", s=50, label="truth")
    for xi, share in zip(x, W.mean(0)):
        axes[2].text(xi, -0.08, f"{100 * share:.0f}%", ha="center", fontsize=8, color="#555555")
        axes[2].text(xi + 0.12, E[3][xi], f"±{(hi[xi] - lo[xi]) / 2:.2f}", fontsize=8, va="center")
    axes[2].set_xticks(x, TYPES, rotation=20)
    axes[2].set_ylim(-0.15, 1.05)
    axes[2].set_ylabel("survival under drug D")
    axes[2].set_title("Rare types get wide intervals\n(grey: mean share of each type in the cohort)")
    axes[2].legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig("figures/demo.png", dpi=120)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
