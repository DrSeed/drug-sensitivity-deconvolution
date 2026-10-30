# Two-step deconvolution in the spirit of CLIMB/CLIFF: bulk -> cell-type proportions -> per-type drug survival.
import numpy as np
from scipy.optimize import nnls


def signature_nnls(bulk, profiles):
    """Classic deconvolution: bulk ~ mean profile per type, non-negative, then normalised to sum 1."""
    w = np.array([nnls(profiles.T, b)[0] for b in bulk])
    return w / w.sum(axis=1, keepdims=True)


def cell_level_nnls(bulk, ref_cells, ref_types, n_types):
    """CLIMB idea: express each bulk sample as a non-negative mix of individual reference cells,
    then add up the weights by cell type. Within-type variety in the reference is kept."""
    out = np.zeros((len(bulk), n_types))
    for i, b in enumerate(bulk):
        alpha = nnls(ref_cells.T, b)[0]
        out[i] = np.bincount(ref_types, weights=alpha, minlength=n_types)
    return out / out.sum(axis=1, keepdims=True)


def cell_type_survival_em(W, y, n_iter=2000, tol=1e-9):
    """CLIFF idea, without covariates: each sample's surviving fraction y_n = sum_k W_nk * pi_k.
    Which cells survived is hidden, so EM alternates between attributing survivors and deaths to
    cell types (E-step) and re-estimating each type's survival pi_k (M-step)."""
    pi = np.full(W.shape[1], np.clip(y.mean(), 0.05, 0.95))
    for _ in range(n_iter):
        alive, dead = W * pi, W * (1 - pi)
        q1 = alive / alive.sum(axis=1, keepdims=True)
        q0 = dead / dead.sum(axis=1, keepdims=True)
        a, b = (q1 * y[:, None]).sum(0), (q0 * (1 - y)[:, None]).sum(0)
        new = a / (a + b)
        if np.abs(new - pi).max() < tol:
            return new
        pi = new
    return pi


def bootstrap_survival(W, y, n_boot=200, seed=0):
    """Resample patients to see which cell types' sensitivities the cohort can actually pin down."""
    rng = np.random.default_rng(seed)
    draws = [cell_type_survival_em(W[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return np.array(draws)
