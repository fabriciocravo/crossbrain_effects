import numpy as np
from parameters import (
    VARIABLE_NUMBER,
    N_MAJOR_REPS,
    DATASET_SIZE,
    HEAVY_TAILED_PARS,
    NORMAL_PARS,
    MISTURE_PARS,
)
from models import (
    normal_model,
    heavy_tailed_model,
    CenterMistureModel,
)
from dataset_sampling import empirical_sample


N_ROIS = 268
N_EDGES = N_ROIS * (N_ROIS - 1) // 2  # 35,778 FC edges

# |effect| ranges: 0-0.1 / 0.1-0.2 / 0.2-0.5 / above 0.5
BIN_EDGES = [0.0, 0.1, 0.2, 0.5, np.inf]
BIN_LABELS = ["0-0.1", "0.1-0.2", "0.2-0.5", ">0.5"]

MODEL_FAMILIES = [
    ("normal", normal_model, NORMAL_PARS),
    ("heavy-tailed", heavy_tailed_model, HEAVY_TAILED_PARS),
    ("center-mixture", CenterMistureModel, MISTURE_PARS),
]


def build_distributions():
    return [
        (name, pars, builder(*pars))
        for name, builder, par_list in MODEL_FAMILIES
        for pars in par_list
    ]


def expected_counts(dist):
    """Expected number of edges with |effect| in each bin."""
    counts = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        # Calculates fraction of effects within the range
        fraction = (dist.cdf(hi) - dist.cdf(lo)) + (dist.cdf(-lo) - dist.cdf(-hi))
        counts.append(N_EDGES * fraction)

    return np.array(counts)


def tv_distance(true_dist, fitted_dist):

    lo = fitted_dist.ppf(1e-15)
    hi = fitted_dist.isf(1e-15)
    x = np.linspace(lo, hi, 10000)
    dx = x[1] - x[0]
    # Continuous part of the true dist plus its point mass (the fitted
    # normal has no atom, so the atom contributes fully to the TV)
    atom = getattr(true_dist, "atom_mass", 0.0)
    return 0.5 * (
        np.sum(np.abs(true_dist.pdf(x) - fitted_dist.pdf(x))) * dx + atom
    )


def run_comparison(true_dist):

    true_counts = expected_counts(true_dist)
    tv_sum = 0.0
    count_diff = np.zeros(len(BIN_EDGES) - 1)

    for _ in range(N_MAJOR_REPS):

        fitted_dist = empirical_sample(
            true_dist,
            VARIABLE_NUMBER,
            DATASET_SIZE
        )
        
        tv_sum += tv_distance(true_dist, fitted_dist)
        count_diff += np.abs(true_counts - expected_counts(fitted_dist))

    return tv_sum / N_MAJOR_REPS, count_diff / N_MAJOR_REPS, true_counts


# Code
results = []
for name, pars, true_dist in build_distributions():

    true_sd = np.std(true_dist.rvs(200_000))
    tv, count_diff, true_counts = run_comparison(true_dist)
    results.append((name, pars, true_sd, tv, count_diff, true_counts))
    print(f"done: {name} {pars}", flush=True)


# Table 1 - divergence and error in the number of effects
header = (
    f"{'model':<13}{'params':<24}{'true SD':>9}{'TV':>11}"
    + "".join(f"{'|d| ' + lab:>13}" for lab in BIN_LABELS)
)
print(f"\nMean over {N_MAJOR_REPS} reps; counts out of {N_EDGES} edges")
print(header)
print("-" * len(header))
for name, pars, true_sd, kl, count_diff, _ in results:
    print(
        f"{name:<13}{str(pars):<24}{true_sd:>9.4f}{kl:>11.6f}"
        + "".join(f"{100 * d / N_EDGES:>12.2f}%" for d in count_diff)
    )
