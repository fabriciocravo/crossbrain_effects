import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from parameters import(
    VARIABLE_NUMBER,
    N_MAJOR_REPS,
    DATASET_SIZE
)
from models import(
    true_normal_dist
)
from dataset_sampling import empirical_sample

# Change this to change model
true_dist = true_normal_dist
z_eff_diff = 0
s_eff_diff = 0
m_eff_diff = 0
l_eff_diff = 0

# Code  
mean_divergence = 0
for _ in range(N_MAJOR_REPS):
   
    fitted_dist = empirical_sample(
        true_dist,
        VARIABLE_NUMBER,
        DATASET_SIZE
    )

    # KL DIVERGENCE (ENTROPY PART)
    # Shared grid — use ppf to get a sensible range covering both distributions' mass
    lo = min(true_dist.ppf(0.0001), fitted_dist.ppf(0.0001))
    hi = max(true_dist.ppf(0.9999), fitted_dist.ppf(0.9999))
    x = np.linspace(lo, hi, 10000)

    p = true_dist.pdf(x)
    q = fitted_dist.pdf(x)

    # Avoid division by zero
    eps = 1e-10
    kl_div = entropy(p + eps, q + eps)

    mean_divergence += kl_div

    # EFFECT NUMBER PART
    true_effects = true_dist.rvs(VARIABLE_NUMBER)
    fitted_effects = fitted_dist.rvs(VARIABLE_NUMBER)

    bins = [0.2, 0.5, 0.8]  # Cohen's d cutoffs: negligible/small/medium/large

    true_counts = np.bincount(
        np.digitize(np.abs(true_effects), bins), minlength=4
    )
    fitted_counts = np.bincount(
        np.digitize(np.abs(fitted_effects), bins), minlength=4
    )

    z_eff_diff += abs(true_counts[0] - fitted_counts[0])
    s_eff_diff += abs(true_counts[1] - fitted_counts[1])
    m_eff_diff += abs(true_counts[2] - fitted_counts[2])
    l_eff_diff += abs(true_counts[3] - fitted_counts[3])

    last_fitted_dist = fitted_dist

mean_divergence = mean_divergence / N_MAJOR_REPS
z_eff_diff /= N_MAJOR_REPS
s_eff_diff /= N_MAJOR_REPS
m_eff_diff /= N_MAJOR_REPS
l_eff_diff /= N_MAJOR_REPS

print(f"Mean KL divergence over {N_MAJOR_REPS} reps: {mean_divergence:.6f}")
print(f"Mean |Δcount| negligible: {z_eff_diff:.2f}")
print(f"Mean |Δcount| small:      {s_eff_diff:.2f}")
print(f"Mean |Δcount| medium:     {m_eff_diff:.2f}")
print(f"Mean |Δcount| large:      {l_eff_diff:.2f}")

# --- Representative plot (uses the last rep's fitted distribution) ---
lo = min(true_dist.ppf(0.0001), last_fitted_dist.ppf(0.0001))
hi = max(true_dist.ppf(0.9999), last_fitted_dist.ppf(0.9999))
x = np.linspace(lo, hi, 1000)

plt.figure(figsize=(8, 5))
plt.plot(
    x,
    true_dist.pdf(x),
    label="True distribution",
    linewidth=2
)
plt.plot(
    x,
    last_fitted_dist.pdf(x),
    label="Fitted (empirical) distribution",
    linewidth=2,
    linestyle="--"
)
plt.fill_between(x, true_dist.pdf(x), alpha=0.15)
plt.fill_between(x, last_fitted_dist.pdf(x), alpha=0.15)
plt.title(
    "True vs. Fitted Distribution"
    f"(mean KL divergence = {mean_divergence:.4f})"
)
plt.xlabel("Effect size")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()



