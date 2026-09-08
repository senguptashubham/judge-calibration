"""C9 (LEARNING.md) - toy hierarchical logistic regression in NumPyro.
Practice for task 1.3b: fit synthetic data with a KNOWN group effect, then
check it can be recovered from the posterior. Deliberately rehearses the
same model shape as the real thing (DECISIONS.md D22, task 5.9b's
src/bayesian.py, built in W5): correct ~ Bernoulli(alpha_q[group]),
alpha_q ~ Normal(0, sigma_q) - just without the real project's features
(Xβ) or real data yet.

Not part of the graded pipeline: no tests, nothing here is imported by
src/. Disposable practice code - run directly:
    python learning/c9_hierarchical_toy.py
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def simulate_data(
    n_groups: int, sigma_true: float, group_sizes: list[int], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate data with a KNOWN hierarchical structure, so you have
    something to check the posterior against afterward.

    TODO:
      1. Draw one true intercept per group from Normal(0, sigma_true) -
         these are the `true_alphas` you'll try to recover later.
      2. For each group g, simulate group_sizes[g] independent Bernoulli
         outcomes using that group's true intercept as the logit
         (P(y=1) = sigmoid(true_alphas[g])).
      3. Return everything flattened to one row per observation (not one
         row per group) - group_idx says which group each observation in
         y belongs to.

    Returns:
      group_idx: shape (n_obs,) int array, which group each observation
        belongs to (values 0..n_groups-1).
      y: shape (n_obs,) int array of 0/1 outcomes.
      true_alphas: shape (n_groups,) - the ground truth to check against.
    """
    rng = np.random.default_rng(seed)
    # 1. Draw the true intercept for each group
    true_alphas = rng.normal(
        loc=0.0,
        scale=sigma_true,
        size=n_groups
    )
    # 2. Generate observations for each group
    group_idx = []
    y = []

    for g in range(n_groups):
        # Probability of y=1 for this group
        p = 1 / (1 + np.exp(-true_alphas[g]))

        # Generate this group's observations
        y_g = rng.binomial(
            n=1,
            p=p,
            size=group_sizes[g]
        )

        group_idx.extend([g] * group_sizes[g])
        y.extend(y_g)

    # 3. Convert to NumPy arrays
    group_idx = np.asarray(group_idx, dtype=int)
    y = np.asarray(y, dtype=int)

    return group_idx, y, true_alphas


def model(group_idx: jnp.ndarray, n_groups: int, y: jnp.ndarray | None = None) -> None:
    """The hierarchical model itself - this is the part that matters most
    for C9's Extract questions.

    TODO:
      1. sigma_q: a prior over how spread-out the group intercepts are
         (e.g. HalfNormal(1.0) - must be positive, since it's a std dev).
      2. alpha_q: one intercept per group, drawn from Normal(0, sigma_q),
         sampled inside a `numpyro.plate("groups", n_groups)`.
      3. y: Bernoulli(logits=alpha_q[group_idx]), sampled inside a
         `numpyro.plate("obs", len(group_idx))`, with `obs=y` so NumPyro
         knows these are observed (not latent) during inference.
    """
    # 1. Population-level standard deviation
    sigma_q = numpyro.sample(
        "sigma_q",
        dist.HalfNormal(1.0)
    )

    # 2. One intercept per group
    with numpyro.plate("groups", n_groups):
        alpha_q = numpyro.sample(
            "alpha_q",
            dist.Normal(0.0, sigma_q)
        )

    # 3. Bernoulli observations
    with numpyro.plate("obs", len(group_idx)):
        numpyro.sample(
            "y",
            dist.Bernoulli(logits=alpha_q[group_idx]),
            obs=y
        )


def run_mcmc(
    group_idx: np.ndarray, n_groups: int, y: np.ndarray, seed: int, num_warmup: int = 1000, num_samples: int = 1000
) -> dict:
    """TODO: NUTS(model) -> MCMC(nuts, num_warmup=..., num_samples=...) ->
    .run(jax.random.PRNGKey(seed), group_idx, n_groups, y) -> .get_samples().

    Also worth printing `mcmc.print_summary()` before returning - it shows
    r_hat and effective sample size per parameter, the same convergence
    diagnostics DECISIONS.md D22 will make mandatory for the real model.
    """
     # 1. Create the NUTS kernel
    nuts_kernel = NUTS(model)

    # 2. Create the MCMC sampler
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples
    )

    # 3. Run NUTS
    rng_key = jax.random.PRNGKey(seed)

    mcmc.run(
        rng_key,
        group_idx=group_idx,
        n_groups=n_groups,
        y=y
    )

    # 4. Print diagnostics
    mcmc.print_summary()

    # 5. Return posterior samples
    return mcmc.get_samples()


def check_recovery(samples: dict, true_alphas: np.ndarray, group_sizes: list[int], sigma_true: float) -> None:
    """Does the posterior actually recover what you simulated?

    TODO:
      1. Compare the posterior mean of `sigma_q` against `sigma_true` -
         print both.
      2. For each group, compare its posterior mean `alpha_q` against its
         true value. Scatter plot: x = true alpha, y = posterior mean
         alpha, one point per group, sized or colored by group_sizes.
         Points should sit close to the y=x diagonal - and the SMALL-N
         groups should sit further from the diagonal (pulled toward the
         population mean) than the LARGE-N groups. That gap is partial
         pooling, made visible rather than just described.
      3. Save the figure (plt.savefig) so you can actually look at it.
    """
    # 1. Posterior estimate of population-level sigma, vs. the true value
    # used to simulate it. With only n_groups groups this won't match
    # exactly - see the 90% credible interval printed by print_summary()
    # for whether sigma_true falls within the posterior's own uncertainty,
    # not just whether the point estimate lands on it.
    sigma_posterior_mean = samples["sigma_q"].mean()
    print(f"sigma_q: true = {sigma_true:.3f}, posterior mean = {sigma_posterior_mean:.3f}")

    # 2. Posterior mean of each group's alpha
    alpha_posterior_mean = samples["alpha_q"].mean(axis=0)

    print("\nGroup recovery:")
    for g in range(len(true_alphas)):
        print(
            f"Group {g:2d}: "
            f"true alpha = {true_alphas[g]: .3f}, "
            f"posterior mean = {alpha_posterior_mean[g]: .3f}, "
            f"n = {group_sizes[g]}"
        )

    # 3. Plot true alpha vs posterior mean alpha
    plt.figure(figsize=(8, 8))

    plt.scatter(
        true_alphas,
        alpha_posterior_mean,
        s=np.asarray(group_sizes) * 5,
        alpha=0.7
    )

    # y = x diagonal
    min_alpha = min(true_alphas.min(), alpha_posterior_mean.min())
    max_alpha = max(true_alphas.max(), alpha_posterior_mean.max())

    plt.plot(
        [min_alpha, max_alpha],
        [min_alpha, max_alpha],
        linestyle="--"
    )

    plt.xlabel("True group intercept")
    plt.ylabel("Posterior mean group intercept")
    plt.title("Hierarchical Logistic Regression: Parameter Recovery")
    plt.grid(alpha=0.2)

    plt.savefig("parameter_recovery.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    n_groups = 8
    sigma_true = 1.0
    # Deliberately uneven group sizes - this unevenness is what makes
    # partial pooling visible in check_recovery()'s plot.
    group_sizes = [5, 8, 10, 15, 20, 30, 40, 50]
    seed = 0

    group_idx, y, true_alphas = simulate_data(n_groups, sigma_true, group_sizes, seed)
    samples = run_mcmc(group_idx, n_groups, y, seed)
    check_recovery(samples, true_alphas, group_sizes, sigma_true)
