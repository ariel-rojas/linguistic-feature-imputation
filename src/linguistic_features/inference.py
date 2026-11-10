"""NUTS, diagnósticos y probabilidades posteriores multirrespuesta."""

import time

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS
from scipy.special import ndtr

from .design import harmonic_scales
from .model import spatiophylogenetic_multivariate_probit


def sample_posterior(
    basis, degrees, phylo_factor, labels, observed_mask, settings, seed
):
    """Obtiene draws de parámetros y campos latentes mediante NUTS.

    Las cadenas se guardan separadas para poder calcular diagnósticos de
    convergencia antes de resumir o adelgazar los draws para predicción.
    """
    started = time.perf_counter()
    sampler = MCMC(
        NUTS(
            spatiophylogenetic_multivariate_probit,
            target_accept_prob=settings["target_accept"],
            max_tree_depth=settings["max_tree_depth"],
        ),
        num_warmup=settings["warmup"],
        num_samples=settings["draws"],
        num_chains=settings["chains"],
        chain_method="sequential",
        progress_bar=True,
    )
    print(
        f"NUTS multirrespuesta: {settings['chains']} cadenas; "
        f"{settings['warmup']} warmup + {settings['draws']} draws por cadena",
        flush=True,
    )
    sampler.run(
        jax.random.PRNGKey(seed),
        jnp.asarray(basis),
        jnp.asarray(degrees),
        jnp.asarray(phylo_factor),
        settings,
        jnp.asarray(labels),
        jnp.asarray(observed_mask),
        extra_fields=("diverging", "num_steps", "accept_prob"),
    )
    samples = {
        name: np.asarray(value)
        for name, value in sampler.get_samples(group_by_chain=True).items()
    }
    extras = {
        name: np.asarray(value)
        for name, value in sampler.get_extra_fields(group_by_chain=True).items()
    }
    diagnostics = diagnose(samples, extras, settings)
    diagnostics["seconds"] = time.perf_counter() - started
    return samples, extras, diagnostics


def diagnose(samples, extras, settings):
    """Resume convergencia y calcula correlaciones posteriores entre rasgos."""
    summary = az.summary(
        az.from_dict({"posterior": samples}),
        var_names=[
            "max_degree", "smoothing", "intercept", "sigma_spatial",
            "sigma_phylogenetic", "spatial_trait_correlations",
            "phylogenetic_trait_correlations",
        ],
        round_to=8,
    )
    max_rhat = float(summary.r_hat.max())
    ess_bulk = float(summary.ess_bulk.min())
    ess_tail = float(summary.ess_tail.min())
    divergences = int(extras["diverging"].sum())
    depth_hits = int(
        (extras["num_steps"] >= 2 ** settings["max_tree_depth"] - 1).sum()
    )
    return {
        "passed": bool(
            np.isfinite(summary[["r_hat", "ess_bulk", "ess_tail"]]).all().all()
            and max_rhat < 1.01 and min(ess_bulk, ess_tail) >= 400
            and divergences == 0 and depth_hits == 0
        ),
        "max_rhat": max_rhat,
        "min_ess_bulk": ess_bulk,
        "min_ess_tail": ess_tail,
        "divergences": divergences,
        "tree_depth_hits": depth_hits,
        "mean_acceptance": float(extras["accept_prob"].mean()),
        "criteria": "R-hat < 1.01; ESS bulk y tail >= 400; cero divergencias y saturaciones de profundidad.",
        "global_parameters": summary.reset_index().to_dict("records"),
        "degree_posterior": {
            "mean": float(samples["max_degree"].mean()),
            "median": float(np.median(samples["max_degree"])),
            "interval_95": np.quantile(samples["max_degree"], [0.025, 0.975]).tolist(),
            "prior": [settings["degree_min"], settings["degree_max"]],
        },
        "trait_correlations": {
            "spatial": correlation_summary(
                samples["spatial_trait_correlation_cholesky"]
            ),
            "phylogenetic": correlation_summary(
                samples["phylogenetic_trait_correlation_cholesky"]
            ),
        },
    }


def correlation_summary(cholesky_samples):
    """Resume cada correlación entre rasgos a partir de una Cholesky LKJ."""
    draws = cholesky_samples.reshape((-1, *cholesky_samples.shape[-2:]))
    correlations = draws @ np.swapaxes(draws, -1, -2)
    count = correlations.shape[-1]
    return {
        f"{i}-{j}": {
            "mean": float(correlations[:, i, j].mean()),
            "interval_95": np.quantile(
                correlations[:, i, j], [0.025, 0.975]
            ).tolist(),
        }
        for i in range(count) for j in range(i + 1, count)
    }


def flatten_samples(samples, maximum):
    """Aplana cadenas y selecciona draws aproximadamente equiespaciados."""
    first = next(iter(samples.values()))
    count = first.shape[0] * first.shape[1]
    chosen = np.linspace(0, count - 1, min(maximum, count), dtype=int)
    return {
        name: value.reshape((-1, *value.shape[2:]))[chosen]
        for name, value in samples.items()
    }


def probability_draws(
    samples, basis, degrees, conditional_mean, residual_factor, seed, settings
):
    """Genera draws de probabilidades por lengua y rasgo, conservando correlación."""
    rng = np.random.default_rng(seed)
    count, n_target, n_traits = (
        len(samples["intercept"]), len(basis), len(samples["intercept"][0])
    )
    probabilities = np.empty((count, n_target, n_traits), dtype=np.float32)
    for draw in range(count):
        scales = harmonic_scales(
            degrees, samples["max_degree"][draw], samples["smoothing"][draw],
            settings["degree_taper_width"],
        )
        spatial_standard = basis @ (
            scales[:, None] * samples["z_spatial"][draw]
        )
        spatial = spatial_standard @ samples["spatial_trait_cholesky"][draw].T

        phylo_standard = conditional_mean @ samples["z_phylogenetic"][draw]
        if residual_factor.shape[1]:
            # Los objetivos comparten la parte explicada por train; esta
            # innovación añade su incertidumbre filogenética condicional.
            innovations = rng.normal(size=(residual_factor.shape[1], n_traits))
            phylo_standard += residual_factor @ innovations
        phylogenetic = (
            phylo_standard @ samples["phylogenetic_trait_cholesky"][draw].T
        )
        eta = samples["intercept"][draw][None, :] + spatial + phylogenetic
        probabilities[draw] = ndtr(eta)
    return probabilities


def summarize_probability_draws(draws):
    """Resume la media, el intervalo creíble y la clase por rasgo."""
    mean = draws.mean(axis=0)
    lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "predicted": (mean >= 0.5).astype(int),
    }
