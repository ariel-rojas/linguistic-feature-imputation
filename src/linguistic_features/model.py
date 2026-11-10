"""Modelo probit bayesiano multirrespuesta con dos campos correlacionados."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.special import log_ndtr
import numpyro
import numpyro.distributions as dist


def spatiophylogenetic_multivariate_probit(
    spatial_basis, degrees, phylogenetic_factor, settings, labels=None,
    observed_mask=None,
):
    """Modela varias respuestas binarias con covarianza espacial y filogenética.

    Las etiquetas observadas son 0/1. Cada una tiene un predictor probit propio,
    mientras las matrices de covarianza entre rasgos permiten compartir señal
    espacial y filogenética entre respuestas.
    """
    n_traits = labels.shape[1] if labels is not None else settings["n_traits"]
    max_degree = numpyro.sample(
        "max_degree", dist.Uniform(settings["degree_min"], settings["degree_max"])
    )
    smoothing = numpyro.sample(
        "smoothing",
        dist.LogNormal(
            jnp.log(settings["smoothing_median"]), settings["smoothing_log_sd"]
        ),
    )
    intercept = numpyro.sample(
        "intercept",
        dist.Normal(0, settings["intercept_scale"])
        .expand((n_traits,)).to_event(1),
    )

    # Cada campo tiene su propia matriz de covarianza entre rasgos:
    # Sigma = diag(sigma) @ Omega @ diag(sigma), con Omega correlación LKJ.
    # Separar espacio y filogenia permite que compartan fuerza entre rasgos
    # de maneras distintas.
    sigma_spatial = numpyro.sample(
        "sigma_spatial",
        dist.HalfNormal(settings["amplitude_scale"])
        .expand((n_traits,)).to_event(1),
    )
    spatial_correlation_cholesky = numpyro.sample(
        "spatial_trait_correlation_cholesky",
        dist.LKJCholesky(
            dimension=n_traits,
            concentration=settings["trait_correlation_concentration"],
        ),
    )
    spatial_trait_cholesky = sigma_spatial[:, None] * spatial_correlation_cholesky
    numpyro.deterministic("spatial_trait_cholesky", spatial_trait_cholesky)
    spatial_correlation = spatial_correlation_cholesky @ spatial_correlation_cholesky.T
    upper_triangle = jnp.triu_indices(n_traits, k=1)
    numpyro.deterministic(
        "spatial_trait_correlations", spatial_correlation[upper_triangle]
    )

    sigma_phylogenetic = numpyro.sample(
        "sigma_phylogenetic",
        dist.HalfNormal(settings["amplitude_scale"])
        .expand((n_traits,)).to_event(1),
    )
    phylogenetic_correlation_cholesky = numpyro.sample(
        "phylogenetic_trait_correlation_cholesky",
        dist.LKJCholesky(
            dimension=n_traits,
            concentration=settings["trait_correlation_concentration"],
        ),
    )
    phylogenetic_trait_cholesky = (
        sigma_phylogenetic[:, None] * phylogenetic_correlation_cholesky
    )
    numpyro.deterministic("phylogenetic_trait_cholesky", phylogenetic_trait_cholesky)
    phylogenetic_correlation = (
        phylogenetic_correlation_cholesky @ phylogenetic_correlation_cholesky.T
    )
    numpyro.deterministic(
        "phylogenetic_trait_correlations",
        phylogenetic_correlation[upper_triangle],
    )

    z_spatial = numpyro.sample(
        "z_spatial",
        dist.Normal(0, 1)
        .expand((spatial_basis.shape[1], n_traits)).to_event(2),
    )
    z_phylogenetic = numpyro.sample(
        "z_phylogenetic",
        dist.Normal(0, 1)
        .expand((phylogenetic_factor.shape[1], n_traits)).to_event(2),
    )

    # Los coeficientes latentes normales pasan por una base esférica ponderada.
    # El corte máximo y el suavizado determinan qué escalas geográficas aportan.
    harmonic_degree = jnp.asarray(degrees)
    weights = 1 / (1 + smoothing * (harmonic_degree * (harmonic_degree + 1)) ** 2)
    weights *= jax.nn.sigmoid(
        (max_degree + 0.5 - harmonic_degree) / settings["degree_taper_width"]
    )
    scales = jnp.sqrt(weights / jnp.sum(weights))

    # Este producto construye un campo por lengua y luego mezcla los rasgos
    # con la Cholesky espacial; su covarianza es R_espacial ⊗ Sigma_espacial.
    spatial_standard = spatial_basis @ (scales[:, None] * z_spatial)
    spatial = spatial_standard @ spatial_trait_cholesky.T

    # La matriz de diseño filogenética ya incorpora covarianza entre lenguas;
    # la Cholesky mezcla los rasgos: Cov(h) = R_filogenética ⊗ Sigma_filogenética.
    phylogenetic_standard = phylogenetic_factor @ z_phylogenetic
    phylogenetic = phylogenetic_standard @ phylogenetic_trait_cholesky.T
    eta = intercept[None, :] + spatial + phylogenetic

    if labels is not None:
        # log_ndtr evalúa log(Phi(eta)) de forma estable; el mask quita del
        # likelihood las celdas faltantes o reservadas para test.
        log_probability = jnp.where(
            labels == 1, log_ndtr(eta), log_ndtr(-eta)
        )
        numpyro.factor(
            "bernoulli_probit_likelihood",
            jnp.where(observed_mask, log_probability, 0).sum(),
        )
