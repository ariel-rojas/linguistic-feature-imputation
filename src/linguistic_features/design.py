"""Diseños espaciales y condicionamiento de efectos filogenéticos."""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import expit, sph_harm_y


def spherical_basis(latitude, longitude, maximum_degree):
    """Base de armónicos reales en la esfera, hasta el grado configurado."""
    # scipy usa colatitud theta; las coordenadas de las fuentes están en latitud.
    theta = np.pi / 2 - np.radians(np.asarray(latitude, dtype=float))
    phi = np.radians(np.asarray(longitude, dtype=float))
    columns, degrees = [], []
    for degree in range(maximum_degree + 1):
        columns.append(sph_harm_y(degree, 0, theta, phi).real)
        degrees.append(degree)
        for order in range(1, degree + 1):
            harmonic = sph_harm_y(degree, order, theta, phi)
            columns.extend((np.sqrt(2) * harmonic.real, np.sqrt(2) * harmonic.imag))
            degrees.extend((degree, degree))
    return np.column_stack(columns) * np.sqrt(4 * np.pi), np.asarray(degrees)


def harmonic_scales(degrees, maximum_degree, smoothing, taper_width):
    """Pondera frecuencias; NUTS aprende el corte efectivo máximo L."""
    degree = np.asarray(degrees, dtype=float)
    # El primer factor suaviza frecuencias altas; el sigmoid permite inferir un
    # corte L continuo sin cambiar el tamaño fijo de la base computacional.
    weights = 1 / (1 + smoothing * (degree * (degree + 1)) ** 2)
    weights *= expit((maximum_degree + 0.5 - degree) / taper_width)
    return np.sqrt(weights / weights.sum())


def covariance_factor(matrix):
    """Raíz simétrica que admite covarianzas semidefinidas positivas."""
    if matrix.shape == (0, 0):
        return np.empty((0, 0))
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2)
    if eigenvalues.min(initial=0) < -1e-8:
        raise ValueError("La covarianza condicional no es semidefinida positiva")
    return eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))


def phylogenetic_design(covariance, training_tips, target_tips):
    """Factoriza efectos de entrenamiento y calcula la ley condicional objetivo.

    Para z ~ N(0,I), el efecto en entrenamiento es L z. El efecto objetivo
    condicionado a entrenamiento tiene media A z y una covarianza residual
    factorizada aparte. Repeticiones de un glottocódigo comparten la misma hoja.
    """
    training_tips = np.asarray(training_tips, dtype=int)
    target_tips = np.asarray(target_tips, dtype=int)
    unique, inverse = np.unique(training_tips, return_inverse=True)
    train_cov = covariance[np.ix_(unique, unique)]
    chol = np.linalg.cholesky(train_cov)
    # Varios registros pueden ser alias del mismo taxón: comparten la hoja MCC.
    factor = chol[inverse]
    cross = covariance[np.ix_(target_tips, unique)]
    # Para z ~ N(0,I), la media condicional en objetivos es A z y el residuo
    # completa la covarianza objetivo sin volver a muestrear las hojas de train.
    conditional_mean = solve_triangular(chol, cross.T, lower=True).T
    target_cov = covariance[np.ix_(target_tips, target_tips)]
    residual = target_cov - conditional_mean @ conditional_mean.T
    return factor, conditional_mean, covariance_factor(residual), unique
