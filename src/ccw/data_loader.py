"""
Lightweight observational data loader for cosmological parameter constraints.

Provides small, self-contained datasets for distance modulus μ(z) from SNe Ia observations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import csv


@dataclass
class DistanceModulusPoint:
    """Single distance modulus measurement."""
    z: float
    mu: float  # distance modulus (mag)
    sigma_mu: float  # uncertainty (mag)
    source: str = "pantheon+"


def load_pantheon_plus_subset(filepath: Optional[Path] = None, max_points: int = 50) -> List[DistanceModulusPoint]:
    """
    Load a small subset of Pantheon+ SNe Ia distance modulus data.

    Parameters
    ----------
    filepath : Path, optional
        Path to CSV file with columns: z, mu, sigma_mu
        If None, uses built-in minimal dataset.
    max_points : int
        Maximum number of points to load (for computational efficiency).

    Returns
    -------
    List[DistanceModulusPoint]
        List of distance modulus measurements.

    Notes
    -----
    Built-in dataset is a small representative sample spanning z ~ 0.01 to 2.0.
    For production use, load from actual Pantheon+ catalog.
    """
    if filepath is not None:
        return _load_from_csv(filepath, max_points)
    else:
        return _get_builtin_dataset(max_points)


def _load_from_csv(filepath: Path, max_points: int) -> List[DistanceModulusPoint]:
    """Load data from CSV file."""
    points = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_points:
                break
            points.append(
                DistanceModulusPoint(
                    z=float(row["z"]),
                    mu=float(row["mu"]),
                    sigma_mu=float(row["sigma_mu"]),
                    source=row.get("source", "pantheon+"),
                )
            )
    return points


def _get_builtin_dataset(max_points: int) -> List[DistanceModulusPoint]:
    """
    Return built-in minimal dataset (representative sample).

    Data source: Approximate Pantheon+ values (Brout et al. 2022).
    This is a *toy* dataset for demonstration; use real data for science.
    """
    # Representative sample spanning z ~ 0.01 to 2.0
    # Format: (z, mu, sigma_mu)
    builtin_data = [
        (0.010, 33.2, 0.15),
        (0.023, 35.1, 0.12),
        (0.050, 37.0, 0.10),
        (0.100, 38.8, 0.08),
        (0.150, 39.9, 0.08),
        (0.200, 40.7, 0.08),
        (0.250, 41.3, 0.08),
        (0.300, 41.8, 0.08),
        (0.350, 42.2, 0.09),
        (0.400, 42.6, 0.09),
        (0.450, 43.0, 0.09),
        (0.500, 43.3, 0.09),
        (0.550, 43.6, 0.10),
        (0.600, 43.9, 0.10),
        (0.650, 44.2, 0.10),
        (0.700, 44.4, 0.11),
        (0.750, 44.7, 0.11),
        (0.800, 44.9, 0.11),
        (0.850, 45.1, 0.12),
        (0.900, 45.3, 0.12),
        (0.950, 45.5, 0.13),
        (1.000, 45.7, 0.13),
        (1.100, 46.0, 0.14),
        (1.200, 46.3, 0.15),
        (1.300, 46.6, 0.16),
        (1.400, 46.8, 0.17),
        (1.500, 47.0, 0.18),
        (1.600, 47.2, 0.19),
        (1.700, 47.4, 0.20),
        (1.800, 47.6, 0.21),
        (1.900, 47.8, 0.22),
        (2.000, 48.0, 0.23),
    ]

    points = []
    for i, (z, mu, sigma_mu) in enumerate(builtin_data):
        if i >= max_points:
            break
        points.append(
            DistanceModulusPoint(
                z=z,
                mu=mu,
                sigma_mu=sigma_mu,
                source="builtin_pantheon_plus_approx",
            )
        )

    return points
