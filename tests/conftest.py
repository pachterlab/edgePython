# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Shared fixtures for edgePython tests."""

import os
import numpy as np
import pandas as pd
import pytest


CSV_DIR = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def csv_dir():
    """Path to R/Python CSV comparison data."""
    return CSV_DIR


@pytest.fixture
def rng():
    """Seeded random number generator for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def small_counts(rng):
    """Small count matrix: 100 genes x 6 samples, Poisson(10)."""
    counts = rng.poisson(10, (100, 6)).astype(np.float64)
    # Make first 10 genes differentially expressed
    counts[:5, 3:6] *= 3
    counts[5:10, 3:6] = 0
    return counts


@pytest.fixture
def group6():
    """Group vector for 6 samples (3+3)."""
    return np.array([0, 0, 0, 1, 1, 1])


@pytest.fixture
def design6():
    """Design matrix for 6 samples with intercept + group."""
    return np.column_stack([np.ones(6), np.array([0, 0, 0, 1, 1, 1])])


@pytest.fixture
def dgelist(small_counts, group6):
    """DGEList from small_counts with group factor."""
    import edgepython as ep
    return ep.make_dgelist(small_counts, group=group6)


@pytest.fixture
def dgelist_disp(dgelist, design6):
    """DGEList with estimated dispersions."""
    import edgepython as ep
    return ep.estimate_disp(dgelist, design6)


@pytest.fixture
def gene_sets():
    """Dict of gene set indices for 100-gene data."""
    return {
        'Set1': list(range(10)),
        'Set2': list(range(50, 70)),
        'Set3': list(range(20, 30)),
    }
