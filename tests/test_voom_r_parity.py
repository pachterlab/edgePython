import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from edgepython.voom_lmfit import voom_basic


def _r_voom_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    cmd = [
        "Rscript",
        "-e",
        "cat(requireNamespace('limma', quietly=TRUE), '\\n')",
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out.stdout.strip().lower() == "true"


def _run_r_voom(counts: np.ndarray, design: np.ndarray, tmpdir: Path):
    counts_path = tmpdir / "counts.csv"
    design_path = tmpdir / "design.csv"
    out_path = tmpdir / "r_voom.csv"

    np.savetxt(counts_path, counts, delimiter=",")
    np.savetxt(design_path, design, delimiter=",")

    r_code = f"""
    suppressPackageStartupMessages(library(limma))
    counts <- as.matrix(read.csv('{counts_path}', header=FALSE, check.names=FALSE))
    design <- as.matrix(read.csv('{design_path}', header=FALSE, check.names=FALSE))
    v <- voom(counts, design=design, plot=FALSE)
    gene_weight <- apply(v$weights, 1, median)
    amean <- rowMeans(v$E)
    out <- data.frame(gene_weight=gene_weight, Amean=amean)
    write.csv(out, file='{out_path}', row.names=FALSE)
    """
    subprocess.run(["Rscript", "-e", r_code], check=True)

    arr = np.genfromtxt(out_path, delimiter=",", names=True)
    return arr["gene_weight"], arr["Amean"]


@pytest.mark.skipif(not _r_voom_available(), reason="Rscript+limma required")
def test_voom_basic_parity_with_r_voom(tmp_path):
    rng = np.random.default_rng(123)
    n_genes = 400
    n_samples = 8

    group = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    design = np.column_stack([np.ones(n_samples), group])

    base_mu = np.exp(rng.uniform(np.log(1.5), np.log(1200), size=n_genes))
    fold_change = rng.lognormal(mean=0.0, sigma=0.22, size=n_genes)

    mu = np.repeat(base_mu[:, None], n_samples, axis=1)
    mu[:, group == 1] *= fold_change[:, None]

    counts = rng.poisson(mu).astype(float)

    py = voom_basic(counts, design=design)
    py_gene_weight = np.median(py["weights"], axis=1)
    py_amean = py["E"].mean(axis=1)

    r_gene_weight, r_amean = _run_r_voom(counts, design, tmp_path)

    # Quantify agreement: strong rank concordance in weights and close mean expression.
    weight_spearman = np.corrcoef(
        np.argsort(np.argsort(np.log(py_gene_weight))),
        np.argsort(np.argsort(np.log(r_gene_weight))),
    )[0, 1]
    amean_rmse = np.sqrt(np.mean((py_amean - r_amean) ** 2))

    assert weight_spearman > 0.999
    assert amean_rmse < 1e-6
