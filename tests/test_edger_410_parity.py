"""Optional parity checks against a local edgeR 4.10.1 source tree."""

import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import edgepython as ep
from edgepython.dgelist import make_dgelist


EDGER_SRC = os.environ.get("EDGER_RELEASE_SRC", "/private/tmp/edgeR_RELEASE_3_23")


def _edger_source_available():
    return (
        shutil.which("Rscript") is not None
        and os.path.exists(os.path.join(EDGER_SRC, "R", "sampleWeights.R"))
    )


pytestmark = pytest.mark.skipif(
    not _edger_source_available(),
    reason="Rscript and local edgeR release source are required",
)


def _run_r(code):
    subprocess.run(["Rscript", "--vanilla", "-e", code], check=True)


def _r_package_available(package):
    out = subprocess.run(
        ["Rscript", "--vanilla", "-e",
         f"cat(requireNamespace('{package}', quietly=TRUE), '\\n')"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip().lower() == "true"


def test_sample_weights_matches_edgeR_410(tmp_path):
    unit_deviance = np.array([
        [2.0, 3.5, 1.2],
        [5.0, 2.0, 4.0],
        [1.5, 1.1, 2.4],
        [8.0, 3.0, 2.5],
    ])
    unit_df = np.array([
        [1.0, 1.0, 0.5],
        [1.0, 0.5, 1.0],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    s2 = np.array([1.4, 2.2, 0.8, 3.5])

    dev_path = tmp_path / "dev.csv"
    df_path = tmp_path / "df.csv"
    s2_path = tmp_path / "s2.csv"
    out_path = tmp_path / "weights.csv"
    np.savetxt(dev_path, unit_deviance, delimiter=",")
    np.savetxt(df_path, unit_df, delimiter=",")
    np.savetxt(s2_path, s2, delimiter=",")

    r_code = f"""
    source('{EDGER_SRC}/R/sampleWeights.R')
    dev <- as.matrix(read.csv('{dev_path}', header=FALSE))
    df <- as.matrix(read.csv('{df_path}', header=FALSE))
    s2 <- scan('{s2_path}', sep=',', quiet=TRUE)
    out <- rbind(
        sampleWeights(dev, df, iter=4),
        sampleWeights(dev, df, s2=s2)
    )
    write.csv(out, '{out_path}', row.names=FALSE)
    """
    _run_r(r_code)
    r = np.asarray(pd.read_csv(out_path), dtype=float)

    py = np.vstack([
        ep.sample_weights(unit_deviance, unit_df, iter=4),
        ep.sample_weights(unit_deviance, unit_df, s2=s2),
    ])
    assert np.allclose(py, r, rtol=1e-12, atol=1e-12)


def test_dgelist_from_tximport_matches_edgeR_410(tmp_path):
    counts = np.array([[10.0, 20.0], [0.0, 0.0], [5.0, 15.0]])
    length = np.array([[100.0, 120.0], [80.0, 80.0], [200.0, 100.0]])
    counts_path = tmp_path / "counts.csv"
    length_path = tmp_path / "length.csv"
    out_counts = tmp_path / "r_counts.csv"
    out_genes = tmp_path / "r_genes.csv"
    out_offset = tmp_path / "r_offset.csv"
    np.savetxt(counts_path, counts, delimiter=",")
    np.savetxt(length_path, length, delimiter=",")

    r_code = f"""
    suppressPackageStartupMessages(library(edgeR))
    source('{EDGER_SRC}/R/DGEListFromTximport.R')
    txi <- list(
        counts=as.matrix(read.csv('{counts_path}', header=FALSE)),
        length=as.matrix(read.csv('{length_path}', header=FALSE)),
        countsFromAbundance='no'
    )
    rownames(txi$counts) <- rownames(txi$length) <- c('tx1','tx2','tx3')
    colnames(txi$counts) <- colnames(txi$length) <- c('s1','s2')
    y <- DGEListFromTximport(txi, group=c('A','B'), remove.zeros=TRUE)
    write.csv(y$counts, '{out_counts}')
    write.csv(y$genes, '{out_genes}')
    write.csv(y$offset.prior, '{out_offset}')
    """
    _run_r(r_code)

    py = ep.dgelist_from_tximport({
        "counts": pd.DataFrame(counts, index=["tx1", "tx2", "tx3"], columns=["s1", "s2"]),
        "length": pd.DataFrame(length, index=["tx1", "tx2", "tx3"], columns=["s1", "s2"]),
        "countsFromAbundance": "no",
    }, group=["A", "B"], remove_zeros=True)

    r_counts = pd.read_csv(out_counts, index_col=0)
    r_genes = pd.read_csv(out_genes, index_col=0)
    r_offset = pd.read_csv(out_offset, index_col=0)

    assert np.allclose(py["counts"], r_counts.values)
    assert np.allclose(py["genes"][["AveLength", "Max2MinLength"]].values, r_genes.values)
    r_offset = r_offset.loc[r_counts.index]
    assert np.allclose(py["offset.prior"], r_offset.values)
    assert py["tximport.counts"] == "raw"


def test_catch_rsem_matches_edgeR_410(tmp_path):
    def write_sample(name, expected, mean, sd):
        quant = pd.DataFrame({
            "transcript_id": ["tx1", "tx2"],
            "gene_id": ["g1", "g2"],
            "length": [100.0, 200.0],
            "effective_length": [90.0, 180.0],
            "expected_count": expected,
            "TPM": [1.0, 2.0],
            "FPKM": [3.0, 4.0],
            "IsoPct": [50.0, 50.0],
            "posterior_mean_count": mean,
            "posterior_standard_deviation_of_count": sd,
        })
        quant.to_csv(tmp_path / name, sep="\t", index=False)

    write_sample("s1.isoforms.results", [10.0, 20.0], [10.0, 20.0], [2.0, 4.0])
    write_sample("s2.isoforms.results", [15.0, 25.0], [15.0, 25.0], [3.0, 5.0])
    out_counts = tmp_path / "r_counts.csv"
    out_ann = tmp_path / "r_ann.csv"
    out_meta = tmp_path / "r_meta.csv"

    r_code = f"""
    suppressPackageStartupMessages(library(edgeR))
    source('{EDGER_SRC}/R/catchRSEM.R')
    y <- catchRSEM(path='{tmp_path}', ngibbs=5, verbose=FALSE)
    write.csv(y$counts, '{out_counts}')
    write.csv(y$annotation, '{out_ann}')
    write.csv(data.frame(overdispersion.prior=y$overdispersion.prior), '{out_meta}', row.names=FALSE)
    """
    _run_r(r_code)

    py = ep.catch_rsem(path=str(tmp_path), ngibbs=5, verbose=False)
    r_counts = pd.read_csv(out_counts, index_col=0)
    r_ann = pd.read_csv(out_ann, index_col=0)
    r_meta = pd.read_csv(out_meta)

    assert np.allclose(py["counts"], r_counts.values)
    assert np.allclose(
        py["annotation"][["Length", "AveLength", "Max2MinLength", "Overdispersion"]].values,
        r_ann[["Length", "AveLength", "Max2MinLength", "Overdispersion"]].values,
    )
    assert np.allclose(py["overdispersion.prior"], r_meta["overdispersion.prior"].iloc[0])


def test_catch_oarfish_matches_edgeR_410_without_bootstraps(tmp_path):
    if not _r_package_available("nanoparquet"):
        pytest.skip("R package nanoparquet is required by edgeR::catchOarfish")

    for sample, counts in {"s1": [10.0, 20.0], "s2": [15.0, 25.0]}.items():
        quant = pd.DataFrame({
            "tname": ["tx1", "tx2"],
            "len": [100.0, 200.0],
            "num_reads": counts,
        })
        quant.to_csv(tmp_path / f"{sample}.quant", sep="\t", index=False)
        (tmp_path / f"{sample}.meta_info.json").write_text('{"num_bootstraps":0}\n')

    out_counts = tmp_path / "r_counts.csv"
    out_ann = tmp_path / "r_ann.csv"
    r_code = f"""
    suppressPackageStartupMessages(library(edgeR))
    source('{EDGER_SRC}/R/catchOarfish.R')
    y <- catchOarfish(path='{tmp_path}', verbose=FALSE)
    write.csv(y$counts, '{out_counts}')
    write.csv(y$annotation, '{out_ann}')
    """
    _run_r(r_code)

    py = ep.catch_oarfish(path=str(tmp_path), verbose=False)
    r_counts = pd.read_csv(out_counts, index_col=0)
    r_ann = pd.read_csv(out_ann, index_col=0)

    assert np.allclose(py["counts"], r_counts.values)
    assert np.allclose(
        py["annotation"][["Length", "Overdispersion"]].values,
        r_ann[["Length", "Overdispersion"]].values,
        equal_nan=True,
    )


def test_normalize_between_arrays_dgelist_matches_edgeR_410_quantile(tmp_path):
    counts = np.array([
        [5.0, 20.0, 8.0],
        [30.0, 10.0, 4.0],
        [12.0, 25.0, 40.0],
        [80.0, 45.0, 20.0],
    ])
    counts_path = tmp_path / "counts.csv"
    out_offset = tmp_path / "r_offset.csv"
    np.savetxt(counts_path, counts, delimiter=",")

    r_code = f"""
    suppressPackageStartupMessages(library(edgeR))
    suppressPackageStartupMessages(library(limma))
    source('{EDGER_SRC}/R/normalizeBetweenArraysDGEList.R')
    counts <- as.matrix(read.csv('{counts_path}', header=FALSE))
    y <- DGEList(counts)
    y <- normalizeBetweenArrays.DGEList(y, method='quantile')
    write.csv(y$offset, '{out_offset}', row.names=FALSE)
    """
    _run_r(r_code)

    py = ep.normalize_between_arrays_dgelist(make_dgelist(counts), method="quantile")
    r_offset = np.asarray(pd.read_csv(out_offset), dtype=float)
    assert np.allclose(py["offset"], r_offset, rtol=1e-10, atol=1e-10)


def test_voom_lmfit_offset_prior_matches_edgeR_410_logcpm(tmp_path):
    counts = np.array([
        [10.0, 20.0, 30.0, 40.0],
        [5.0, 15.0, 25.0, 35.0],
        [80.0, 60.0, 40.0, 20.0],
        [100.0, 120.0, 140.0, 160.0],
    ])
    design = np.column_stack([np.ones(4), [0.0, 0.0, 1.0, 1.0]])
    offset = np.array([
        [0.0, 0.1, 0.2, 0.3],
        [0.4, 0.1, -0.1, -0.2],
        [-0.2, 0.0, 0.2, 0.4],
        [0.3, 0.2, 0.1, 0.0],
    ])
    counts_path = tmp_path / "counts.csv"
    design_path = tmp_path / "design.csv"
    offset_path = tmp_path / "offset.csv"
    out_e = tmp_path / "r_e.csv"
    out_offset_prior = tmp_path / "r_offset_prior.csv"
    out_lib_matrix = tmp_path / "r_lib_matrix.csv"
    np.savetxt(counts_path, counts, delimiter=",")
    np.savetxt(design_path, design, delimiter=",")
    np.savetxt(offset_path, offset, delimiter=",")

    r_code = f"""
    suppressPackageStartupMessages(library(edgeR))
    suppressPackageStartupMessages(library(limma))
    source('{EDGER_SRC}/R/voomLmFit.R')
    counts <- as.matrix(read.csv('{counts_path}', header=FALSE))
    design <- as.matrix(read.csv('{design_path}', header=FALSE))
    offset <- as.matrix(read.csv('{offset_path}', header=FALSE))
    offset.prior <- offset - rowMeans(offset)
    lib.size <- colSums(counts)
    lib.size.matrix <- exp(log(matrix(lib.size,nrow(counts),ncol(counts),byrow=TRUE))+offset.prior)
    fit <- voomLmFit(
        counts, design=design, offset=offset, normalize.method='none',
        adaptive.span=FALSE, keep.EList=TRUE
    )
    write.csv(fit$EList$E, '{out_e}', row.names=FALSE)
    write.csv(offset.prior, '{out_offset_prior}', row.names=FALSE)
    write.csv(lib.size.matrix, '{out_lib_matrix}', row.names=FALSE)
    """
    _run_r(r_code)

    py = ep.voom_lmfit(
        counts, design=design, offset=offset, normalize_method="none",
        adaptive_span=False, keep_elist=True)
    r_e = np.asarray(pd.read_csv(out_e), dtype=float)
    r_offset_prior = np.asarray(pd.read_csv(out_offset_prior), dtype=float)
    r_lib_matrix = np.asarray(pd.read_csv(out_lib_matrix), dtype=float)

    assert np.allclose(py["E"], r_e, rtol=1e-12, atol=1e-12)
    assert np.allclose(py["offset_prior"], r_offset_prior, rtol=1e-12, atol=1e-12)
    assert np.allclose(py["lib_size_matrix"], r_lib_matrix, rtol=1e-12, atol=1e-12)
