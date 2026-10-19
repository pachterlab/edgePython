"""Writes the small simulated quantifier outputs in quant/.

Run from tests/data:  python make_quant_files.py
The files are read by edger5_parity.R (edgeR's catch* functions) and by
the tests. 40 transcripts in 14 genes, 3 samples, with Gencode names
(salmon, oarfish, rsem, lr_kallisto) or plain versioned IDs with a
tx2gene table (kallisto); one gene with no reads; short transcripts whose
effective length equals their length.
"""

import gzip
import json
import os
import shutil

import h5py
import numpy as np
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

OUT = "quant"
rng = np.random.default_rng(20261018)
NS = 3
TYPES = ["protein_coding", "retained_intron", "nonsense_mediated_decay", "lncRNA",
         "protein_coding_CDS_not_defined", "processed_pseudogene", "TEC"]

# Genes and transcripts
ntx_per_gene = [3, 1, 2, 2, 4, 1, 3, 5, 2, 1, 4, 3, 6, 3]
gene_of = np.repeat(np.arange(len(ntx_per_gene)), ntx_per_gene)
ntx = len(gene_of)
length = rng.integers(300, 4000, ntx)
length[[5, 17, 30]] = [120, 150, 90]
tx_type = rng.choice(TYPES, ntx, p=[0.4, 0.15, 0.1, 0.15, 0.08, 0.07, 0.05])
gencode = []
for t in range(ntx):
    g = gene_of[t]
    hav_g = f"OTTHUMG{g:08d}.{1 + g % 2}" if g % 3 else "-"
    hav_t = f"OTTHUMT{t:08d}.1" if g % 3 else "-"
    gencode.append(f"ENST{t + 1:011d}.{1 + t % 3}|ENSG{g + 1:011d}.{1 + g % 4}|{hav_g}|{hav_t}|"
                   f"GENE{g + 1}-{201 + t}|GENE{g + 1}|{length[t]}|{tx_type[t]}|")
plain_ids = [s.split("|")[0] for s in gencode]
tx2gene = pd.DataFrame({"tx": plain_ids, "gene": [s.split("|")[1] for s in gencode]})

# Expression: gene 4 (index 3) has no reads; some transcripts are zero in one sample
mu = rng.gamma(0.8, 300, ntx)
mu[gene_of == 3] = 0
mu[[8, 21]] = 0
counts = np.zeros((ntx, NS))
for j in range(NS):
    counts[:, j] = rng.gamma(20, mu * (0.8 + 0.2 * j) / 20)
counts[9, 1] = 0
counts = np.round(counts, 3)
frag = [180.0, 210.0, 195.0]
eff_len = np.column_stack([np.maximum(length - f + 1, 1.0) + rng.normal(0, 3, ntx) for f in frag])
eff_len = np.round(eff_len, 4)
eff_len[[5, 17, 30], :] = length[[5, 17, 30], None]
eff_len[12, 2] = length[12]
tpm = counts / eff_len
tpm = 1e6 * tpm / tpm.sum(axis=0)


def resamples(j, n, over):
    """Bootstrap-like draws around counts[:, j], overdispersed by `over`."""
    c = counts[:, j]
    shape = np.where(c > 0, c / over, 1.0)
    draws = rng.gamma(shape[:, None], over[:, None], (ntx, n))
    return np.where(c[:, None] > 0, draws, 0.0)


over = 1 + rng.gamma(2, 1.5, ntx)
if os.path.exists(OUT):
    shutil.rmtree(OUT)

# Salmon: 6 Gibbs samples
for j in range(NS):
    d = os.path.join(OUT, "salmon", f"s{j + 1}")
    os.makedirs(os.path.join(d, "aux_info", "bootstrap"))
    pd.DataFrame({"Name": gencode, "Length": length, "EffectiveLength": eff_len[:, j],
                  "TPM": tpm[:, j], "NumReads": counts[:, j]}).to_csv(
        os.path.join(d, "quant.sf"), sep="\t", index=False, float_format="%.17g")
    with open(os.path.join(d, "aux_info", "meta_info.json"), "w") as f:
        json.dump({"num_targets": ntx, "num_bootstraps": 6, "samp_type": "gibbs"}, f)
    boot = resamples(j, 6, over)
    with open(os.path.join(d, "aux_info", "bootstrap", "bootstraps.gz"), "wb") as raw, \
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        f.write(np.asfortranarray(boot).astype("<f8").tobytes(order="F"))


def write_kallisto_h5(path, ids, counts_j, eff_j, boot):
    with h5py.File(path, "w", track_order=True) as f:
        aux = f.create_group("aux")
        aux.create_dataset("ids", data=np.array(ids, dtype=object), dtype=h5py.string_dtype())
        aux.create_dataset("lengths", data=length.astype(np.int32))
        aux.create_dataset("eff_lengths", data=eff_j)
        aux.create_dataset("num_bootstrap", data=np.array([boot.shape[1]], dtype=np.int32))
        f.create_dataset("est_counts", data=counts_j)
        bs = f.create_group("bootstrap")
        for b in range(boot.shape[1]):
            bs.create_dataset(f"bs{b}", data=boot[:, b])


def write_kallisto_tsv(path, ids, counts_j, eff_j):
    x = counts_j / eff_j
    pd.DataFrame({"target_id": ids, "length": length, "eff_length": eff_j, "est_counts": counts_j,
                  "tpm": 1e6 * x / x.sum()}).to_csv(path, sep="\t", index=False, float_format="%.17g")


# kallisto: plain IDs, 5 bootstraps, abundance.h5; the same in plaintext
kboot = [resamples(j, 5, over) for j in range(NS)]
for j in range(NS):
    d = os.path.join(OUT, "kallisto", f"k{j + 1}")
    os.makedirs(d)
    write_kallisto_h5(os.path.join(d, "abundance.h5"), plain_ids, counts[:, j], eff_len[:, j], kboot[j])
    d = os.path.join(OUT, "kallisto_tsv", f"k{j + 1}")
    os.makedirs(d)
    write_kallisto_tsv(os.path.join(d, "abundance.tsv"), plain_ids, counts[:, j], eff_len[:, j])
    for b in range(5):
        write_kallisto_tsv(os.path.join(d, f"bs_abundance_{b}.tsv"), plain_ids, kboot[j][:, b], eff_len[:, j])
tx2gene.to_csv(os.path.join(OUT, "tx2gene.tsv"), sep="\t", index=False)

# oarfish: Gencode names, 8 bootstraps in parquet
os.makedirs(os.path.join(OUT, "oarfish"))
for j in range(NS):
    prefix = os.path.join(OUT, "oarfish", f"o{j + 1}")
    pd.DataFrame({"tname": gencode, "len": length, "num_reads": counts[:, j]}).to_csv(
        prefix + ".quant", sep="\t", index=False, float_format="%.17g")
    with open(prefix + ".meta_info.json", "w") as f:
        json.dump({"num_bootstraps": 8}, f)
    boot = resamples(j, 8, over)
    pd.DataFrame(boot, columns=[f"infrep{b}" for b in range(8)]).to_parquet(prefix + ".infreps.pq", index=False)

# RSEM: isoform and gene results with Gibbs summaries (50 Gibbs draws)
os.makedirs(os.path.join(OUT, "rsem"))
gene_ids = [s.split("|")[1] for s in gencode]
for j in range(NS):
    draws = resamples(j, 50, over)
    pm, psd = draws.mean(axis=1), draws.std(axis=1, ddof=1)
    rsem_eff = eff_len[:, j].copy()
    rsem_eff[[5, 30]] = 0.0
    iso = pd.DataFrame({
        "transcript_id": gencode, "gene_id": gene_ids, "length": length, "effective_length": rsem_eff,
        "expected_count": counts[:, j], "TPM": tpm[:, j], "FPKM": tpm[:, j] / 2, "IsoPct": 50.0,
        "posterior_mean_count": pm, "posterior_standard_deviation_of_count": psd,
        "pme_TPM": tpm[:, j], "pme_FPKM": tpm[:, j] / 2, "IsoPct_from_pme_TPM": 50.0})
    iso.to_csv(os.path.join(OUT, "rsem", f"r{j + 1}.isoforms.results"), sep="\t", index=False,
               float_format="%.17g")
    gdraws = pd.DataFrame(draws).groupby(gene_of).sum().to_numpy()
    w = tpm[:, j]
    geff = pd.Series(w * rsem_eff).groupby(gene_of).sum() / pd.Series(w).groupby(gene_of).sum()
    genes = pd.DataFrame({
        "gene_id": pd.unique(np.array(gene_ids)),
        "transcript_id(s)": pd.Series(plain_ids).groupby(gene_of).agg(",".join).to_numpy(),
        "length": pd.Series(length).groupby(gene_of).mean().to_numpy(),
        "effective_length": geff.fillna(0).to_numpy(),
        "expected_count": pd.Series(counts[:, j]).groupby(gene_of).sum().to_numpy(),
        "TPM": pd.Series(w).groupby(gene_of).sum().to_numpy(), "FPKM": 0.0,
        "posterior_mean_count": gdraws.mean(axis=1),
        "posterior_standard_deviation_of_count": gdraws.std(axis=1, ddof=1),
        "pme_TPM": 0.0, "pme_FPKM": 0.0})
    genes.to_csv(os.path.join(OUT, "rsem", f"r{j + 1}.genes.results"), sep="\t", index=False,
                 float_format="%.17g")

# lr-kallisto (kallisto quant-tcc): per-sample directories with h5, flat
# plaintext files, and the matrix output, from the same values
lr = os.path.join(OUT, "lr_kallisto")
for j in range(NS):
    d = os.path.join(lr, "dirs", f"abundance_{j + 1}")
    os.makedirs(d)
    write_kallisto_h5(os.path.join(d, "abundance.h5"), gencode, counts[:, j], eff_len[:, j], kboot[j])
    flat = os.path.join(lr, "flat")
    os.makedirs(flat, exist_ok=True)
    write_kallisto_tsv(os.path.join(flat, f"abundance_{j + 1}.tsv"), gencode, counts[:, j], eff_len[:, j])
    for b in range(5):
        write_kallisto_tsv(os.path.join(flat, f"bs_abundance_{j + 1}_{b}.tsv"), gencode, kboot[j][:, b],
                           eff_len[:, j])
m = os.path.join(lr, "matrix")
os.makedirs(m)
mmwrite(os.path.join(m, "matrix.abundance.mtx"), csr_matrix(counts.T), precision=17)
mmwrite(os.path.join(m, "matrix.abundance.tpm.mtx"), csr_matrix(tpm.T), precision=17)
mmwrite(os.path.join(m, "matrix.efflens.mtx"), csr_matrix(np.where(counts > 0, eff_len, 0).T), precision=17)
with open(os.path.join(m, "transcripts.txt"), "w") as f:
    f.write("\n".join(gencode) + "\n")
