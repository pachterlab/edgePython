# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Readers for transcript quantifications with resampling uncertainty.

Ports of edgeR 5.0's catchSalmon, catchSalmonGene, catchKallisto,
catchKallistoGene, catchOarfish, catchRSEM, splitGencodeTxNames and
collapseGencodeTx2Gene. catch_oarfish_gene, catch_rsem_gene,
catch_lr_kallisto and catch_lr_kallisto_gene are edgePython additions
built the same way.

Each reader estimates an RTA overdispersion for every transcript (or gene)
from the bootstrap or Gibbs samples, as in Baldoni et al. (2024), and by
default divides the counts by it.
"""

import glob
import gzip
import json
import os
import re
import warnings

import numpy as np
import pandas as pd


# =====================================================================
# Gencode transcript names
# =====================================================================

_GENCODE_COLUMNS = ["EnsT", "EnsG", "HavG", "HavT", "TxName", "GeneName", "AnnLength", "Type"]


def _is_gencode(name):
    """Gencode transcript names have at least 8 '|' separators."""
    return str(name).count("|") >= 8


_VERSION = re.compile(r"\.[0-9]+")


def _remove_version(x):
    """Remove the first '.<digits>' from each ID, as sub('\\.[0-9]+', '', x)."""
    sub = _VERSION.sub
    return np.array([sub("", str(s), 1) for s in x], dtype=object)


def split_gencode_tx_names(x, remove_version=True):
    """Split Gencode transcript names into their eight fields.

    Port of edgeR's splitGencodeTxNames(). Gencode transcript names hold
    '|'-separated annotation.

    Parameters
    ----------
    x : array-like of str
        Gencode transcript names.
    remove_version : bool
        Remove version numbers from the Ensembl transcript and gene IDs.

    Returns
    -------
    DataFrame with columns EnsT, EnsG, HavG, HavT, TxName, GeneName,
    AnnLength and Type.
    """
    # A trailing '|' ends the name, as strsplit() drops a final empty field
    x = [s[:-1] if s.endswith("|") else s for s in map(str, x)]
    nfield = np.array([s.count("|") for s in x]) + 1
    if np.any(nfield != 8):
        raise ValueError(f"'{x[int(np.argmax(nfield != 8))]}' is not a Gencode transcript name with 8 fields")
    fields = np.array("|".join(x).split("|") if x else [], dtype=object).reshape(len(x), 8)
    A = pd.DataFrame(fields, columns=_GENCODE_COLUMNS, dtype=object)
    if remove_version:
        A["EnsT"] = _remove_version(A["EnsT"])
        A["EnsG"] = _remove_version(A["EnsG"])
    return A


def _abbreviate_type(t):
    """Keep the first three characters of each '_'-separated word."""
    return re.sub(r"([^_]{3})[^_]*", r"\1", t) if "_" in t else t


def collapse_gencode_tx2gene(txann):
    """Collapse split Gencode transcript annotation to one row per gene.

    Port of edgeR's collapseGencodeTx2Gene().

    Parameters
    ----------
    txann : DataFrame
        Output of split_gencode_tx_names().

    Returns
    -------
    DataFrame indexed by gene ID (in order of first appearance) with
    GeneName, TxTypes (the gene's abbreviated transcript types, sorted and
    joined by '|') and NPCTx (number of protein-coding transcripts).
    """
    ensg = np.asarray(txann["EnsG"], dtype=object)
    codes, genes = pd.factorize(ensg, sort=False)
    first = np.unique(codes, return_index=True)[1]
    tx_type = [_abbreviate_type(t) for t in txann["Type"]]
    types = [set() for _ in genes]
    for c, t in zip(codes, tx_type):
        types[c].add(t)
    # R's sort() collates case-insensitively
    gene_type = ["|".join(sorted(t, key=lambda s: (s.casefold(), s))) for t in types]
    gene_type = [g.replace("pro_cod_CDS_not_def", "CDS_not_def").replace("pro_cod_LoF", "LoF")
                 for g in gene_type]
    npc = np.bincount(codes, weights=(np.asarray(txann["Type"]) == "protein_coding"),
                      minlength=len(genes))
    return pd.DataFrame({"GeneName": np.asarray(txann["GeneName"], dtype=object)[first],
                         "TxTypes": gene_type, "NPCTx": npc.astype(np.int64)},
                        index=pd.Index(np.asarray(genes, dtype=object)))


# =====================================================================
# Shared estimation
# =====================================================================

class _Overdispersion:
    """Accumulates the resampling statistics of edgeR's catch* functions."""

    def __init__(self, n):
        self.sum = np.zeros(n)
        self.df = np.zeros(n)

    def add_boot(self, boot):
        """Bootstrap or Gibbs draws, rows matching the accumulator."""
        n_boot = boot.shape[1]
        M = boot.mean(axis=1)
        pos = M > 0
        self.sum[pos] += np.sum((boot[pos] - M[pos, None]) ** 2, axis=1) / M[pos]
        self.df[pos] += n_boot - 1

    def add_gibbs_moments(self, M, S, n_gibbs):
        """RSEM's posterior means and standard deviations of n_gibbs draws."""
        pos = M > 0
        self.sum[pos] += (n_gibbs - 1) * S[pos] ** 2 / M[pos]
        self.df[pos] += n_gibbs - 1

    def estimate(self):
        """Moderated overdispersions and their prior (NaN without resamples)."""
        from scipy.stats import f as f_dist
        od = self.sum.copy()
        i = self.df > 0
        if not np.any(i):
            return np.full(len(od), np.nan), np.nan
        od[i] = od[i] / self.df[i]
        df_median = np.median(self.df[i])
        df_prior = 3.0
        prior = np.median(od[i]) / f_dist.ppf(0.5, df_median, df_prior)
        prior = max(prior, 1.0)
        od[i] = (df_prior * prior + self.df[i] * od[i]) / (df_prior + self.df[i])
        od = np.maximum(od, 1.0)
        od[~i] = prior
        return od, prior


def _impute_effective_lengths(length, eff_len):
    """Port of edgeR's .imputeEffectiveLengths().

    An effective length within 0.5 of the transcript length means the
    quantifier could not estimate one; it is reset to the smallest
    estimated effective length of that transcript in the other samples,
    or to 1 if there is none.
    """
    eff_len = eff_len.copy()
    bad = (np.asarray(length, dtype=np.float64)[:, None] - eff_len) < 0.5
    rows = np.flatnonzero(bad.any(axis=1))
    if len(rows):
        ok = np.where(bad[rows], np.inf, eff_len[rows])
        m = ok.min(axis=1)
        m[~np.isfinite(m)] = 1.0
        eff_len[rows] = np.where(bad[rows], m[:, None], eff_len[rows])
    return eff_len


def _length_stats(log_len):
    return np.exp(log_len.mean(axis=1)), np.exp(log_len.max(axis=1) - log_len.min(axis=1))


def _divide(counts, od, n_resampled, divide, verbose):
    """Divide counts by the overdispersions when there were resamples."""
    if not divide:
        return counts, False
    if n_resampled == 0:
        if verbose:
            print("No bootstrap or Gibbs samples, so counts not divided")
        return counts, False
    return counts / od[:, None], True


def _make_output(counts, ann, names, od_prior, resample_type, divided, offset_prior,
                 eff_len, dgelist, tpm=None):
    """DGEList, or plain dict, as edgeR's catch* functions return them."""
    ids = list(ann.index)
    names = list(names)
    eff_df = None if eff_len is None else pd.DataFrame(eff_len, index=ids, columns=names)
    if not dgelist:
        out = {"counts": pd.DataFrame(counts, index=ids, columns=names),
               "effective.length": eff_df, "annotation": ann,
               "overdispersion.prior": od_prior, "resample.type": list(resample_type),
               "divided.counts": divided}
        if tpm is not None:
            out["tpm"] = pd.DataFrame(tpm, index=ids, columns=names)
        if eff_len is None:
            del out["effective.length"]
        return out

    from .dgelist import make_dgelist
    y = make_dgelist(counts, genes=ann.reset_index(drop=True))
    y["genes"].index = pd.Index(ids)
    y["samples"].index = pd.Index(names)
    y["overdispersion.prior"] = od_prior
    y["resample.type"] = list(resample_type)
    y["divided.counts"] = divided
    if offset_prior is not None:
        y["offset.prior"] = offset_prior
    other = {}
    if eff_df is not None:
        other["effective.length"] = eff_df
    if tpm is not None:
        other["tpm"] = pd.DataFrame(tpm, index=ids, columns=names)
    if other:
        y["other"] = other
    return y


def _rowsum(x, codes, n):
    """Sum rows of x within groups (R's rowsum with reorder = FALSE)."""
    from scipy.sparse import csr_matrix
    ind = csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(n, len(codes)))
    return np.asarray(ind @ x)


# =====================================================================
# Transcript-level and gene-level assembly
# =====================================================================
#
# Each format reader yields one dict per sample with keys
#   name, ids, length, counts, eff_len, tpm, boot, gibbs, type
# (ids and length are only needed from the first sample; eff_len, tpm,
# boot and gibbs may be None; gibbs is (M, S, n_gibbs)).

def _assemble_tx(samples, length_name="Length", has_eff_len=True, impute_eff_len=True,
                 log_zero=None, drop_ann_length=True, divide=True, offset_prior=True,
                 dgelist=True, verbose=True):
    """Transcript-level output, as catchSalmon/catchKallisto/catchRSEM/catchOarfish."""
    names, types = [], []
    n_resampled = 0
    for j, s in enumerate(samples):
        if j == 0:
            ids = np.asarray(s["ids"], dtype=object)
            length = s["length"]
            cols_counts, cols_eff = [], []
            od = _Overdispersion(len(ids))
        names.append(s["name"])
        types.append(s["type"])
        cols_counts.append(np.asarray(s["counts"], dtype=np.float64))
        if has_eff_len:
            cols_eff.append(np.asarray(s["eff_len"], dtype=np.float64))
        if s.get("boot") is not None:
            od.add_boot(s["boot"])
            n_resampled += 1
        elif s.get("gibbs") is not None:
            od.add_gibbs_moments(*s["gibbs"])
            n_resampled += 1
    if not names:
        raise ValueError("No samples found")
    counts = np.column_stack(cols_counts)

    ann = pd.DataFrame(index=pd.Index(ids))
    if length is not None:
        ann[length_name] = length
    log_len = eff_len = None
    if has_eff_len:
        eff_len = np.column_stack(cols_eff)
        if impute_eff_len:
            eff_len = _impute_effective_lengths(length, eff_len)
        with np.errstate(divide="ignore"):
            log_len = np.log(eff_len)
        if log_zero is not None:
            log_len[np.isinf(log_len)] = np.log(log_zero)
        ann["AveEffLen"], ann["Max2MinEffLen"] = _length_stats(log_len)
    overdisp, od_prior = od.estimate()
    ann["Overdispersion"] = overdisp

    if len(ids) and _is_gencode(ids[0]):
        A = split_gencode_tx_names(ids)
        for c in _GENCODE_COLUMNS[1:]:
            if c != "AnnLength" or not drop_ann_length:
                ann[c] = A[c].to_numpy()
        ann.index = pd.Index(A["EnsT"].to_numpy())

    counts, divided = _divide(counts, overdisp, n_resampled, divide, verbose)
    op = None
    if offset_prior and log_len is not None:
        op = log_len - log_len.mean(axis=1, keepdims=True)
    return _make_output(counts, ann, names, od_prior, types, divided, op, eff_len, dgelist)


def _gene_map(tx_ids, tx2gene, remove_version, verbose):
    """Gene IDs per transcript, and gene annotation when unpacked from Gencode."""
    tx_ids = np.asarray(tx_ids, dtype=object)
    is_gencode = len(tx_ids) > 0 and _is_gencode(tx_ids[0])
    if tx2gene is None:
        if not is_gencode:
            raise ValueError("tx2gene not provided and transcript names do not appear to be from Gencode")
        if verbose:
            print("Summarizing to genewise counts using Gencode's embedded annotation")
        txann = split_gencode_tx_names(tx_ids, remove_version=remove_version)
        return np.asarray(txann["EnsG"], dtype=object), collapse_gencode_tx2gene(txann)

    tx2gene = pd.DataFrame(tx2gene)
    if tx2gene.shape[1] < 2:
        raise ValueError("tx2gene doesn't have two columns")
    tx_col = tx2gene.iloc[:, 0].astype(str).to_numpy(dtype=object)
    gene_col = tx2gene.iloc[:, 1].astype(str).to_numpy(dtype=object)
    if is_gencode and not _is_gencode(tx_col[0]):
        if verbose:
            print("Transcript names are Gencode-style but tx2gene's are not, so unpacking them")
        tx_ids = np.array([s.split("|", 1)[0] for s in tx_ids], dtype=object)
    if remove_version:
        tx_col, gene_col, tx_ids = (_remove_version(tx_col), _remove_version(gene_col),
                                    _remove_version(tx_ids))
    pos = pd.Index(tx_col).get_indexer(tx_ids)
    if np.any(pos < 0):
        raise ValueError("Tx names not found in first column of tx2gene")
    if verbose:
        print("Summarizing to genewise counts using tx2gene")
    return gene_col[pos], None


def _gene_lengths(method, tpm, eff_tx, codes, ngene, ntx_per_gene, gene_counts):
    """Genewise effective lengths per sample, as catchSalmonGene computes them."""
    if method == "moderate":
        tpm2 = tpm + 1e-6 + tpm.mean(axis=1, keepdims=True) / 100
        return _rowsum(tpm2 * eff_tx, codes, ngene) / _rowsum(tpm2, codes, ngene)
    if method == "simple":
        return _rowsum(eff_tx, codes, ngene) / ntx_per_gene[:, None]
    if method == "tximport":
        # Weighted by TPM; genes with no counts get their transcripts'
        # average effective length, and samples where a gene has no TPM
        # get the gene's geometric mean over the other samples
        zero_gene = gene_counts.sum(axis=1) == 0
        zero_tx = zero_gene[codes]
        with np.errstate(divide="ignore", invalid="ignore"):
            len_ = _rowsum(tpm * eff_tx * ~zero_tx[:, None], codes, ngene) / \
                _rowsum(tpm * ~zero_tx[:, None], codes, ngene)
        m = _rowsum(eff_tx.mean(axis=1, keepdims=True) * zero_tx[:, None], codes, ngene)[:, 0]
        len_[zero_gene] = (m[zero_gene] / ntx_per_gene[zero_gene])[:, None]
        if np.any(np.isnan(len_)):
            with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                gm = np.exp(np.nanmean(np.log(len_), axis=1))
            # Genes with no TPM in any sample: the unweighted average, as
            # for genes with no counts
            none = np.isnan(gm)
            if np.any(none):
                avg = _rowsum(eff_tx.mean(axis=1, keepdims=True), codes, ngene)[:, 0] / ntx_per_gene
                gm[none] = avg[none]
            len_ = np.where(np.isnan(len_), gm[:, None], len_)
        return len_
    raise ValueError("gene_length must be 'moderate', 'tximport' or 'simple'")


def _assemble_gene(samples, tx2gene=None, remove_version=True, has_eff_len=True,
                   impute_eff_len=True, gene_length="moderate", divide=True,
                   offset_prior=True, dgelist=True, verbose=True):
    """Gene-level output, as catchSalmonGene/catchKallistoGene."""
    names, types = [], []
    n_resampled = 0
    for j, s in enumerate(samples):
        if j == 0:
            ens_g, gene_ann = _gene_map(s["ids"], tx2gene, remove_version, verbose)
            codes, genes = pd.factorize(ens_g, sort=False)
            ngene, ntx = len(genes), len(ens_g)
            ntx_per_gene = np.bincount(codes, minlength=ngene)
            length = None if s["length"] is None else np.asarray(s["length"], dtype=np.float64)
            cols_counts, cols_tpm, cols_eff = [], [], []
            od = _Overdispersion(ngene)
        names.append(s["name"])
        types.append(s["type"])
        cols_counts.append(_rowsum(np.asarray(s["counts"], dtype=np.float64)[:, None], codes, ngene)[:, 0])
        if has_eff_len:
            cols_tpm.append(np.asarray(s["tpm"], dtype=np.float64))
            cols_eff.append(np.asarray(s["eff_len"], dtype=np.float64))
        if s.get("boot") is not None:
            od.add_boot(_rowsum(s["boot"], codes, ngene))
            n_resampled += 1
    if not names:
        raise ValueError("No samples found")
    counts = np.column_stack(cols_counts)

    ann = pd.DataFrame(index=pd.Index(np.asarray(genes, dtype=object)))
    if gene_ann is not None:
        ann = gene_ann.copy()
    ann["NTx"] = ntx_per_gene
    if length is not None:
        max_len = np.full(ngene, -np.inf)
        np.maximum.at(max_len, codes, length)
        ann["MaxTxLen"] = max_len
    log_len = eff_gene = tpm_gene = None
    if has_eff_len:
        tpm = np.column_stack(cols_tpm)
        eff_tx = np.column_stack(cols_eff)
        if impute_eff_len:
            eff_tx = _impute_effective_lengths(length, eff_tx)
        eff_gene = _gene_lengths(gene_length, tpm, eff_tx, codes, ngene, ntx_per_gene, counts)
        log_len = np.log(eff_gene)
        ann["AveEffLen"], ann["Max2MinEffLen"] = _length_stats(log_len)
        tpm_gene = _rowsum(tpm, codes, ngene)
    overdisp, od_prior = od.estimate()
    ann["Overdispersion"] = overdisp

    counts, divided = _divide(counts, overdisp, n_resampled, divide, verbose)
    op = None
    if offset_prior and log_len is not None:
        op = log_len - log_len.mean(axis=1, keepdims=True)
    return _make_output(counts, ann, names, od_prior, types, divided, op, eff_gene,
                        dgelist, tpm=tpm_gene)


# =====================================================================
# Format readers
# =====================================================================

def _sample_paths(parent_dir, sample_dirs, marker):
    """Sample directories: sample_dirs under parent_dir, or every
    subdirectory of parent_dir containing one of the marker files."""
    if isinstance(parent_dir, (list, tuple)):
        # Older edgePython calls passed a list of sample directories
        parent_dir, sample_dirs = None, list(parent_dir)
    if sample_dirs is None:
        parent = "." if parent_dir is None else parent_dir
        sample_dirs = sorted(d for d in os.listdir(parent)
                             if any(os.path.exists(os.path.join(parent, d, m)) for m in marker))
        if parent_dir is None:
            parent_dir = "."
    if isinstance(sample_dirs, str):
        sample_dirs = [sample_dirs]
    paths = [str(d) if parent_dir is None else os.path.join(parent_dir, str(d)) for d in sample_dirs]
    if not paths:
        raise ValueError("No sample directories found")
    return paths


def _salmon_samples(paths, verbose):
    for p in paths:
        name = os.path.basename(os.path.normpath(p))
        quant_file = os.path.join(p, "quant.sf")
        if not os.path.exists(quant_file):
            found = sorted(glob.glob(os.path.join(p, "quant.sf*")))
            if not found:
                raise FileNotFoundError(f"quant.sf file not found in {p}")
            quant_file = found[0]
        with open(os.path.join(p, "aux_info", "meta_info.json")) as f:
            meta = json.load(f)
        ntx = meta.get("num_targets", meta.get("num_valid_targets"))
        if ntx is None:
            raise ValueError("Can't find number of targets")
        n_boot = meta.get("num_bootstraps")
        if n_boot is None:
            raise ValueError("Can't find number of bootstraps")
        samp_type = meta.get("samp_type", "bootstrap")
        if verbose:
            print(f"Reading {name}, {ntx} transcripts, {n_boot} {samp_type} samples")
        q = pd.read_csv(quant_file, sep="\t")
        boot = None
        if n_boot > 0:
            with gzip.open(os.path.join(p, "aux_info", "bootstrap", "bootstraps.gz"), "rb") as f:
                raw = f.read(8 * ntx * n_boot)
            boot = np.frombuffer(raw, dtype="<f8").reshape((ntx, n_boot), order="F")
        yield {"name": name, "ids": q["Name"].astype(str).to_numpy(), "length": q["Length"].to_numpy(),
               "counts": q["NumReads"].to_numpy(), "eff_len": q["EffectiveLength"].to_numpy(),
               "tpm": q["TPM"].to_numpy(), "boot": boot, "type": samp_type}


def _kallisto_sample(name, h5_file=None, tsv_file=None, boot_tsvs=(), verbose=True):
    """One kallisto sample from abundance.h5, or from abundance.tsv with
    any plaintext bootstraps (bs_abundance_*.tsv)."""
    if h5_file is not None:
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py package required for kallisto H5 files. "
                              "Install with: pip install h5py")
        with h5py.File(h5_file, "r") as f:
            aux = f["aux"]
            ids = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in aux["ids"][:]],
                           dtype=object)
            length = np.asarray(aux["lengths"][:], dtype=np.float64)
            eff_len = np.asarray(aux["eff_lengths"][:], dtype=np.float64)
            n_boot = int(np.asarray(aux["num_bootstrap"]).ravel()[0])
            counts = np.asarray(f["est_counts"][:], dtype=np.float64)
            boot = None
            if n_boot > 0:
                boot = np.column_stack([np.asarray(f["bootstrap"][k][:], dtype=np.float64)
                                        for k in f["bootstrap"].keys()])
    else:
        q = pd.read_csv(tsv_file, sep="\t")
        ids = q["target_id"].astype(str).to_numpy(dtype=object)
        length = q["length"].to_numpy(dtype=np.float64)
        eff_len = q["eff_length"].to_numpy(dtype=np.float64)
        counts = q["est_counts"].to_numpy(dtype=np.float64)
        n_boot = len(boot_tsvs)
        boot = None
        if n_boot:
            boot = np.column_stack([pd.read_csv(b, sep="\t")["est_counts"].to_numpy(dtype=np.float64)
                                    for b in boot_tsvs])
    if verbose:
        print(f"Reading {name}, {len(ids)} transcripts, {n_boot} bootstraps")
    x = counts / eff_len
    return {"name": name, "ids": ids, "length": length, "counts": counts, "eff_len": eff_len,
            "tpm": 1e6 * x / x.sum(), "boot": boot, "type": "bootstrap"}


def _kallisto_dir_sample(p, verbose, fmt=None):
    """One kallisto sample directory: abundance.h5 if present (or fmt='h5'),
    otherwise abundance.tsv (fmt='tsv')."""
    name = os.path.basename(os.path.normpath(p))
    h5_file = os.path.join(p, "abundance.h5")
    if fmt == "h5" or (fmt is None and os.path.exists(h5_file)):
        return _kallisto_sample(name, h5_file=h5_file, verbose=verbose)
    tsv_file = os.path.join(p, "abundance.tsv")
    if not os.path.exists(tsv_file):
        raise FileNotFoundError(f"abundance.h5 or abundance.tsv not found in {p}")
    return _kallisto_sample(name, tsv_file=tsv_file,
                            boot_tsvs=_numbered(os.path.join(p, "bs_abundance_*.tsv")),
                            verbose=verbose)


def _numbered(pattern):
    """Files matching pattern, ordered by the number before the extension."""
    files = glob.glob(pattern)
    key = lambda f: int(re.search(r"(\d+)\.\w+$", f).group(1))
    return sorted(files, key=key)


def _kallisto_samples(paths, verbose, fmt=None):
    for p in paths:
        yield _kallisto_dir_sample(p, verbose, fmt)


def _oarfish_prefixes(parent_dir, prefixes):
    if prefixes is None:
        parent = "." if parent_dir is None else parent_dir
        quant = sorted(f for f in os.listdir(parent) if f.endswith(".quant"))
        prefixes = [f[:-6] for f in quant]
    else:
        prefixes = [str(p)[:-6] if str(p).endswith(".quant") else str(p) for p in np.atleast_1d(prefixes)]
    if not prefixes:
        raise ValueError("No oarfish output files")
    if parent_dir is not None:
        prefixes = [os.path.join(parent_dir, p) for p in prefixes]
    return prefixes


def _oarfish_samples(prefixes, verbose):
    for prefix in prefixes:
        quant_file = prefix + ".quant"
        if not os.path.exists(quant_file):
            raise FileNotFoundError(f"{quant_file} not found")
        with open(prefix + ".meta_info.json") as f:
            meta = json.load(f)
        n_boot = meta.get("num_bootstraps")
        if n_boot is None:
            raise ValueError("Can't find number of bootstraps")
        if verbose:
            print(f"Reading {prefix}, {n_boot} bootstraps")
        q = pd.read_csv(quant_file, sep="\t")
        boot = None
        if n_boot > 0:
            try:
                boot = pd.read_parquet(prefix + ".infreps.pq").to_numpy(dtype=np.float64)
            except ImportError:
                raise ImportError("pyarrow package required for oarfish bootstraps. "
                                  "Install with: pip install pyarrow")
        yield {"name": os.path.basename(prefix), "ids": q["tname"].astype(str).to_numpy(dtype=object),
               "length": q["len"].to_numpy(dtype=np.float64), "counts": q["num_reads"].to_numpy(),
               "eff_len": None, "tpm": None, "boot": boot, "type": "bootstrap"}


def _rsem_files(parent_dir, files, suffix):
    parent = "." if parent_dir is None else parent_dir
    if files is None:
        files = sorted(f for f in os.listdir(parent) if f.endswith(suffix))
    else:
        files = [str(f) for f in np.atleast_1d(files)]
    if not files:
        raise ValueError(f"No {suffix[1:]} files")
    return [os.path.join(parent, f) for f in files]


def _rsem_name(f):
    """File name without its last two extensions (edgeR's removeExt twice)."""
    return os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]


def _rsem_samples(files, ngibbs, id_col, verbose):
    ngibbs = np.resize(np.atleast_1d(ngibbs), len(files))
    for f, ng in zip(files, ngibbs):
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} not found")
        q = pd.read_csv(f, sep="\t")
        if "expected_count" not in q.columns:
            raise ValueError("File doesn't contain expected_count column")
        if verbose:
            print(f"Reading {f}, {len(q)} transcripts, {int(ng)} Gibbs samples")
        gibbs = None
        if "posterior_mean_count" in q.columns and ng > 0:
            gibbs = (q["posterior_mean_count"].to_numpy(dtype=np.float64),
                     q["posterior_standard_deviation_of_count"].to_numpy(dtype=np.float64), int(ng))
        yield {"name": _rsem_name(f), "ids": q[id_col].astype(str).to_numpy(dtype=object),
               "length": q["length"].to_numpy(dtype=np.float64),
               "counts": q["expected_count"].to_numpy(dtype=np.float64),
               "eff_len": q["effective_length"].to_numpy(dtype=np.float64), "tpm": None,
               "boot": None, "gibbs": gibbs, "type": "gibbs",
               "ntx": q[q.columns[1]].astype(str).str.count(",").to_numpy() + 1}


def _lr_kallisto_samples(parent_dir, sample_names, verbose):
    """Samples from kallisto quant-tcc output: per-sample directories or
    files written with --matrix-to-directories / --matrix-to-files, or the
    matrix.abundance.mtx matrix (which has no bootstraps)."""
    def name(j, default):
        return default if sample_names is None else str(sample_names[j])

    dirs = _numbered_dirs(os.path.join(parent_dir, "abundance_*"))
    if dirs:
        for j, d in enumerate(dirs):
            s = _kallisto_dir_sample(d, verbose)
            s["name"] = name(j, s["name"])
            yield s
        return
    h5s = _numbered(os.path.join(parent_dir, "abundance_*.h5"))
    tsvs = _numbered(os.path.join(parent_dir, "abundance_*.tsv"))
    if h5s or tsvs:
        for j, f in enumerate(h5s or tsvs):
            n = re.search(r"abundance_(\d+)\.\w+$", f).group(1)
            stem = f"abundance_{n}"
            if h5s:
                yield _kallisto_sample(name(j, stem), h5_file=f, verbose=verbose)
            else:
                boots = _numbered(os.path.join(parent_dir, f"bs_abundance_{n}_*.tsv"))
                yield _kallisto_sample(name(j, stem), tsv_file=f, boot_tsvs=boots, verbose=verbose)
        return

    mtx = os.path.join(parent_dir, "matrix.abundance.mtx")
    if not os.path.exists(mtx):
        raise FileNotFoundError(f"No kallisto quant-tcc output found in {parent_dir}")
    from scipy.io import mmread
    warnings.warn("matrix.abundance.mtx has no bootstraps, so no overdispersions are estimated; "
                  "run kallisto quant-tcc with -b and --matrix-to-directories to get them")
    counts = np.asarray(mmread(mtx).todense(), dtype=np.float64)
    with open(os.path.join(parent_dir, "transcripts.txt")) as f:
        ids = np.array([line.strip() for line in f if line.strip()], dtype=object)
    eff = None
    eff_file = os.path.join(parent_dir, "matrix.efflens.mtx")
    if os.path.exists(eff_file):
        eff = np.asarray(mmread(eff_file).todense(), dtype=np.float64)
    tpm = np.asarray(mmread(os.path.join(parent_dir, "matrix.abundance.tpm.mtx")).todense(),
                     dtype=np.float64) if os.path.exists(
        os.path.join(parent_dir, "matrix.abundance.tpm.mtx")) else None
    for j in range(counts.shape[0]):
        if verbose:
            print(f"Reading sample {j + 1} of {counts.shape[0]}, {len(ids)} transcripts, 0 bootstraps")
        yield {"name": name(j, f"sample{j + 1}"), "ids": ids, "length": None, "counts": counts[j],
               "eff_len": None if eff is None else eff[j], "tpm": None if tpm is None else tpm[j],
               "boot": None, "type": "none"}


def _numbered_dirs(pattern):
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    return sorted(dirs, key=lambda d: int(re.search(r"_(\d+)$", d).group(1)))


def _lr_eff_len(samples):
    """Matrix-mode effective lengths are only written for transcripts with
    reads; fill the others with the transcript's geometric mean, or 1."""
    samples = list(samples)
    if not samples or samples[0]["eff_len"] is None:
        return samples, False
    eff = np.column_stack([s["eff_len"] for s in samples])
    if samples[0]["length"] is None and np.any(eff == 0):
        with np.errstate(divide="ignore"):
            log_eff = np.where(eff > 0, np.log(eff), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            gm = np.exp(np.nanmean(log_eff, axis=1))
        gm[~np.isfinite(gm)] = 1.0
        eff = np.where(eff > 0, eff, gm[:, None])
        for j, s in enumerate(samples):
            s["eff_len"] = eff[:, j]
    return samples, True


# =====================================================================
# Public readers
# =====================================================================

def catch_salmon(parent_dir=None, sample_dirs=None, dgelist=True, divide=True,
                 impute_eff_len=True, offset_prior=True, verbose=True):
    """Read Salmon transcript quantifications with bootstrap or Gibbs samples.

    Port of edgeR 5.0's catchSalmon(). Overdispersions estimated from the
    resamples are stored in genes['Overdispersion'] and, by default, the
    counts are divided by them. Gencode transcript names are unpacked into
    annotation columns.

    Parameters
    ----------
    parent_dir : str, optional
        Directory holding one Salmon output directory per sample. By
        default every subdirectory with an aux_info directory is read.
    sample_dirs : list of str, optional
        Sample directories (under parent_dir, if given).
    dgelist : bool
        Return a DGEList; otherwise a dict with counts, effective.length,
        annotation, overdispersion.prior, resample.type and divided.counts.
    divide : bool
        Divide the counts by the overdispersions.
    impute_eff_len : bool
        Replace effective lengths Salmon could not estimate (equal to the
        transcript length) by the transcript's smallest estimate in other
        samples.
    offset_prior : bool
        Store the centred log effective lengths as offset.prior.
    verbose : bool
        Print progress.
    """
    paths = _sample_paths(parent_dir, sample_dirs, ["aux_info"])
    return _assemble_tx(_salmon_samples(paths, verbose), impute_eff_len=impute_eff_len,
                        divide=divide, offset_prior=offset_prior, dgelist=dgelist, verbose=verbose)


def catch_salmon_gene(parent_dir=None, sample_dirs=None, tx2gene=None, remove_version=True,
                      dgelist=True, divide=True, impute_eff_len=True, offset_prior=True,
                      gene_length="moderate", verbose=True):
    """Read Salmon quantifications and summarize them to genes.

    Port of edgeR 5.0's catchSalmonGene(). Counts and resamples are summed
    over each gene's transcripts before the overdispersions are estimated.

    Parameters
    ----------
    parent_dir, sample_dirs : optional
        As for catch_salmon.
    tx2gene : DataFrame, optional
        Transcript IDs (first column) and gene IDs (second column).
        Without it, genes are read from Gencode transcript names.
    remove_version : bool
        Remove version numbers from transcript and gene IDs.
    gene_length : str
        Genewise effective lengths: 'moderate' (TPM-weighted, moderated
        towards the unweighted average), 'tximport' (TPM-weighted) or
        'simple' (unweighted).
    dgelist, divide, impute_eff_len, offset_prior, verbose :
        As for catch_salmon.
    """
    paths = _sample_paths(parent_dir, sample_dirs, ["aux_info"])
    return _assemble_gene(_salmon_samples(paths, verbose), tx2gene=tx2gene,
                          remove_version=remove_version, impute_eff_len=impute_eff_len,
                          gene_length=gene_length, divide=divide, offset_prior=offset_prior,
                          dgelist=dgelist, verbose=verbose)


def catch_kallisto(parent_dir=None, sample_dirs=None, dgelist=True, divide=True,
                   impute_eff_len=True, offset_prior=True, verbose=True):
    """Read kallisto transcript quantifications with bootstrap samples.

    Port of edgeR 5.0's catchKallisto(). Each sample directory holds
    abundance.h5 or, as an edgePython extension, abundance.tsv with any
    plaintext bootstraps (bs_abundance_*.tsv). Arguments as for
    catch_salmon; by default every subdirectory of parent_dir holding
    abundance.h5 or abundance.tsv is read.
    """
    paths = _sample_paths(parent_dir, sample_dirs, ["abundance.h5", "abundance.tsv"])
    return _assemble_tx(_kallisto_samples(paths, verbose), impute_eff_len=impute_eff_len,
                        divide=divide, offset_prior=offset_prior, dgelist=dgelist, verbose=verbose)


def catch_kallisto_gene(parent_dir=None, sample_dirs=None, tx2gene=None, remove_version=True,
                        dgelist=True, divide=True, impute_eff_len=True, offset_prior=True,
                        gene_length="moderate", verbose=True):
    """Read kallisto quantifications and summarize them to genes.

    Port of edgeR 5.0's catchKallistoGene(). Arguments as for
    catch_salmon_gene and catch_kallisto.
    """
    paths = _sample_paths(parent_dir, sample_dirs, ["abundance.h5", "abundance.tsv"])
    return _assemble_gene(_kallisto_samples(paths, verbose), tx2gene=tx2gene,
                          remove_version=remove_version, impute_eff_len=impute_eff_len,
                          gene_length=gene_length, divide=divide, offset_prior=offset_prior,
                          dgelist=dgelist, verbose=verbose)


def catch_oarfish(parent_dir=None, prefixes=None, dgelist=True, divide=True, verbose=True):
    """Read oarfish transcript quantifications with bootstrap samples.

    Port of edgeR 5.0's catchOarfish(). Each sample is a set of files
    <prefix>.quant, <prefix>.meta_info.json and <prefix>.infreps.pq (the
    bootstraps, read with pyarrow).

    Parameters
    ----------
    parent_dir : str, optional
        Directory holding the files; by default every *.quant file in it
        (or the working directory) is read.
    prefixes : list of str, optional
        Sample file prefixes.
    dgelist, divide, verbose :
        As for catch_salmon. oarfish reports no effective lengths, so
        there is no offset.prior.
    """
    prefixes = _oarfish_prefixes(parent_dir, prefixes)
    return _assemble_tx(_oarfish_samples(prefixes, verbose), has_eff_len=False,
                        drop_ann_length=False, divide=divide, dgelist=dgelist, verbose=verbose)


def catch_oarfish_gene(parent_dir=None, prefixes=None, tx2gene=None, remove_version=True,
                       dgelist=True, divide=True, verbose=True):
    """Read oarfish quantifications and summarize them to genes.

    edgePython addition, built as catchSalmonGene() is: counts and
    bootstraps are summed over each gene's transcripts. Arguments as for
    catch_oarfish and catch_salmon_gene.
    """
    prefixes = _oarfish_prefixes(parent_dir, prefixes)
    return _assemble_gene(_oarfish_samples(prefixes, verbose), tx2gene=tx2gene,
                          remove_version=remove_version, has_eff_len=False, divide=divide,
                          dgelist=dgelist, verbose=verbose)


def catch_rsem(parent_dir=None, files=None, ngibbs=100, dgelist=True, divide=True,
               offset_prior=True, verbose=True):
    """Read RSEM transcript quantifications with Gibbs sampling summaries.

    Port of edgeR 5.0's catchRSEM(). Reads *.isoforms.results files, which
    need RSEM's posterior_mean_count and
    posterior_standard_deviation_of_count columns (rsem-calculate-expression
    --calc-pme) for the overdispersions.

    Parameters
    ----------
    parent_dir : str, optional
        Directory holding the files; by default every *.isoforms.results
        file in it (or the working directory) is read.
    files : list of str, optional
        File names (under parent_dir, if given).
    ngibbs : int or array-like
        Number of Gibbs samples RSEM drew for each sample.
    dgelist, divide, offset_prior, verbose :
        As for catch_salmon.
    """
    if isinstance(parent_dir, (list, tuple)):
        # Older edgePython calls passed a list of files
        parent_dir, files = None, list(parent_dir)
    files = _rsem_files(parent_dir, files, ".isoforms.results")
    return _assemble_tx(_rsem_samples(files, ngibbs, "transcript_id", verbose),
                        impute_eff_len=False, log_zero=1e-8, divide=divide,
                        offset_prior=offset_prior, dgelist=dgelist, verbose=verbose)


def catch_rsem_gene(parent_dir=None, files=None, ngibbs=100, remove_version=True,
                    dgelist=True, divide=True, offset_prior=True, verbose=True):
    """Read RSEM gene quantifications with Gibbs sampling summaries.

    edgePython addition. Reads RSEM's own gene-level *.genes.results
    files, whose Gibbs summaries already account for the transcripts'
    correlations, and estimates gene overdispersions from them as
    catch_rsem does for transcripts. Gene effective lengths of zero (genes
    with no expression in a sample) are replaced by the gene's geometric
    mean over the other samples, or 1.

    Parameters
    ----------
    parent_dir, files : optional
        As for catch_rsem, but files default to *.genes.results.
    ngibbs : int or array-like
        Number of Gibbs samples RSEM drew for each sample.
    remove_version : bool
        Remove version numbers from gene IDs.
    dgelist, divide, offset_prior, verbose :
        As for catch_salmon.
    """
    files = _rsem_files(parent_dir, files, ".genes.results")
    samples = list(_rsem_samples(files, ngibbs, "gene_id", verbose))
    eff = np.column_stack([s["eff_len"] for s in samples])
    with np.errstate(divide="ignore"):
        log_eff = np.where(eff > 0, np.log(eff), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gm = np.exp(np.nanmean(log_eff, axis=1))
    gm[~np.isfinite(gm)] = 1.0
    eff = np.where(eff > 0, eff, gm[:, None])
    ntx = samples[0]["ntx"]
    for j, s in enumerate(samples):
        s["eff_len"] = eff[:, j]
        s["length"] = None
        if remove_version:
            s["ids"] = _remove_version(s["ids"])
    y = _assemble_tx(samples, impute_eff_len=False, divide=divide,
                     offset_prior=offset_prior, dgelist=dgelist, verbose=verbose)
    ann = y["genes"] if dgelist else y["annotation"]
    ann.insert(0, "NTx", ntx)
    return y


def catch_lr_kallisto(parent_dir=".", sample_names=None, dgelist=True, divide=True,
                      impute_eff_len=True, offset_prior=True, verbose=True):
    """Read lr-kallisto (kallisto quant-tcc) transcript quantifications.

    edgePython addition, built as catchKallisto() is. Reads the output
    directory of kallisto quant-tcc (used by lr-kallisto for long reads).
    Run quant-tcc with -b <n> and --matrix-to-directories (or
    --matrix-to-files) so that each sample's abundance.h5 or
    abundance.tsv carries bootstraps; then overdispersions are estimated
    as for catch_kallisto. With only matrix.abundance.mtx there are no
    bootstraps: counts are read and not divided, and effective lengths
    come from matrix.efflens.mtx when quant-tcc wrote one.

    Parameters
    ----------
    parent_dir : str
        The quant-tcc output directory.
    sample_names : list of str, optional
        Sample names, in quant-tcc's sample order.
    dgelist, divide, impute_eff_len, offset_prior, verbose :
        As for catch_kallisto.
    """
    samples, has_eff = _lr_eff_len(_lr_kallisto_samples(parent_dir, sample_names, verbose))
    has_length = bool(samples) and samples[0]["length"] is not None
    return _assemble_tx(samples, has_eff_len=has_eff, impute_eff_len=impute_eff_len and has_length,
                        divide=divide, offset_prior=offset_prior, dgelist=dgelist, verbose=verbose)


def catch_lr_kallisto_gene(parent_dir=".", sample_names=None, tx2gene=None, remove_version=True,
                           dgelist=True, divide=True, impute_eff_len=True, offset_prior=True,
                           gene_length="moderate", verbose=True):
    """Read lr-kallisto (kallisto quant-tcc) quantifications and summarize
    them to genes.

    edgePython addition, built as catchKallistoGene() is. Arguments as for
    catch_lr_kallisto and catch_kallisto_gene.
    """
    samples, has_eff = _lr_eff_len(_lr_kallisto_samples(parent_dir, sample_names, verbose))
    if has_eff and samples[0]["tpm"] is None:
        for s in samples:
            x = s["counts"] / s["eff_len"]
            s["tpm"] = 1e6 * x / x.sum()
    has_length = bool(samples) and samples[0]["length"] is not None
    return _assemble_gene(samples, tx2gene=tx2gene, remove_version=remove_version,
                          has_eff_len=has_eff, impute_eff_len=impute_eff_len and has_length,
                          gene_length=gene_length, divide=divide, offset_prior=offset_prior,
                          dgelist=dgelist, verbose=verbose)
