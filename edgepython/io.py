# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
I/O functions for edgePython.

Port of edgeR's readDGE, read10X, featureCountsToMatrix,
catchSalmon, catchKallisto, catchRSEM, catchOarfish.
"""

import os
import warnings
import numpy as np
import pandas as pd


def read_dge(files, path=None, columns=(0, 1), group=None, labels=None, sep='\t'):
    """Read and collate count data files.

    Port of edgeR's readDGE.

    Parameters
    ----------
    files : list of str or DataFrame
        File names or DataFrame with 'files' column.
    path : str, optional
        Path prefix for files.
    columns : tuple of int
        Column indices for gene IDs and counts (0-indexed).
    group : array-like, optional
        Group factor.
    labels : list of str, optional
        Sample labels.
    sep : str
        Field separator.

    Returns
    -------
    DGEList-like dict.
    """
    from .dgelist import make_dgelist

    if isinstance(files, pd.DataFrame):
        samples = files.copy()
        if labels is not None:
            samples.index = labels
        if 'files' not in samples.columns:
            raise ValueError("file names not found")
        file_list = samples['files'].astype(str).tolist()
    else:
        file_list = [str(f) for f in files]
        if labels is None:
            labels = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
        samples = pd.DataFrame({'files': file_list}, index=labels)

    nfiles = len(file_list)

    if group is not None:
        samples['group'] = group
    if 'group' not in samples.columns:
        samples['group'] = 1

    # Read files
    all_data = {}
    all_tags = {}
    for i, fn in enumerate(file_list):
        if path is not None:
            fp = os.path.join(path, fn)
        else:
            fp = fn
        df = pd.read_csv(fp, sep=sep, header=0)
        tag_col = df.columns[columns[0]]
        count_col = df.columns[columns[1]]
        tags = df[tag_col].astype(str).values
        if len(tags) != len(set(tags)):
            raise ValueError(f"Repeated row names in {fn}. Row names must be unique.")
        all_tags[fn] = tags
        all_data[fn] = df[count_col].values

    # Collate counts
    all_gene_ids = []
    seen = set()
    for fn in file_list:
        for t in all_tags[fn]:
            if t not in seen:
                all_gene_ids.append(t)
                seen.add(t)

    ntags = len(all_gene_ids)
    counts = np.zeros((ntags, nfiles), dtype=np.float64)
    tag_to_idx = {t: i for i, t in enumerate(all_gene_ids)}

    for i, fn in enumerate(file_list):
        for j, tag in enumerate(all_tags[fn]):
            counts[tag_to_idx[tag], i] = all_data[fn][j]

    samples['lib.size'] = counts.sum(axis=0)
    samples['norm.factors'] = 1.0

    return make_dgelist(counts, samples=samples,
                        genes=pd.DataFrame({'GeneID': all_gene_ids}))


def read_10x(path='.', mtx=None, genes=None, barcodes=None, as_dgelist=True):
    """Read 10X Genomics CellRanger output.

    Port of edgeR's read10X.

    Parameters
    ----------
    path : str
        Directory containing 10X files.
    mtx : str, optional
        Matrix file name.
    genes : str, optional
        Genes/features file name.
    barcodes : str, optional
        Barcodes file name.
    as_dgelist : bool
        Return DGEList-like dict.

    Returns
    -------
    DGEList-like dict or dict with counts/genes/samples.
    """
    from scipy.io import mmread

    files = os.listdir(path)

    if mtx is None:
        for candidate in ['matrix.mtx.gz', 'matrix.mtx']:
            if candidate in files:
                mtx = candidate
                break
        if mtx is None:
            raise FileNotFoundError("Can't find matrix.mtx file")

    if genes is None:
        for candidate in ['features.tsv.gz', 'features.tsv', 'genes.tsv.gz', 'genes.tsv']:
            if candidate in files:
                genes = candidate
                break
        if genes is None:
            raise FileNotFoundError("Can't find genes/features file")

    if barcodes is None:
        for candidate in ['barcodes.tsv.gz', 'barcodes.tsv']:
            if candidate in files:
                barcodes = candidate
                break

    mtx_path = os.path.join(path, mtx)
    genes_path = os.path.join(path, genes)

    # Read sparse matrix
    sparse_mat = mmread(mtx_path)
    y = np.array(sparse_mat.todense(), dtype=np.float64)

    # Read gene info
    gene_df = pd.read_csv(genes_path, sep='\t', header=None)
    if gene_df.shape[1] >= 2:
        gene_df.columns = ['GeneID', 'Symbol'] + [f'col{i}' for i in range(2, gene_df.shape[1])]
    else:
        gene_df.columns = ['GeneID']

    # Read barcodes
    samples_df = None
    if barcodes is not None:
        barcodes_path = os.path.join(path, barcodes)
        bc = pd.read_csv(barcodes_path, sep='\t', header=None)
        samples_df = pd.DataFrame({'Barcode': bc.iloc[:, 0].values})

    if as_dgelist:
        from .dgelist import make_dgelist
        return make_dgelist(y, genes=gene_df, samples=samples_df)

    return {'counts': y, 'genes': gene_df, 'samples': samples_df}


def catch_salmon(paths, verbose=True):
    """Read Salmon quantification output.

    Parameters
    ----------
    paths : list of str
        Paths to Salmon output directories.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with counts, annotation, and samples.
    """
    return _catch_quant(paths, tool='salmon', verbose=verbose)


def catch_kallisto(paths, verbose=True):
    """Read kallisto quantification output.

    Parameters
    ----------
    paths : list of str
        Paths to kallisto output directories.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with counts, annotation, and samples.
    """
    return _catch_quant(paths, tool='kallisto', verbose=verbose)


def catch_rsem(files, verbose=True):
    """Read RSEM quantification output.

    Parameters
    ----------
    files : list of str
        RSEM output files.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with counts and annotation.
    """
    all_data = []
    gene_ids = None

    for f in files:
        df = pd.read_csv(f, sep='\t')
        if 'expected_count' in df.columns:
            count_col = 'expected_count'
        elif 'FPKM' in df.columns:
            count_col = 'FPKM'
        else:
            count_col = df.columns[1]

        if gene_ids is None:
            gene_ids = df.iloc[:, 0].values
        all_data.append(df[count_col].values)

    counts = np.column_stack(all_data)
    labels = [os.path.splitext(os.path.basename(f))[0] for f in files]

    return {
        'counts': counts,
        'annotation': pd.DataFrame({'GeneID': gene_ids}),
        'samples': pd.DataFrame({'files': files}, index=labels)
    }


def feature_counts_to_dgelist(files):
    """Convert featureCounts output to DGEList.

    Parameters
    ----------
    files : str or list of str
        featureCounts output file(s).

    Returns
    -------
    DGEList-like dict.
    """
    from .dgelist import make_dgelist

    if isinstance(files, str):
        files = [files]

    all_counts = []
    gene_info = None
    sample_names = []

    for f in files:
        df = pd.read_csv(f, sep='\t', comment='#')
        # featureCounts format: Geneid, Chr, Start, End, Strand, Length, Count1, Count2, ...
        if gene_info is None:
            meta_cols = ['Geneid', 'Chr', 'Start', 'End', 'Strand', 'Length']
            gene_cols = [c for c in meta_cols if c in df.columns]
            if gene_cols:
                gene_info = df[gene_cols].copy()
            else:
                gene_info = df.iloc[:, :1].copy()

        count_cols = [c for c in df.columns if c not in
                      ['Geneid', 'Chr', 'Start', 'End', 'Strand', 'Length']]
        for col in count_cols:
            all_counts.append(df[col].values)
            sample_names.append(col)

    counts = np.column_stack(all_counts)
    return make_dgelist(counts, genes=gene_info)


def _catch_quant(paths, tool='salmon', verbose=True):
    """Internal function to read Salmon/kallisto output (legacy)."""
    all_data = []
    gene_ids = None
    labels = []

    for p in paths:
        label = os.path.basename(os.path.normpath(p))
        labels.append(label)

        if tool == 'salmon':
            quant_file = os.path.join(p, 'quant.sf')
        else:
            quant_file = os.path.join(p, 'abundance.tsv')

        if not os.path.exists(quant_file):
            raise FileNotFoundError(f"Cannot find {quant_file}")

        df = pd.read_csv(quant_file, sep='\t')

        if tool == 'salmon':
            count_col = 'NumReads'
            id_col = 'Name'
            length_col = 'EffectiveLength'
        else:
            count_col = 'est_counts'
            id_col = 'target_id'
            length_col = 'eff_length'

        if gene_ids is None:
            gene_ids = df[id_col].values
            if length_col in df.columns:
                eff_length = df[length_col].values
            else:
                eff_length = None

        all_data.append(df[count_col].values)

        if verbose:
            print(f"Reading {label}...")

    counts = np.column_stack(all_data)
    annotation = pd.DataFrame({'GeneID': gene_ids})
    if eff_length is not None:
        annotation['Length'] = eff_length

    return {
        'counts': counts,
        'annotation': annotation,
        'samples': pd.DataFrame({'files': paths}, index=labels)
    }


# =====================================================================
# Overdispersion estimation (shared core)
# =====================================================================

def _accumulate_overdispersion(boot, overdisp, df_arr):
    """Accumulate bootstrap overdispersion statistics for one sample.

    Port of the per-sample loop body shared across edgeR's catchSalmon,
    catchKallisto, catchRSEM, catchOarfish.

    Parameters
    ----------
    boot : ndarray, shape (n_tx, n_boot)
        Bootstrap count matrix for one sample.
    overdisp : ndarray, shape (n_tx,)
        Running overdispersion accumulator (modified in-place).
    df_arr : ndarray, shape (n_tx,)
        Running degrees of freedom accumulator (modified in-place).
    """
    n_boot = boot.shape[1]
    M = boot.mean(axis=1)
    pos = M > 0
    overdisp[pos] += np.sum((boot[pos] - M[pos, np.newaxis]) ** 2, axis=1) / M[pos]
    df_arr[pos] += n_boot - 1


def _estimate_overdispersion(overdisp, df_arr):
    """Estimate per-transcript overdispersion with moderate shrinkage.

    Port of the overdispersion finalization shared across all edgeR
    catch* functions. Applies limited moderation with DFPrior=3.

    Parameters
    ----------
    overdisp : ndarray, shape (n_tx,)
        Accumulated sum of (Boot - M)^2 / M across samples.
    df_arr : ndarray, shape (n_tx,)
        Accumulated degrees of freedom.

    Returns
    -------
    overdisp_final : ndarray, shape (n_tx,)
        Moderated overdispersion estimates (>= 1).
    overdisp_prior : float
        Prior overdispersion value used for shrinkage.
    """
    from scipy.stats import f as f_dist

    pos = df_arr > 0
    n_pos = np.sum(pos)

    if n_pos > 0:
        overdisp[pos] = overdisp[pos] / df_arr[pos]

        df_median = float(np.median(df_arr[pos]))
        df_prior = 3.0
        overdisp_prior = float(np.median(overdisp[pos])) / f_dist.ppf(0.5, dfn=df_median, dfd=df_prior)
        if overdisp_prior < 1.0:
            overdisp_prior = 1.0

        overdisp[pos] = (df_prior * overdisp_prior + df_arr[pos] * overdisp[pos]) / (df_prior + df_arr[pos])
        overdisp = np.maximum(overdisp, 1.0)
        overdisp[~pos] = overdisp_prior
    else:
        overdisp[:] = np.nan
        overdisp_prior = np.nan

    return overdisp, overdisp_prior


# =====================================================================
# Format-specific readers for read_data()
# =====================================================================

def _read_kallisto_h5(paths, verbose):
    """Read kallisto H5 output with bootstrap overdispersion.

    Port of edgeR's catchKallisto.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py package required for kallisto H5 format. "
            "Install with: pip install h5py"
        )

    n_samples = len(paths)
    counts = None
    overdisp = None
    df_arr = None
    ids = None
    lengths = None
    eff_lengths = None

    for j, p in enumerate(paths):
        h5_file = os.path.join(p, 'abundance.h5')
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"abundance.h5 not found in {p}")

        with h5py.File(h5_file, 'r') as f:
            n_tx = len(f['aux']['lengths'])
            n_boot = int(np.asarray(f['aux']['num_bootstrap']).flat[0])

            if verbose:
                label = os.path.basename(os.path.normpath(p))
                print(f"Reading {label}, {n_tx} transcripts, {n_boot} bootstraps")

            if j == 0:
                counts = np.zeros((n_tx, n_samples), dtype=np.float64)
                overdisp = np.zeros(n_tx, dtype=np.float64)
                df_arr = np.zeros(n_tx, dtype=np.int64)

            # Store annotation from each sample (R uses the last sample's aux)
            raw_ids = f['aux']['ids'][:]
            ids = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s)
                            for s in raw_ids])
            lengths = np.asarray(f['aux']['lengths'][:], dtype=np.int64)
            eff_lengths = np.asarray(f['aux']['eff_lengths'][:], dtype=np.float64)

            counts[:, j] = f['est_counts'][:]

            if n_boot > 0 and 'bootstrap' in f:
                boot = np.column_stack([f['bootstrap'][f'bs{k}'][:] for k in range(n_boot)])
                _accumulate_overdispersion(boot, overdisp, df_arr)

    has_bootstraps = np.any(df_arr > 0)
    ann_dict = {'Length': lengths, 'EffectiveLength': eff_lengths}
    if has_bootstraps:
        overdisp_final, overdisp_prior = _estimate_overdispersion(overdisp, df_arr)
        ann_dict['Overdispersion'] = overdisp_final
    else:
        overdisp_prior = None
    annotation = pd.DataFrame(ann_dict, index=ids)

    return counts, annotation, ids, overdisp_prior


def _read_kallisto_tsv(paths, verbose):
    """Read kallisto TSV output (no bootstraps)."""
    n_samples = len(paths)
    all_data = []
    ids = None
    lengths = None
    eff_lengths = None

    for j, p in enumerate(paths):
        tsv_file = os.path.join(p, 'abundance.tsv')
        if not os.path.exists(tsv_file):
            raise FileNotFoundError(f"abundance.tsv not found in {p}")

        if verbose:
            label = os.path.basename(os.path.normpath(p))
            print(f"Reading {label}...")

        df = pd.read_csv(tsv_file, sep='\t')
        if j == 0:
            ids = df['target_id'].values.astype(str)
            lengths = df['length'].values.astype(np.int64)
            eff_lengths = df['eff_length'].values.astype(np.float64)
        all_data.append(df['est_counts'].values.astype(np.float64))

    counts = np.column_stack(all_data)
    annotation = pd.DataFrame({
        'Length': lengths,
        'EffectiveLength': eff_lengths,
    }, index=ids)

    return counts, annotation, ids, None


def _read_kallisto(paths, fmt, verbose):
    """Read kallisto output, dispatching to H5 or TSV."""
    if fmt is None:
        h5_path = os.path.join(paths[0], 'abundance.h5')
        fmt = 'h5' if os.path.exists(h5_path) else 'tsv'

    if fmt == 'h5':
        return _read_kallisto_h5(paths, verbose)
    else:
        return _read_kallisto_tsv(paths, verbose)


def _read_salmon(paths, verbose):
    """Read Salmon output with bootstrap overdispersion.

    Port of edgeR's catchSalmon.
    """
    import json
    import gzip

    n_samples = len(paths)
    counts = None
    overdisp = None
    df_arr = None
    ids = None
    lengths = None
    eff_lengths = None

    for j, p in enumerate(paths):
        label = os.path.basename(os.path.normpath(p))
        quant_file = os.path.join(p, 'quant.sf')
        meta_file = os.path.join(p, 'aux_info', 'meta_info.json')
        boot_file = os.path.join(p, 'aux_info', 'bootstrap', 'bootstraps.gz')

        if not os.path.exists(quant_file):
            raise FileNotFoundError(f"quant.sf not found in {p}")

        # Read meta info for bootstrap count
        n_boot = 0
        n_tx_meta = None
        if os.path.exists(meta_file):
            with open(meta_file) as mf:
                meta = json.load(mf)
            n_tx_meta = meta.get('num_targets') or meta.get('num_valid_targets')
            n_boot = meta.get('num_bootstraps', 0)
            samp_type = meta.get('samp_type', 'bootstrap')
        else:
            samp_type = 'bootstrap'

        quant_df = pd.read_csv(quant_file, sep='\t')
        n_tx = len(quant_df)

        if verbose:
            if n_boot > 0:
                print(f"Reading {label}, {n_tx} transcripts, {n_boot} {samp_type} samples")
            else:
                print(f"Reading {label}, {n_tx} transcripts")

        if j == 0:
            counts = np.zeros((n_tx, n_samples), dtype=np.float64)
            overdisp = np.zeros(n_tx, dtype=np.float64)
            df_arr = np.zeros(n_tx, dtype=np.int64)
            ids = quant_df['Name'].values.astype(str)
            lengths = quant_df['Length'].values.astype(np.int64)
            eff_lengths = quant_df['EffectiveLength'].values.astype(np.float64)

        counts[:, j] = quant_df['NumReads'].values.astype(np.float64)

        # Read binary bootstrap samples
        if n_boot > 0 and os.path.exists(boot_file):
            with gzip.open(boot_file, 'rb') as bf:
                raw = bf.read()
            # R: readBin(con, "double", n=NTx*NBoot); dim(Boot) <- c(NTx, NBoot)
            # R fills column-major: first NTx values = bootstrap 0, etc.
            all_vals = np.frombuffer(raw, dtype=np.float64)
            boot = all_vals.reshape((n_tx, n_boot), order='F')
            _accumulate_overdispersion(boot, overdisp, df_arr)

    has_bootstraps = np.any(df_arr > 0)
    ann_dict = {'Length': lengths, 'EffectiveLength': eff_lengths}
    if has_bootstraps:
        overdisp_final, overdisp_prior = _estimate_overdispersion(overdisp, df_arr)
        ann_dict['Overdispersion'] = overdisp_final
    else:
        overdisp_prior = None
    annotation = pd.DataFrame(ann_dict, index=ids)

    return counts, annotation, ids, overdisp_prior


def _read_oarfish(paths, path, verbose):
    """Read oarfish output with parquet bootstrap overdispersion.

    Port of edgeR's catchOarfish.
    """
    import json

    # If paths is None, auto-discover .quant files in path
    if paths is None:
        if path is None:
            path = '.'
        quant_files = sorted([f for f in os.listdir(path) if f.endswith('.quant')])
        if not quant_files:
            raise FileNotFoundError(f"No oarfish .quant files found in {path}")
        prefixes = [os.path.join(path, f[:-6]) for f in quant_files]
    else:
        # paths is list of prefixes or full .quant paths
        prefixes = []
        for p in paths:
            if p.endswith('.quant'):
                prefixes.append(p[:-6])
            else:
                prefixes.append(p)

    n_samples = len(prefixes)
    counts = None
    overdisp = None
    df_arr = None
    ids = None
    lengths = None

    for j, prefix in enumerate(prefixes):
        quant_file = f"{prefix}.quant"
        meta_file = f"{prefix}.meta_info.json"
        boot_file = f"{prefix}.infreps.pq"

        if not os.path.exists(quant_file):
            raise FileNotFoundError(f"{quant_file} not found")

        n_boot = 0
        if os.path.exists(meta_file):
            with open(meta_file) as mf:
                meta = json.load(mf)
            n_boot = meta.get('num_bootstraps', 0)

        quant_df = pd.read_csv(quant_file, sep='\t')
        n_tx = len(quant_df)

        if verbose:
            label = os.path.basename(prefix)
            print(f"Reading {label}, {n_tx} transcripts, {n_boot} bootstraps")

        if j == 0:
            counts = np.zeros((n_tx, n_samples), dtype=np.float64)
            overdisp = np.zeros(n_tx, dtype=np.float64)
            df_arr = np.zeros(n_tx, dtype=np.int64)
            ids = quant_df['tname'].values.astype(str)
            lengths = quant_df['len'].values.astype(np.int64)

        counts[:, j] = quant_df['num_reads'].values.astype(np.float64)

        if n_boot > 0 and os.path.exists(boot_file):
            try:
                boot = pd.read_parquet(boot_file).values.astype(np.float64)
            except ImportError:
                raise ImportError(
                    "pyarrow package required for oarfish parquet bootstraps. "
                    "Install with: pip install pyarrow"
                )
            _accumulate_overdispersion(boot, overdisp, df_arr)

    has_bootstraps = np.any(df_arr > 0)
    ann_dict = {'Length': lengths}
    if has_bootstraps:
        overdisp_final, overdisp_prior = _estimate_overdispersion(overdisp, df_arr)
        ann_dict['Overdispersion'] = overdisp_final
    else:
        overdisp_prior = None
    annotation = pd.DataFrame(ann_dict, index=ids)

    return counts, annotation, ids, overdisp_prior


def _read_rsem_data(files, path, ngibbs, verbose):
    """Read RSEM output with Gibbs-based overdispersion.

    Port of edgeR's catchRSEM.
    """
    n_samples = len(files)
    if isinstance(ngibbs, (int, float)):
        ngibbs = [int(ngibbs)] * n_samples

    counts = None
    overdisp = None
    df_arr = None
    ids = None
    lengths = None
    eff_lengths = None

    for j, f in enumerate(files):
        full_path = os.path.join(path, f) if path else f
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"{full_path} not found")

        quant_df = pd.read_csv(full_path, sep='\t')
        if 'expected_count' not in quant_df.columns:
            raise ValueError(f"File {f} doesn't contain expected_count column")

        n_tx = len(quant_df)
        ng = ngibbs[j]

        if verbose:
            print(f"Reading {os.path.basename(f)}, {n_tx} transcripts, {ng} Gibbs samples")

        if j == 0:
            counts = np.zeros((n_tx, n_samples), dtype=np.float64)
            overdisp = np.zeros(n_tx, dtype=np.float64)
            df_arr = np.zeros(n_tx, dtype=np.int64)
            id_col = 'transcript_id' if 'transcript_id' in quant_df.columns else quant_df.columns[0]
            ids = quant_df[id_col].values.astype(str)
            lengths = quant_df['length'].values.astype(np.int64) if 'length' in quant_df.columns else None
            eff_lengths = quant_df['effective_length'].values.astype(np.float64) if 'effective_length' in quant_df.columns else None

        counts[:, j] = quant_df['expected_count'].values.astype(np.float64)

        # RSEM Gibbs overdispersion: (ngibbs-1) * S^2 / M
        M_col = quant_df.get('posterior_mean_count')
        S_col = quant_df.get('posterior_standard_deviation_of_count')
        if M_col is not None and S_col is not None and ng > 0:
            M = M_col.values.astype(np.float64)
            S = S_col.values.astype(np.float64)
            pos = M > 0
            overdisp[pos] += (ng - 1) * (S[pos] ** 2) / M[pos]
            df_arr[pos] += ng - 1

    has_bootstraps = np.any(df_arr > 0)
    ann_dict = {}
    if lengths is not None:
        ann_dict['Length'] = lengths
    if eff_lengths is not None:
        ann_dict['EffectiveLength'] = eff_lengths
    if has_bootstraps:
        overdisp_final, overdisp_prior = _estimate_overdispersion(overdisp, df_arr)
        ann_dict['Overdispersion'] = overdisp_final
    else:
        overdisp_prior = None
    annotation = pd.DataFrame(ann_dict, index=ids)

    return counts, annotation, ids, overdisp_prior


def _read_anndata(data, group, labels, obs_col, layer, verbose):
    """Read AnnData object or .h5ad file."""
    try:
        import anndata
    except ImportError:
        raise ImportError(
            "anndata package required for AnnData/.h5ad import. "
            "Install with: pip install anndata"
        )
    from .dgelist import make_dgelist

    if isinstance(data, str):
        if verbose:
            print(f"Reading {data}...")
        adata = anndata.read_h5ad(data)
    else:
        adata = data

    # Extract count matrix: AnnData is obs×var (samples×genes)
    # edgePython needs genes×samples -> transpose
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in AnnData. "
                             f"Available: {list(adata.layers.keys())}")
        X = adata.layers[layer]
    elif 'counts' in adata.layers:
        # Prefer raw counts layer when available (common Scanpy convention)
        X = adata.layers['counts']
    else:
        X = adata.X

    # Handle sparse matrices
    if hasattr(X, 'toarray') and hasattr(X, 'nnz'):
        shape = X.shape
        nnz = X.nnz
        density = nnz / (shape[0] * shape[1]) if shape[0] * shape[1] > 0 else 0
        warnings.warn(
            f"Densifying sparse AnnData matrix ({shape[0]} x {shape[1]}, "
            f"{100*density:.1f}% non-zero, "
            f"{shape[0] * shape[1] * 8 / 1e6:.0f} MB dense). "
            f"edgePython stores counts as dense arrays.",
            stacklevel=2,
        )
        X = X.toarray()
    counts = np.asarray(X, dtype=np.float64).T  # genes × samples

    # Gene annotation from .var
    genes_df = adata.var.copy() if len(adata.var.columns) > 0 else None

    # Sample labels
    if labels is None:
        labels = list(adata.obs_names)

    # Group from obs_col
    if group is None and obs_col is not None:
        if obs_col in adata.obs.columns:
            group = adata.obs[obs_col].values
        else:
            raise ValueError(f"Column '{obs_col}' not found in AnnData.obs. "
                             f"Available: {list(adata.obs.columns)}")

    dge = make_dgelist(counts, group=group, genes=genes_df)
    if labels is not None:
        dge['samples'].index = labels
    return dge


def _parse_rds_metadata(path):
    """Parse key=value metadata file written by R extraction script."""
    metadata = {}
    if not os.path.exists(path):
        return metadata
    with open(path) as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                metadata[key] = None if value == 'NA' else value
    return metadata


def _build_rds_extraction_script(filepath, tmpdir):
    """Build R script that extracts components from an RDS file."""
    r_filepath = filepath.replace('\\', '/')
    r_tmpdir = tmpdir.replace('\\', '/')

    return f'''
suppressPackageStartupMessages(library(methods))
x <- readRDS("{r_filepath}")
tmpdir <- "{r_tmpdir}"
cls <- class(x)[1]

if (inherits(x, "DGEList")) {{
    write.csv(x$counts, file.path(tmpdir, "counts.csv"))
    write.csv(x$samples, file.path(tmpdir, "samples.csv"))
    if (!is.null(x$genes)) {{
        write.csv(x$genes, file.path(tmpdir, "genes.csv"))
    }}

    has_common <- !is.null(x$common.dispersion)
    has_trended <- !is.null(x$trended.dispersion)
    has_tagwise <- !is.null(x$tagwise.dispersion)
    has_alc <- !is.null(x$AveLogCPM)

    if (has_trended || has_tagwise) {{
        disp_df <- data.frame(row.names=rownames(x$counts))
        if (has_trended) disp_df$trended <- x$trended.dispersion
        if (has_tagwise) disp_df$tagwise <- x$tagwise.dispersion
        write.csv(disp_df, file.path(tmpdir, "dispersions.csv"))
    }}

    if (has_alc) {{
        write.csv(data.frame(value=x$AveLogCPM, row.names=rownames(x$counts)),
                  file.path(tmpdir, "AveLogCPM.csv"))
    }}

    prior_df <- ifelse(is.null(x$prior.df), "NA", as.character(x$prior.df))

    metadata <- c(
        paste0("class=", cls),
        paste0("nrow=", nrow(x$counts)),
        paste0("ncol=", ncol(x$counts)),
        paste0("has_genes=", !is.null(x$genes)),
        paste0("has_common_dispersion=", has_common),
        paste0("has_trended_dispersion=", has_trended),
        paste0("has_tagwise_dispersion=", has_tagwise),
        paste0("has_AveLogCPM=", has_alc),
        paste0("common.dispersion=", ifelse(has_common, x$common.dispersion, "NA")),
        paste0("prior.df=", prior_df),
        paste0("has_size_factors=FALSE")
    )
    writeLines(metadata, file.path(tmpdir, "metadata.txt"))

}} else if (isClass("SummarizedExperiment") && is(x, "SummarizedExperiment")) {{
    suppressPackageStartupMessages(library(SummarizedExperiment))

    counts_mat <- as.matrix(assay(x))
    write.csv(counts_mat, file.path(tmpdir, "counts.csv"))

    cd <- as.data.frame(colData(x))
    write.csv(cd, file.path(tmpdir, "samples.csv"))

    rd <- as.data.frame(rowData(x))
    if (ncol(rd) > 0) {{
        write.csv(rd, file.path(tmpdir, "genes.csv"))
    }}

    has_sf <- FALSE
    if (is(x, "DESeqDataSet")) {{
        tryCatch({{
            suppressPackageStartupMessages(library(DESeq2))
            sf <- sizeFactors(x)
            if (!is.null(sf)) {{
                write.csv(data.frame(value=sf, row.names=colnames(x)),
                          file.path(tmpdir, "size_factors.csv"))
                has_sf <- TRUE
            }}
        }}, error = function(e) {{}})
    }}

    metadata <- c(
        paste0("class=", cls),
        paste0("nrow=", nrow(counts_mat)),
        paste0("ncol=", ncol(counts_mat)),
        paste0("has_genes=", ncol(rd) > 0),
        paste0("has_common_dispersion=FALSE"),
        paste0("has_trended_dispersion=FALSE"),
        paste0("has_tagwise_dispersion=FALSE"),
        paste0("has_AveLogCPM=FALSE"),
        paste0("common.dispersion=NA"),
        paste0("prior.df=NA"),
        paste0("has_size_factors=", has_sf)
    )
    writeLines(metadata, file.path(tmpdir, "metadata.txt"))

}} else if (is.matrix(x) || is.data.frame(x)) {{
    write.csv(as.matrix(x), file.path(tmpdir, "counts.csv"))
    metadata <- c(
        paste0("class=", cls),
        paste0("nrow=", nrow(x)),
        paste0("ncol=", ncol(x)),
        paste0("has_genes=FALSE"),
        paste0("has_common_dispersion=FALSE"),
        paste0("has_trended_dispersion=FALSE"),
        paste0("has_tagwise_dispersion=FALSE"),
        paste0("has_AveLogCPM=FALSE"),
        paste0("common.dispersion=NA"),
        paste0("prior.df=NA"),
        paste0("has_size_factors=FALSE")
    )
    writeLines(metadata, file.path(tmpdir, "metadata.txt"))

}} else {{
    stop(paste0("Unsupported R object class: ", cls,
                ". Expected DGEList, SummarizedExperiment, DESeqDataSet, matrix, or data.frame."))
}}
'''


def _read_rds(filepath, group=None, verbose=True):
    """Read an R .rds file containing a DGEList, SummarizedExperiment, or DESeqDataSet.

    Uses R (via subprocess) to extract components to temporary CSV files,
    then loads them into a DGEList. Requires R to be installed and
    accessible as 'Rscript' on PATH.
    """
    import subprocess
    import shutil
    import tempfile

    from .dgelist import make_dgelist

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"RDS file not found: {filepath}")

    rscript = shutil.which('Rscript')
    if rscript is None:
        raise RuntimeError(
            "Rscript not found on PATH. R must be installed to read .rds files. "
            "Install R from https://cran.r-project.org/"
        )

    tmpdir = tempfile.mkdtemp(prefix='edgepy_rds_')

    try:
        r_script = _build_rds_extraction_script(os.path.abspath(filepath), tmpdir)
        script_path = os.path.join(tmpdir, 'extract.R')
        with open(script_path, 'w') as f:
            f.write(r_script)

        if verbose:
            print(f"Reading {os.path.basename(filepath)} via R...")

        result = subprocess.run(
            [rscript, '--no-save', '--no-restore', script_path],
            capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            err_msg = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"R failed to read {filepath}:\n{err_msg}")

        # Parse metadata
        metadata = _parse_rds_metadata(os.path.join(tmpdir, 'metadata.txt'))

        if verbose:
            cls = metadata.get('class', 'unknown')
            nrow = metadata.get('nrow', '?')
            ncol = metadata.get('ncol', '?')
            print(f"  {cls}: {nrow} genes x {ncol} samples")

        # Load counts
        counts_path = os.path.join(tmpdir, 'counts.csv')
        if not os.path.exists(counts_path):
            raise RuntimeError("R extraction did not produce counts.csv")
        counts_df = pd.read_csv(counts_path, index_col=0)
        gene_ids = list(counts_df.index.astype(str))
        sample_names = list(counts_df.columns)
        counts = counts_df.values.astype(np.float64)

        # Load sample info
        samples_path = os.path.join(tmpdir, 'samples.csv')
        samples_df = pd.read_csv(samples_path, index_col=0) if os.path.exists(samples_path) else None

        r_group = None
        lib_size = None
        norm_factors = None

        if samples_df is not None:
            if 'group' in samples_df.columns:
                r_group = samples_df['group'].values
            if 'lib.size' in samples_df.columns:
                lib_size = samples_df['lib.size'].values.astype(np.float64)
            if 'norm.factors' in samples_df.columns:
                norm_factors = samples_df['norm.factors'].values.astype(np.float64)

        if group is None:
            group = r_group

        # Load gene annotation
        genes_path = os.path.join(tmpdir, 'genes.csv')
        if os.path.exists(genes_path):
            genes_df = pd.read_csv(genes_path, index_col=0)
        else:
            # Create minimal genes DataFrame to preserve row names
            genes_df = pd.DataFrame(index=gene_ids)

        # Build DGEList
        dge = make_dgelist(
            counts, lib_size=lib_size, norm_factors=norm_factors,
            group=group, genes=genes_df,
        )

        # Restore original row/column names
        if gene_ids:
            if 'genes' in dge and dge['genes'] is not None:
                dge['genes'].index = gene_ids
        if sample_names:
            dge['samples'].index = sample_names

        # Restore dispersions
        if metadata.get('has_common_dispersion') == 'TRUE':
            val = metadata.get('common.dispersion')
            if val is not None:
                dge['common.dispersion'] = float(val)

        disp_path = os.path.join(tmpdir, 'dispersions.csv')
        if os.path.exists(disp_path):
            disp_df = pd.read_csv(disp_path, index_col=0)
            if 'trended' in disp_df.columns:
                dge['trended.dispersion'] = disp_df['trended'].values.astype(np.float64)
            if 'tagwise' in disp_df.columns:
                dge['tagwise.dispersion'] = disp_df['tagwise'].values.astype(np.float64)

        alc_path = os.path.join(tmpdir, 'AveLogCPM.csv')
        if metadata.get('has_AveLogCPM') == 'TRUE' and os.path.exists(alc_path):
            alc_df = pd.read_csv(alc_path, index_col=0)
            dge['AveLogCPM'] = alc_df['value'].values.astype(np.float64)

        if metadata.get('prior.df') is not None:
            dge['prior.df'] = float(metadata['prior.df'])

        # DESeqDataSet size factors
        sf_path = os.path.join(tmpdir, 'size_factors.csv')
        if metadata.get('has_size_factors') == 'TRUE' and os.path.exists(sf_path):
            sf_df = pd.read_csv(sf_path, index_col=0)
            dge['deseq2.size.factors'] = sf_df['value'].values.astype(np.float64)

        return dge

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"R subprocess timed out reading {filepath}. "
            "The file may be very large or R may be unresponsive."
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _read_table_file(data, path, columns, sep, group, verbose):
    """Read CSV/TSV count table (e.g., exported from R).

    Handles two formats:
    1. Single file with gene IDs as first column/index and samples as columns
    2. List of per-sample files (delegates to read_dge)
    """
    from .dgelist import make_dgelist

    files = [data] if isinstance(data, str) else list(data)

    # Single file: try reading as a count matrix (R-style export)
    if len(files) == 1:
        f = files[0]
        fp = os.path.join(path, f) if path else f

        # Detect separator from extension if not specified
        actual_sep = sep
        if fp.endswith('.csv'):
            actual_sep = ','

        if verbose:
            print(f"Reading {os.path.basename(fp)}...")

        df = pd.read_csv(fp, sep=actual_sep, index_col=0)

        # Check if this looks like a count matrix (all numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == len(df.columns):
            # All columns are numeric — treat as gene×sample count matrix
            counts = df.values.astype(np.float64)
            genes_df = pd.DataFrame(index=df.index)
            dge = make_dgelist(counts, group=group, genes=genes_df)
            dge['samples'].index = list(df.columns)
            return dge

    # Multiple files or non-numeric single file: delegate to read_dge
    return read_dge(files, path=path, columns=columns or (0, 1),
                    group=group, sep=sep)


def _auto_detect_source(data, fmt):
    """Detect data source from data argument type and file structure."""
    if isinstance(data, np.ndarray):
        return 'matrix'
    if isinstance(data, pd.DataFrame):
        return 'dataframe'
    # scipy.sparse matrices
    try:
        import scipy.sparse as sp
        if sp.issparse(data):
            return 'sparse'
    except ImportError:
        pass
    if isinstance(data, str):
        if data.endswith('.h5ad'):
            return 'anndata'
        if data.lower().endswith('.rds'):
            return 'rds'
        if os.path.isdir(data):
            try:
                contents = os.listdir(data)
            except OSError:
                raise ValueError(f"Cannot list directory: {data}")
            if 'matrix.mtx' in contents or 'matrix.mtx.gz' in contents:
                return '10x'
            if 'quant.sf' in contents:
                return 'salmon'
            if 'abundance.tsv' in contents or 'abundance.h5' in contents:
                return 'kallisto'
            if any(f.endswith('.quant') for f in contents):
                return 'oarfish'
        elif os.path.isfile(data):
            if data.endswith('.isoforms.results'):
                return 'rsem'
            return 'table'
        raise ValueError(f"Cannot auto-detect source from path: {data}")

    if isinstance(data, (list, tuple)) and len(data) > 0:
        first = str(data[0])
        if os.path.isdir(first):
            try:
                contents = os.listdir(first)
            except OSError:
                raise ValueError(f"Cannot list directory: {first}")
            if 'quant.sf' in contents:
                return 'salmon'
            if 'abundance.tsv' in contents or 'abundance.h5' in contents:
                return 'kallisto'
        elif os.path.isfile(first):
            if first.endswith('.isoforms.results'):
                return 'rsem'
            if first.endswith('.quant'):
                return 'oarfish'
            return 'table'

    raise ValueError(
        "Cannot auto-detect data source. Please specify source='kallisto', "
        "'salmon', 'oarfish', 'rsem', 'anndata', '10x', 'table', or 'matrix'."
    )


def _get_anndata_type():
    """Return anndata.AnnData class without hard import."""
    try:
        import anndata
        return anndata.AnnData
    except ImportError:
        return None


# =====================================================================
# Universal read_data() function
# =====================================================================

def read_data(
    data,
    *,
    source=None,
    format=None,
    path=None,
    group=None,
    labels=None,
    columns=None,
    sep='\t',
    obs_col=None,
    layer=None,
    ngibbs=100,
    verbose=True,
):
    """Universal data import for edgePython.

    Reads count data from various sources and returns a DGEList.
    When bootstrap/Gibbs resampling information is available, computes
    overdispersion estimates following edgeR's algorithm.

    Parameters
    ----------
    data : various
        Input data. Accepts:
        - list of str: paths to quantification directories
          (kallisto/salmon) or files (RSEM/oarfish/table)
        - str: path to .h5ad file, .rds file (DGEList/SummarizedExperiment/
          DESeqDataSet), 10X directory, quantification directory, or
          count table (.csv/.tsv/.txt)
        - AnnData object: in-memory AnnData
        - DGEList: returned as-is (pass-through)
        - ndarray: count matrix (genes x samples)
        - scipy.sparse matrix (CSR/CSC): sparse count matrix
          (will be densified with a warning)
        - DataFrame: count matrix with gene names as index
    source : str or None
        Data source. Auto-detected if None.
        One of: 'kallisto', 'salmon', 'oarfish', 'rsem', '10x',
        'table', 'anndata', 'rds', 'sparse', 'matrix', 'dataframe'.
    format : str or None
        For kallisto: 'h5' or 'tsv'. If None, prefers H5 when available.
    path : str or None
        Base path prefix for relative file paths.
    group : array-like or None
        Sample group assignments.
    labels : list of str or None
        Sample labels. Auto-generated from directory/file names if None.
    columns : tuple of int or None
        For table format: (gene_id_col, count_col) as 0-based indices.
    sep : str
        Field separator for table/CSV files.
    obs_col : str or None
        For AnnData: column in .obs to use as group factor.
    layer : str or None
        For AnnData: layer name to use instead of .X.
    ngibbs : int or array-like
        For RSEM: number of Gibbs samples per sample.
    verbose : bool
        Print progress messages.

    Returns
    -------
    DGEList
        With keys: counts, samples, genes.
        When bootstraps are available: genes DataFrame includes
        'Overdispersion' column and dge['overdispersion.prior'] is set.
    """
    from .dgelist import make_dgelist
    from .classes import DGEList

    # --- Pass-through ---
    if isinstance(data, DGEList):
        return data
    if isinstance(data, dict) and 'counts' in data:
        return data

    # --- AnnData (in-memory object) ---
    _anndata_cls = _get_anndata_type()
    if _anndata_cls is not None and isinstance(data, _anndata_cls):
        source = 'anndata'

    # --- Auto-detect ---
    if source is None:
        source = _auto_detect_source(data, format)

    # --- Dispatch ---
    if source == 'anndata':
        return _read_anndata(data, group=group, labels=labels,
                             obs_col=obs_col, layer=layer, verbose=verbose)

    if source == 'rds':
        return _read_rds(data, group=group, verbose=verbose)

    if source == '10x':
        p = data if isinstance(data, str) else path
        return read_10x(p, as_dgelist=True)

    if source == 'sparse':
        shape = data.shape
        nnz = data.nnz
        density = nnz / (shape[0] * shape[1]) if shape[0] * shape[1] > 0 else 0
        warnings.warn(
            f"Densifying sparse matrix ({shape[0]} x {shape[1]}, "
            f"{100*density:.1f}% non-zero, "
            f"{shape[0] * shape[1] * 8 / 1e6:.0f} MB dense). "
            f"edgePython stores counts as dense arrays.",
            stacklevel=2,
        )
        counts = np.asarray(data.toarray(), dtype=np.float64)
        return make_dgelist(counts, group=group)

    if source == 'matrix':
        counts = np.asarray(data, dtype=np.float64)
        if counts.ndim == 1:
            counts = counts.reshape(-1, 1)
        return make_dgelist(counts, group=group)

    if source == 'dataframe':
        genes_df = pd.DataFrame(index=data.index)
        counts = data.values.astype(np.float64)
        dge = make_dgelist(counts, group=group, genes=genes_df)
        dge['samples'].index = list(data.columns)
        return dge

    if source == 'table':
        return _read_table_file(data, path=path, columns=columns,
                                sep=sep, group=group, verbose=verbose)

    # --- Quantification tools: data is path(s) ---
    if isinstance(data, str):
        paths = [data]
    elif isinstance(data, (list, tuple)):
        paths = list(data)
    else:
        raise ValueError(f"Expected path or list of paths for source='{source}'")

    if path is not None:
        paths = [os.path.join(path, p) for p in paths]

    if labels is None:
        labels = [os.path.basename(os.path.normpath(p)) for p in paths]

    overdisp_prior = None
    if source == 'kallisto':
        counts, annotation, ids, overdisp_prior = _read_kallisto(paths, format, verbose)
    elif source == 'salmon':
        counts, annotation, ids, overdisp_prior = _read_salmon(paths, verbose)
    elif source == 'oarfish':
        counts, annotation, ids, overdisp_prior = _read_oarfish(
            paths if isinstance(data, (list, tuple)) else None,
            path=data if isinstance(data, str) and os.path.isdir(data) else None,
            verbose=verbose)
    elif source == 'rsem':
        counts, annotation, ids, overdisp_prior = _read_rsem_data(
            paths, path=None, ngibbs=ngibbs, verbose=verbose)
    else:
        raise ValueError(f"Unknown source: {source!r}")

    dge = make_dgelist(counts, group=group, genes=annotation)

    # Restore transcript IDs as genes index (make_dgelist overwrites with numeric)
    if ids is not None and 'genes' in dge:
        dge['genes'].index = ids

    if overdisp_prior is not None and not np.isnan(overdisp_prior):
        dge['overdispersion.prior'] = overdisp_prior

    if labels is not None:
        dge['samples'].index = labels

    return dge


# =====================================================================
# Bismark methylation coverage
# =====================================================================

def read_bismark2dge(files, sample_names=None, verbose=True):
    """Read Bismark methylation coverage files into a DGEList.

    Port of edgeR's readBismark2DGE.

    Reads Bismark ``.cov`` coverage files and collates them into a single
    DGEList with two columns per sample (methylated and unmethylated
    counts).  Column ordering is interleaved:
    ``Sample1-Me, Sample1-Un, Sample2-Me, Sample2-Un, ...``

    Parameters
    ----------
    files : list of str
        Paths to Bismark coverage files.  Each file is tab-delimited with
        six columns: chr, start, end, methylation%, count_methylated,
        count_unmethylated.
    sample_names : list of str, optional
        Sample names.  If None, derived from file names (extensions stripped).
    verbose : bool
        Print progress messages.

    Returns
    -------
    DGEList
        With ``2 * nsamples`` columns and a ``genes`` DataFrame containing
        ``Chr`` and ``Locus`` columns.
    """
    from .dgelist import make_dgelist

    files = [str(f) for f in files]
    nsamples = len(files)

    if sample_names is None:
        sample_names = []
        for f in files:
            name = os.path.basename(f)
            # Strip up to 3 extensions (matching R's removeExt×3)
            for _ in range(3):
                root, ext = os.path.splitext(name)
                if ext:
                    name = root
                else:
                    break
            sample_names.append(name)

    # Read all files, collecting chromosome names and loci
    chr_rle_list = []
    locus_list = []
    count_list = []
    chr_names = []

    for i, f in enumerate(files):
        if verbose:
            print(f"Reading {f}")
        x = pd.read_csv(f, sep='\t', header=None)
        # Columns: 0=chr, 1=start, 2=end, 3=meth%, 4=Me, 5=Un
        chrs = x.iloc[:, 0].values.astype(str)
        loci = x.iloc[:, 1].values.astype(np.int64)
        me_un = x.iloc[:, [4, 5]].values.astype(np.int64)

        # Collect unique chromosome names in order of appearance
        for c in chrs:
            if c not in chr_names:
                chr_names.append(c)

        chr_rle_list.append(chrs)
        locus_list.append(loci)
        count_list.append(me_un)

    if verbose:
        print("Hashing ...")

    # Map chromosome names to integers
    chr_to_int = {c: i + 1 for i, c in enumerate(chr_names)}
    hash_base = len(chr_names) + 1

    # Hash genomic positions: chr_int / hash_base + locus
    hash_list = []
    hash_unique = []
    hash_set = set()
    for i in range(nsamples):
        chr_ints = np.array([chr_to_int[c] for c in chr_rle_list[i]],
                            dtype=np.float64)
        h = chr_ints / hash_base + locus_list[i].astype(np.float64)
        hash_list.append(h)
        for v in h:
            if v not in hash_set:
                hash_unique.append(v)
                hash_set.add(v)

    hash_unique = np.array(hash_unique, dtype=np.float64)
    n_loci = len(hash_unique)

    if verbose:
        print("Collating counts ...")

    # Build merged count matrix with interleaved columns
    # Column order: S1-Me, S1-Un, S2-Me, S2-Un, ...
    counts = np.zeros((n_loci, nsamples * 2), dtype=np.int64)
    hash_to_row = {v: idx for idx, v in enumerate(hash_unique)}

    for i in range(nsamples):
        h = hash_list[i]
        rows = np.array([hash_to_row[v] for v in h], dtype=np.int64)
        counts[rows, 2 * i] = count_list[i][:, 0]      # Me
        counts[rows, 2 * i + 1] = count_list[i][:, 1]   # Un

    # Unhash: recover chromosome and locus
    locus_arr = hash_unique.astype(np.int64)
    chr_int_arr = np.round((hash_unique - locus_arr) * hash_base).astype(int)
    chr_arr = np.array([chr_names[ci - 1] for ci in chr_int_arr])

    # Column names: interleaved
    col_names = []
    for sn in sample_names:
        col_names.append(f"{sn}-Me")
        col_names.append(f"{sn}-Un")

    # Row names
    row_names = [f"{c}-{l}" for c, l in zip(chr_arr, locus_arr)]

    # Gene annotation
    genes_df = pd.DataFrame({'Chr': chr_arr, 'Locus': locus_arr},
                             index=row_names)

    # Build DGEList
    counts_float = counts.astype(np.float64)
    y = make_dgelist(counts_float, genes=genes_df)
    y['samples'].index = col_names
    # Set row names on the DGEList
    if 'genes' in y and y['genes'] is not None:
        y['genes'].index = row_names

    return y


# =====================================================================
# Pseudo-bulk aggregation (Seurat2PB)
# =====================================================================

def seurat_to_pb(object, sample, cluster="cluster"):
    """Convert single-cell data to pseudo-bulk DGEList.

    Port of edgeR's Seurat2PB. Aggregates raw counts of cells sharing
    the same sample and cluster identity into pseudo-bulk columns.

    In R, this function takes a Seurat object. In Python, this function
    accepts an AnnData object (the standard Python single-cell container),
    a dict with 'counts' and 'obs' keys, or a raw count matrix with
    separate metadata.

    Parameters
    ----------
    object : AnnData, dict, or ndarray
        Single-cell data. Accepted formats:

        - **AnnData**: Uses ``.X`` or ``.layers['counts']`` for the
          count matrix (obs x var = cells x genes). The ``sample``
          and ``cluster`` columns must be present in ``.obs``.
        - **dict**: Must contain ``'counts'`` (genes x cells ndarray)
          and ``'obs'`` (DataFrame with sample/cluster columns).
        - **ndarray**: Raw count matrix (genes x cells). In this case,
          ``sample`` must be an array-like of per-cell sample labels
          and ``cluster`` must be an array-like of per-cell cluster
          labels.
    sample : str or array-like
        If str, the column name in ``.obs`` (AnnData) or ``obs``
        (dict) identifying the biological sample each cell belongs to.
        If array-like, per-cell sample labels (length = n_cells).
    cluster : str or array-like
        If str, the column name in ``.obs`` or ``obs`` identifying
        the cell cluster. Default ``"cluster"``.
        If array-like, per-cell cluster labels.

    Returns
    -------
    DGEList
        Pseudo-bulk DGEList with one column per sample-cluster
        combination. The ``samples`` DataFrame contains ``sample``
        and ``cluster`` columns.
    """
    from .dgelist import make_dgelist

    # --- Extract counts and metadata ---
    counts = None
    obs = None

    # Check for AnnData
    _anndata_cls = _get_anndata_type()
    if _anndata_cls is not None and isinstance(object, _anndata_cls):
        # AnnData: obs x var (cells x genes) -> need to transpose
        if 'counts' in object.layers:
            X = object.layers['counts']
        else:
            X = object.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        counts = np.asarray(X, dtype=np.float64).T  # genes x cells
        obs = object.obs
        gene_names = list(object.var_names)
        gene_info = object.var.copy() if len(object.var.columns) > 0 else None
    elif isinstance(object, dict):
        counts = np.asarray(object['counts'], dtype=np.float64)
        obs = object.get('obs')
        gene_names = None
        gene_info = object.get('genes')
    elif isinstance(object, np.ndarray):
        counts = np.asarray(object, dtype=np.float64)
        obs = None
        gene_names = None
        gene_info = None
    else:
        raise TypeError(
            f"Expected AnnData, dict, or ndarray, got {type(object).__name__}"
        )

    n_genes, n_cells = counts.shape

    # --- Get sample and cluster labels ---
    if isinstance(sample, str):
        if obs is None:
            raise ValueError(
                "sample is a column name but no obs/metadata provided"
            )
        if sample not in obs.columns:
            raise ValueError(
                f"Column '{sample}' not found in obs. "
                f"Available: {list(obs.columns)}"
            )
        sample_labels = obs[sample].values
    else:
        sample_labels = np.asarray(sample)
        if len(sample_labels) != n_cells:
            raise ValueError(
                f"sample length ({len(sample_labels)}) != "
                f"number of cells ({n_cells})"
            )

    if isinstance(cluster, str):
        if obs is None:
            raise ValueError(
                "cluster is a column name but no obs/metadata provided"
            )
        if cluster not in obs.columns:
            raise ValueError(
                f"Column '{cluster}' not found in obs. "
                f"Available: {list(obs.columns)}"
            )
        cluster_labels = obs[cluster].values
    else:
        cluster_labels = np.asarray(cluster)
        if len(cluster_labels) != n_cells:
            raise ValueError(
                f"cluster length ({len(cluster_labels)}) != "
                f"number of cells ({n_cells})"
            )

    # --- Create combined sample_cluster factor ---
    sample_labels = np.asarray(sample_labels, dtype=str)
    cluster_labels = np.asarray(cluster_labels, dtype=str)
    combined = np.array([
        f"{s}_cluster{c}" for s, c in zip(sample_labels, cluster_labels)
    ])

    # Get unique groups preserving order of appearance
    seen = {}
    unique_groups = []
    for g in combined:
        if g not in seen:
            seen[g] = len(unique_groups)
            unique_groups.append(g)

    n_groups = len(unique_groups)

    # --- Aggregate counts by matrix multiplication (matching R) ---
    # Build indicator matrix: n_cells x n_groups
    group_idx = np.array([seen[g] for g in combined])
    indicator = np.zeros((n_cells, n_groups), dtype=np.float64)
    indicator[np.arange(n_cells), group_idx] = 1.0

    # counts (genes x cells) @ indicator (cells x groups) = pb (genes x groups)
    counts_pb = counts @ indicator

    # --- Build sample metadata ---
    sample_pb = []
    cluster_pb = []
    for g in unique_groups:
        # Parse "samplename_clusterX" back into sample and cluster
        idx = g.index("_cluster")
        sample_pb.append(g[:idx])
        cluster_pb.append(g[idx + 8:])  # len("_cluster") == 8

    samples_df = pd.DataFrame({
        'sample': sample_pb,
        'cluster': cluster_pb,
    }, index=unique_groups)

    # --- Build gene annotation ---
    genes_df = None
    if gene_info is not None:
        genes_df = gene_info.copy()
        if 'gene' not in genes_df.columns and gene_names is not None:
            genes_df.insert(0, 'gene', gene_names)
    elif gene_names is not None:
        genes_df = pd.DataFrame({'gene': gene_names})

    # --- Create DGEList ---
    dge = make_dgelist(counts_pb, genes=genes_df)
    # Set sample metadata
    dge['samples']['sample'] = samples_df['sample'].values
    dge['samples']['cluster'] = samples_df['cluster'].values
    dge['samples'].index = unique_groups

    return dge


# =====================================================================
# AnnData export
# =====================================================================

def to_anndata(obj, adata=None):
    """Convert edgePython results to AnnData format.

    Stores results in a predictable schema compatible with the Scanpy
    ecosystem.  Can either create a new AnnData or update an existing
    one in-place.

    Schema
    ------
    .X : ndarray
        Raw counts (samples x genes, transposed from edgePython layout).
    .layers['counts'] : ndarray
        Copy of raw counts in the standard Scanpy layer.
    .obs : DataFrame
        Sample metadata: group, lib_size, norm_factors.
    .var : DataFrame
        Gene metadata from genes DataFrame, plus per-gene DE results
        (logFC, logCPM, PValue, FDR, F/LR) and dispersions.
    .varm['edgepython_coefficients'] : ndarray
        GLM coefficient matrix (genes x coefficients) when available.
    .uns['edgepython'] : dict
        Global results: common_dispersion, prior_df, design matrix,
        test method, contrast info, overdispersion prior.

    Parameters
    ----------
    obj : DGEList, DGELRT, DGEExact, TopTags, or DGEGLM
        edgePython result object.
    adata : AnnData or None
        Existing AnnData to update. If None, creates a new one.

    Returns
    -------
    AnnData
    """
    try:
        import anndata
    except ImportError:
        raise ImportError(
            "anndata package required for AnnData export. "
            "Install with: pip install anndata"
        )

    from .classes import DGEList, DGEExact, DGEGLM, DGELRT, TopTags

    # --- Extract counts if available ---
    counts = None
    if 'counts' in obj and obj['counts'] is not None:
        counts = obj['counts']

    n_genes = None
    if counts is not None:
        n_genes, n_samples = counts.shape
    elif 'coefficients' in obj and obj['coefficients'] is not None:
        n_genes = obj['coefficients'].shape[0]
    elif 'table' in obj and obj['table'] is not None:
        n_genes = len(obj['table'])

    # --- Build or update AnnData ---
    if adata is None:
        if counts is not None:
            # Create new with counts: AnnData is obs×var (samples×genes)
            adata = anndata.AnnData(
                X=counts.T.copy(),
                dtype=np.float64,
            )
            adata.layers['counts'] = counts.T.copy()
        else:
            # No counts (e.g., TopTags or DGELRT without counts)
            # Create minimal AnnData from table or coefficients
            if 'table' in obj and obj['table'] is not None:
                table = obj['table']
                n_vars = len(table)
                n_obs = obj['samples'].shape[0] if 'samples' in obj else 1
                adata = anndata.AnnData(
                    X=np.zeros((n_obs, n_vars), dtype=np.float64),
                    dtype=np.float64,
                )
            else:
                raise ValueError(
                    "No counts or results table found in object. "
                    "Pass a DGEList, DGELRT, DGEExact, or TopTags."
                )
    else:
        # Update existing — just add results, don't touch .X
        pass

    # --- .obs: sample metadata ---
    if 'samples' in obj and obj['samples'] is not None:
        sam = obj['samples']
        adata.obs_names = list(sam.index)
        if 'group' in sam.columns:
            adata.obs['group'] = sam['group'].values
        if 'lib.size' in sam.columns:
            adata.obs['lib_size'] = sam['lib.size'].values
        if 'norm.factors' in sam.columns:
            adata.obs['norm_factors'] = sam['norm.factors'].values

    # --- .var: gene metadata ---
    if 'genes' in obj and obj['genes'] is not None:
        genes_df = obj['genes']
        adata.var_names = list(genes_df.index)
        for col in genes_df.columns:
            adata.var[col] = genes_df[col].values

    # --- .var: DE test results ---
    table = None
    if 'table' in obj and obj['table'] is not None:
        t = obj['table']
        if isinstance(t, pd.DataFrame) and len(t) > 0:
            table = t

    if table is not None:
        n_var = adata.shape[1]
        if len(table) == n_var:
            # Full table — assign directly by position
            for col in table.columns:
                adata.var[col] = table[col].values
        else:
            # Partial (e.g., top n genes) — fill NaN first
            for col in table.columns:
                adata.var[col] = np.nan
                adata.var.loc[table.index, col] = table[col].values

    # --- .var: dispersions ---
    n_var = adata.shape[1]
    if 'tagwise.dispersion' in obj and obj['tagwise.dispersion'] is not None:
        v = obj['tagwise.dispersion']
        if hasattr(v, '__len__') and len(v) == n_var:
            adata.var['tagwise_dispersion'] = v
    if 'trended.dispersion' in obj and obj['trended.dispersion'] is not None:
        v = obj['trended.dispersion']
        if hasattr(v, '__len__') and len(v) == n_var:
            adata.var['trended_dispersion'] = v
    if 'dispersion' in obj and obj['dispersion'] is not None:
        v = obj['dispersion']
        if hasattr(v, '__len__') and len(v) == n_var:
            adata.var['dispersion'] = v
    if 'AveLogCPM' in obj and obj['AveLogCPM'] is not None:
        v = obj['AveLogCPM']
        n_var = adata.shape[1]
        if hasattr(v, '__len__') and len(v) == n_var:
            adata.var['AveLogCPM'] = v

    # --- .varm: GLM coefficients ---
    if 'coefficients' in obj and obj['coefficients'] is not None:
        coefs = obj['coefficients']
        n_var = adata.shape[1]
        if isinstance(coefs, np.ndarray) and coefs.shape[0] == n_var:
            adata.varm['edgepython_coefficients'] = coefs

    # --- .uns: global / scalar results ---
    uns = {}
    if 'common.dispersion' in obj and obj['common.dispersion'] is not None:
        uns['common_dispersion'] = float(obj['common.dispersion'])
    if 'method' in obj and obj['method'] is not None:
        uns['method'] = obj['method']
    if 'prior.df' in obj and obj['prior.df'] is not None:
        uns['prior_df'] = float(obj['prior.df'])
    if 'df.prior' in obj and obj['df.prior'] is not None:
        v = obj['df.prior']
        uns['df_prior'] = float(v) if np.isscalar(v) else v.tolist()
    if 'design' in obj and obj['design'] is not None:
        uns['design'] = obj['design'].tolist()
    if 'comparison' in obj:
        uns['comparison'] = obj['comparison']
    if 'test' in obj:
        uns['test'] = obj['test']
    if 'adjust.method' in obj:
        uns['adjust_method'] = obj['adjust.method']
    if 'overdispersion.prior' in obj:
        uns['overdispersion_prior'] = float(obj['overdispersion.prior'])

    if uns:
        adata.uns['edgepython'] = uns

    return adata
