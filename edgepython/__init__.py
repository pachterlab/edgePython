# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
edgePython: Python port of the edgeR Bioconductor package.

Empirical analysis of digital gene expression data in Python.
"""

__version__ = "0.5.0"

# --- Classes ---
from .classes import DGEList, DGEExact, DGEGLM, DGELRT, TopTags, PCList, DGEBIN
from .classes import cbind_dgelist, rbind_dgelist
from .compressed_matrix import CompressedMatrix

# --- DGEList construction & accessors ---
from .dgelist import (
    make_dgelist,
    valid_dgelist,
    get_counts,
    get_dispersion,
    get_dispersion_type,
    get_offset,
    get_norm_lib_sizes,
)

# --- Normalization ---
from .normalization import calc_norm_factors, norm_lib_sizes, normalize_chip_to_input, calc_norm_offsets_for_chip

# --- Expression ---
from .expression import cpm, rpkm, tpm, ave_log_cpm, cpm_by_group, rpkm_by_group

# --- Filtering ---
from .filtering import filter_by_expr

# --- Dispersion estimation ---
from .dispersion import (
    estimate_disp,
    WLEB,
    estimate_common_disp,
    estimate_tagwise_disp,
    estimate_trended_disp,
    estimate_glm_common_disp,
    estimate_glm_trended_disp,
    estimate_glm_tagwise_disp,
    estimate_glm_robust_disp,
)

# --- GLM fitting ---
from .glm_fit import glm_fit, glm_ql_fit, mglm_one_group, mglm_one_way
from .ql_weights import sample_weights

# --- GLM testing ---
from .glm_test import glm_lrt, glm_ql_ftest, glm_treat

# --- Exact test ---
from .exact_test import (
    exact_test,
    exact_test_double_tail,
    equalize_lib_sizes,
    q2q_nbinom,
    split_into_groups,
)

# --- Results ---
from .results import top_tags, decide_tests

# --- I/O ---
from .io import (
    read_data,
    to_anndata,
    read_dge,
    read_10x,
    feature_counts_to_dgelist,
    read_bismark2dge,
    seurat_to_pb,
    dgelist_from_tximport,
    dgelist_from_tximeta,
)
from .binomial import (
    make_pclist,
    read_bismark2pc,
    rowsum_pclist,
    bin_fit,
    bin_ql_fit,
    bin_ql_ftest,
    mbin_one_way,
    mbin_iwls,
)
from .quant_readers import (
    catch_salmon,
    catch_salmon_gene,
    catch_kallisto,
    catch_kallisto_gene,
    catch_oarfish,
    catch_oarfish_gene,
    catch_rsem,
    catch_rsem_gene,
    catch_lr_kallisto,
    catch_lr_kallisto_gene,
    split_gencode_tx_names,
    collapse_gencode_tx2gene,
)

# --- Visualization ---
from .visualization import (
    plot_md,
    plot_bcv,
    plot_mds,
    plot_smear,
    plot_ql_disp,
    ma_plot,
    gof,
)

# --- Splicing ---
from .splicing import diff_splice, diff_splice_dge, splice_variants

# --- Gene sets ---
from .gene_sets import camera, fry, roast, mroast, romer, goana, kegga

# --- Utilities ---
from .utils import (
    model_matrix,
    model_matrix_meth,
    nearest_tss,
    add_prior_count,
    pred_fc,
    good_turing,
    thin_counts,
    gini,
    sum_tech_reps,
    zscore_nbinom,
)

# --- Single-cell GLM ---
from .sc_fit import glm_sc_fit, glm_sc_test, shrink_sc_disp

# --- limma utilities ---
from .limma_port import squeeze_var

# --- voom ---
from .voom_lmfit import (
    voom,
    voom_lmfit,
    voom_basic,
    array_weights,
    duplicate_correlation,
)
