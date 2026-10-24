"""edgeR-parity fixes: one test per fix, plus a norm_lib_sizes DGEList test.

Reference values come from edgeR 4.99.10 (R 4.5.2, limma 3.99.2), computed by
data/edger5_fixes.R from the same seeded inputs this file builds with numpy.
They are hard-coded here so the tests need no R at runtime. Each numerical test
fails on the clean base 5e07dde (v0.5.0); fixes 4, 10 and 12 are behaviour-only.
"""

import numpy as np
import pytest

import edgepython as ep
from edgepython.utils import binom_test
from edgepython.smoothing import locfit_by_col
from edgepython.exact_test import q2q_nbinom


# ===== seeded inputs (numpy default_rng; the R script embeds the same arrays) =====

# Fix 1: low-count table -> ties in the TMMwsp A statistic.
rng = np.random.default_rng(11)
COUNTS1 = rng.poisson(1.5, (40, 6)).astype(np.float64)

# Fix 2: 30 x 10 at logratio_trim=0.30 -> a genuinely restrictive trim window.
rng = np.random.default_rng(22)
_mu2 = np.outer(np.geomspace(2, 200, 30), np.ones(10))
COUNTS2 = rng.poisson(_mu2).astype(np.float64)

# Fix 3: 60 x 10 with a short lib.size to recycle.
rng = np.random.default_rng(33)
_mu3 = np.outer(np.geomspace(1, 50, 60), np.ones(10))
COUNTS3 = rng.poisson(_mu3).astype(np.float64)
LIB_SIZE3 = np.array([1000.0, 2000.0])

# Fix 5: an all-zero count matrix.
COUNTS5 = np.zeros((7, 4), dtype=np.float64)

# Fix 6: 24 x 6 with a tagwise dispersion present alongside a common one.
rng = np.random.default_rng(66)
_mu6 = np.outer(np.geomspace(2, 100, 24), np.ones(6))
COUNTS6 = rng.poisson(_mu6).astype(np.float64)
GROUP6 = np.array([1, 1, 1, 2, 2, 2])
TAGWISE6 = 0.1 + 0.4 * np.linspace(0.0, 1.0, 24) + 0.05 * (np.arange(24) % 3)

# Fix 7: binomTest chi-squared branch (size > 10000).
Y1_7 = np.array([12000, 8000, 15000, 9000, 11000], dtype=float)
Y2_7 = np.array([13000, 7000, 14000, 8500, 10500], dtype=float)

# Fix 8: binomTest small-probability branch (size 8017, p != 0.5).
Y1_8 = np.array([4360, 4380, 4400, 4420, 4440], dtype=float)
Y2_8 = 8017.0 - Y1_8

# Fix 9: WLEB locfit.mixed, 30 genes x 11 grid; a broad central dip on rows
# 8:17 makes the locfit.mixed blend bimodal so the unimodality loop has rows to fix.
rng = np.random.default_rng(99)
_theta9 = np.linspace(-2.0, 2.0, 11)
_center9 = np.linspace(0.0, 1.0, 30)
_spread9 = 1.0 + 0.5 * np.sin(np.linspace(0, 6, 30))
LOGLIK9 = -(_theta9[None, :] - _center9[:, None]) ** 2 * _spread9[:, None]
_bimodal = np.zeros(30, dtype=bool)
_bimodal[8:17] = True
LOGLIK9[_bimodal] += -4.0 * np.exp(-0.5 * (_theta9 / 0.5) ** 2)[None, :]
LOGLIK9 += rng.normal(0, 0.05, (30, 11))
COVARIATE9 = np.linspace(-1.0, 1.0, 30) + rng.normal(0, 0.1, 30)

# Fix 11: q2q_nbinom far tail at dispersion=0; counts well above the input mean.
_input_mean11 = np.linspace(50.0, 120.0, 20)
X11 = _input_mean11 * np.linspace(1.6, 6.0, 20)
OUTPUT_MEAN11 = _input_mean11 * np.linspace(1.4, 2.2, 20)

# Fix 12: 40 x 6 with a per-gene dispersion vector.
rng = np.random.default_rng(122)
_mu12 = np.outer(np.geomspace(2, 300, 40), np.ones(6))
COUNTS12 = rng.poisson(_mu12).astype(np.float64)
GROUP12 = np.array([1, 1, 1, 2, 2, 2])
DISP12 = 0.05 + 0.1 * np.linspace(0, 1, 40)


# ===== edgeR 4.99.10 reference values (hard-coded; from data/edger5_fixes.R) =====

REF_TMMWSP_TIES = np.array([0.936354848809833, 1.00660671172035, 1.00333647269179, 0.873732666076212, 1.05725505585048, 1.14470814010234])
REF_TMMWSP_TRIM = np.array([0.997295372564062, 1.02090906555259, 1.02118475847587, 0.995710390858498, 1.02598450745303, 0.937309972657714, 1.0205168544119, 0.975496849220365, 1.00122062944915, 1.0077478714099])
REF_TMM_RECYCLE = np.array([1.38822750644795, 0.75378588409871, 1.4881389639392, 0.704116578485063, 1.41588850225083, 0.70291631747863, 1.42825315737936, 0.692474835916482, 1.30944693243486, 0.707576702244474])
REF_AVELOGCPM_ZERO = 17.1242136472666
REF_CPM_TAGWISE = np.array(
[[4955.53822049459, 1565.93199673431],
    [4402.92644887076, 5207.98278299713],
    [2749.49599817122, 4185.55388037714],
    [9365.46683122308, 7850.22468458026],
    [3846.19140438996, 5775.39797095135],
    [6078.98530367781, 8867.81980324906],
    [6618.02051000473, 7833.54519599552],
    [12660.9643394327, 8870.57674668696],
    [13753.2219250474, 9336.84960343471],
    [9861.43418346611, 16746.2857054593],
    [11040.415445675, 17147.6340399113],
    [21506.9667474198, 21322.3965425477],
    [21407.2255578932, 23420.5757787989],
    [27586.8319068498, 31322.3201858528],
    [20414.8090246539, 31256.1775854611],
    [40141.7591419365, 42748.6422722711],
    [48993.5819665449, 50530.1855951939],
    [46345.607317468, 57077.0226777514],
    [77970.1578742722, 71342.864201161],
    [82078.0156909363, 80087.1143009125],
    [103960.076320732, 93209.1368143936],
    [124802.643163141, 115791.152398264],
    [140470.047065227, 133222.577629646],
    [158953.876049237, 155285.445485009]]
)
REF_BINOM_CHISQ = np.array([0.000211886004610521, 2.81220545550063e-30, 1.38251701210346e-27, 2.6430460544597e-12, 1.56703219156398e-12])
REF_BINOM_SMALLP = np.array([9.32221055328002e-16, 2.19163375315179e-17, 4.22352522881727e-19, 6.66887843955343e-21, 8.62426674326123e-23])
REF_WLEB_MIXED = np.array(
[[-6.38501623389184, -4.16849314047791, -2.48466574708268, -1.39788451290223, -0.910047356944764, -0.756429172259334, -0.670048024005655, -0.899962582764949, -1.76962270826431, -3.22413616414088, -5.20315820272822],
    [-6.38516314612905, -4.16841571058025, -2.48443929575807, -1.39745780343981, -0.909349066393166, -0.755620453012276, -0.669613547179051, -0.900065771882176, -1.77019992601506, -3.22514672138491, -5.2046504975458],
    [-6.35633807559305, -4.24298805523753, -2.62804188812525, -1.65000937556151, -1.32353381084331, -1.24108552331058, -0.938641936902967, -0.860690865656289, -1.47121456400261, -2.70169063196161, -4.4352553463269],
    [-6.35638376043012, -4.24286658891421, -2.62780889452871, -1.64960017275709, -1.32286323802679, -1.24029967688547, -0.938206734670095, -0.860754990950406, -1.47169912516607, -2.70253859580564, -4.43650164676643],
    [-6.3406532168936, -4.26409810211623, -2.67364131823457, -1.73194301506373, -1.45761206048425, -1.39752536462296, -1.02429627209565, -0.845294374235765, -1.36925245569409, -2.52332295147634, -4.1726349572881],
    [-6.35987116075581, -4.23087984300159, -2.60548900785966, -1.61064249256021, -1.25899457232289, -1.1653600676549, -0.896573505830909, -0.866519335098154, -1.51717559699183, -2.78213086236155, -4.55342179631776],
    [-6.34491144385001, -4.26073642304447, -2.66521890104919, -1.71643594867157, -1.43227203327896, -1.36809432171839, -1.00837606752313, -0.848711266929753, -1.38952279230233, -2.55877207426579, -4.22491561809313],
    [-6.32520214300602, -4.27058010196596, -2.69465657277312, -1.77196222015849, -1.52288679609844, -1.47287195078484, -1.06438232080917, -0.834742451303374, -1.31358302137971, -2.42599967660376, -4.0288151306454],
    [-6.22442778759184, -4.26425140517608, -2.75064962818733, -1.89535027233451, -1.72344581623551, -1.69910428921916, -1.17743173328071, -0.782674470260743, -1.1033134938569, -2.05809018883966, -3.48194769344193],
    [-6.20633467736299, -4.27486096153215, -2.78046862473994, -1.95175272466345, -1.81685734331728, -1.80783430197378, -1.2371580645765, -0.771200990623111, -1.03000397230079, -1.92871656376583, -3.29103477574847],
    [-6.28707843211843, -4.27074530054308, -2.72007707296419, -1.82571727137809, -1.61010716206859, -1.57173593415938, -1.114354179058, -0.813832394253071, -1.22575198119176, -2.27256142385091, -3.80102214590387],
    [-2.08210538080621, -2.08210538080621, -2.08210538080621, -2.08210538080621, -2.08210538080621, -2.08210538080621, -1.38267433152624, -0.727329873768387, -0.813308528496617, -1.54659934244367, -2.72457377372801],
    [-2.09013334561058, -2.09013334561058, -2.09013334561058, -2.09013334561058, -2.09013334561058, -2.09013334561058, -1.36639684309547, -0.670149038336406, -0.689863496160267, -1.33186998850997, -2.39874213524381],
    [-1.89838181709169, -1.89838181709169, -1.89838181709169, -1.89838181709169, -1.89838181709169, -1.89838181709169, -1.28751064398273, -0.763116563646127, -0.971927078837622, -1.82602781668675, -3.13968874599367],
    [-2.0769336141216, -2.0769336141216, -2.0769336141216, -2.0769336141216, -2.0769336141216, -2.0769336141216, -1.38059112604182, -0.729963420949823, -0.821183029491372, -1.56039431306868, -2.74526823756168],
    [-5.868558056439, -4.16064422591675, -2.81207832478362, -2.08793651132469, -2.01318016032891, -1.9935061126488, -1.28842842714243, -0.615182005867423, -0.620649247678425, -1.21441540993774, -2.21624292400407],
    [-2.08813936658059, -2.08813936658059, -2.08813936658059, -2.08813936658059, -2.08813936658059, -2.08813936658059, -1.36440223587654, -0.667923747809503, -0.686111561733161, -1.32539123691537, -2.38879709827266],
    [-5.84347580850896, -4.14827673320837, -2.80710683135331, -2.08082022093566, -1.99185805101979, -1.9606080827391, -1.2627483611298, -0.59928814937165, -0.603426694300254, -1.18563184794983, -2.17135238732166],
    [-5.7831179838178, -4.1223391569016, -2.80009478599397, -2.06362344483248, -1.92702400772424, -1.85761392059872, -1.18241262691853, -0.550712127175881, -0.552938193533267, -1.10187824329758, -2.04096644531441],
    [-5.85047621567959, -4.1516589332307, -2.80840429332694, -2.08280758137833, -1.99805473743456, -1.97022288053893, -1.27025256015098, -0.603911880489301, -0.608397982734623, -1.19392884272728, -2.18428800894384],
    [-5.73872756324448, -4.10892490379148, -2.80227017065265, -2.05087373288859, -1.85932775341885, -1.74660084262909, -1.09588978488064, -0.499695914833467, -0.502413666242721, -1.01882550522843, -1.91195989593697],
    [-5.73797650408671, -4.10874313686726, -2.80236557972801, -2.05065763782378, -1.85802383092583, -1.74444281504452, -1.09420802509109, -0.498711077000658, -0.501451881454731, -1.01724890459941, -1.9095126707275],
    [-5.68578927024839, -4.09551655103621, -2.80825276289811, -2.03583821409977, -1.77004344983679, -1.598948697174, -0.980745015800708, -0.432053081987921, -0.436019095189484, -0.909905992078597, -1.74287972423359],
    [-5.64484025571636, -4.0808996458656, -2.80744768031795, -2.0246835856922, -1.71709395075059, -1.51277487917576, -0.913335993939766, -0.391544717749833, -0.394672288898065, -0.841626396507877, -1.63675588757902],
    [-5.62966540854753, -4.07359938949861, -2.80441601441265, -2.01879658659414, -1.69924382839581, -1.48548871128082, -0.892702167722392, -0.380081279581015, -0.383966863035264, -0.823975173177335, -1.60920631435795],
    [-5.6249155507739, -4.07126222085926, -2.80335699027529, -2.01669504563577, -1.69314330347242, -1.47624941338126, -0.885814701313628, -0.376416643301741, -0.38076894647828, -0.818721995359647, -1.60100422867049],
    [-5.62607441354104, -4.07183992971885, -2.80362632498254, -2.01721508844848, -1.69462475219119, -1.47848562616731, -0.887477955074523, -0.377296658030007, -0.38153059228752, -0.819973200564958, -1.6029582267177],
    [-5.6133809784671, -4.06489081339202, -2.79989804624342, -2.01170963960332, -1.68109509554644, -1.458611178074, -0.872752145881623, -0.369482420324657, -0.374628806876004, -0.808567371677985, -1.58509988032802],
    [-5.57719905552761, -4.04491658198704, -2.78937946662477, -1.99800236528205, -1.64839762161701, -1.41078755715838, -0.836736175941438, -0.34929132678017, -0.355098234324702, -0.776119401741009, -1.53425798490842],
    [-5.610641267782, -4.06326645460645, -2.79894940793509, -2.01063275873768, -1.67890999174088, -1.45553431034759, -0.870463607600897, -0.368221793490123, -0.373418657420918, -0.806545391265526, -1.58192210391826]]
)
REF_Q2Q_FAR = np.array([105.107912893603, 130.182964258986, 157.932491538048, 188.401164610546, 221.632116691423, 257.667100326174, 296.546614367566, 338.310011553107, 382.995590351757, 430.640674699993, 481.281683886191, 534.954194305932, 591.692994411782, 651.532133898812, 714.504967960598, 780.644197294759, 849.981904417427, 922.549586752217, 998.378186884689, 1077.49812031332])


def _close(actual, expected, rtol=1e-9, atol=1e-12):
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float64),
                               np.asarray(expected, dtype=np.float64),
                               rtol=rtol, atol=atol)


# ===== tests =====

def test_tmmwsp_stable_sort():
    # Fix 1: order(A) is a stable sort; the base's quicksort permuted tied genes.
    _close(ep.calc_norm_factors(COUNTS1, method="TMMwsp"), REF_TMMWSP_TIES)


def test_tmmwsp_trim_window():
    # Fix 2: inclusive trim windows hiM = n + 1 - loM, hiA = n + 1 - loA.
    _close(ep.calc_norm_factors(COUNTS2, method="TMMwsp", logratio_trim=0.30),
           REF_TMMWSP_TRIM)


def test_norm_factors_recycle_lib_size():
    # Fix 3: lib.size <- rep_len(lib.size, nsamples), not the mean.
    _close(ep.calc_norm_factors(COUNTS3, lib_size=LIB_SIZE3, method="TMM"),
           REF_TMM_RECYCLE)


def test_norm_lib_sizes_rejects_negative_count():
    # Fix 4 (behaviour-only): normLibSizes.default stops on a negative count;
    # the deprecated calcNormFactors alias does not.
    counts = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 5.0]])
    with pytest.raises(ValueError, match="Negative counts not allowed"):
        ep.norm_lib_sizes(counts)
    ep.calc_norm_factors(counts)  # the alias accepts it


def test_ave_log_cpm_all_zero():
    # Fix 5: all-zero matrix -> (-log(ntags) + log(1e6)) / log(2), not -1e8.
    _close(ep.ave_log_cpm(COUNTS5), np.full(7, REF_AVELOGCPM_ZERO))


def test_cpm_by_group_tagwise_dispersion():
    # Fix 6: dispersion=None resolves via getDispersion (tagwise first), not
    # common.dispersion alone.
    dge = ep.make_dgelist(counts=COUNTS6, group=GROUP6)
    dge["common.dispersion"] = 0.1
    dge["tagwise.dispersion"] = TAGWISE6
    _close(ep.cpm_by_group(dge), REF_CPM_TAGWISE)


def test_binom_test_chi_squared():
    # Fix 7: chisq.test with its default correct = TRUE (Yates' correction).
    _close(binom_test(Y1_7, Y2_7, n1=50000.0, n2=52000.0), REF_BINOM_CHISQ,
           rtol=1e-8, atol=0)


def test_binom_test_small_probability():
    # Fix 8: order(d) + cumsum with no tolerance; the base's +1e-15 slop
    # inflated p-values below about 1e-15. Pure relative: the p-values reach
    # 1e-23, so any absolute tolerance would mask the base's inflation.
    _close(binom_test(Y1_8, Y2_8, n1=4000.0, n2=4017.0), REF_BINOM_SMALLP,
           rtol=1e-8, atol=0)


def test_wleb_locfit_mixed_unimodality():
    # Fix 9: the unimodality backfill loop, locfit.mixed only (estimateDisp.R:291-297).
    out = ep.WLEB(theta=_theta9, loglik=LOGLIK9, covariate=COVARIATE9,
                  trend_method="locfit.mixed", overall=False, trend=False,
                  individual=False, m0_out=True)
    _close(out["shared.loglik"], REF_WLEB_MIXED, rtol=1e-9, atol=1e-11)


def test_locfit_by_col_degree2_raises():
    # Fix 10 (behaviour-only): local-quadratic tree evaluation is not ported.
    y = np.outer(np.arange(10, dtype=float), np.ones(3))
    with pytest.raises(ValueError, match="degree"):
        locfit_by_col(y, x=np.arange(10, dtype=float), span=0.5, degree=2)
    # degree 0 and 1 still fit.
    locfit_by_col(y, x=np.arange(10, dtype=float), span=0.5, degree=0)


def test_q2q_nbinom_far_tail():
    # Fix 11: the far-tail helpers invert the exact log-tail on the log scale;
    # the base's exp(p) underflowed to 0 and returned inf at dispersion = 0.
    got = q2q_nbinom(X11, _input_mean11, OUTPUT_MEAN11, dispersion=0)
    assert not np.any(np.isinf(got))
    _close(got, REF_Q2Q_FAR, rtol=1e-9, atol=1e-9)


def test_exact_test_dispersion_vector():
    # Fix 12 (behaviour-only): a per-gene dispersion vector is accepted, not
    # "The truth value of an array ... is ambiguous".
    dge = ep.make_dgelist(counts=COUNTS12, group=GROUP12)
    res = ep.exact_test(dge, dispersion=DISP12)
    assert np.asarray(res["table"]["PValue"]).shape[0] == 40


def test_norm_lib_sizes_dgelist():
    # 13th: norm_lib_sizes accepts a DGEList (a dict subclass) and takes counts
    # from x['counts']; it agrees with calc_norm_factors on the same factors.
    dge = ep.make_dgelist(counts=COUNTS3, group=np.ones(10, dtype=int))
    out = ep.norm_lib_sizes(dge)
    ref = ep.calc_norm_factors(dge["counts"], lib_size=dge["samples"]["lib.size"].values,
                               method="TMM")
    _close(out["samples"]["norm.factors"].values, ref)


def test_exact_test_dispersion_series_and_list():
    # A pandas Series or a list of per-gene dispersions behaves like the ndarray.
    import pandas as pd
    dge = ep.make_dgelist(counts=COUNTS12, group=GROUP12)
    ref = np.asarray(ep.exact_test(dge, dispersion=DISP12)["table"]["PValue"])
    for disp in (pd.Series(DISP12), list(DISP12)):
        got = ep.exact_test(dge, dispersion=disp)["table"]["PValue"]
        _close(got, ref, rtol=0, atol=0)


def test_norm_lib_sizes_signature():
    # The wrapper keeps its own name but reports calc_norm_factors' parameters.
    import inspect
    assert ep.norm_lib_sizes.__name__ == "norm_lib_sizes"
    assert (inspect.signature(ep.norm_lib_sizes)
            == inspect.signature(ep.calc_norm_factors))
