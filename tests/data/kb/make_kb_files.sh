#!/bin/sh
# Writes the kallisto and kb count outputs in this directory, read by test_kallisto_readers.py.
# Run from tests/data/kb:  sh make_kb_files.sh
# Made with kallisto 0.52.0, bustools 0.45.1 and kb-python 0.30.2; set KALLISTO
# and BUSTOOLS to the binaries if they are not on PATH.
#   bulk/s1..s3      kallisto quant -b 8 (abundance.h5, run_info.json)
#   t2g.txt          kb ref's transcript-to-gene table
#   sc_std           kb count, standard workflow
#   sc_nac           kb count --workflow nac --filter bustools (unfiltered and filtered)
#   sc_nac_sum       kb count --workflow nac --sum total (unfiltered only)
set -e
KALLISTO=${KALLISTO:-$(command -v kallisto)}
BUSTOOLS=${BUSTOOLS:-$(command -v bustools)}
OUT=$(pwd)
WORK=$(mktemp -d)
python make_kb_sim.py "$WORK"
cd "$WORK"
kb ref --kallisto "$KALLISTO" --bustools "$BUSTOOLS" -i std.idx -g t2g.txt -f1 cdna.fa \
    genome.fa genes.gtf
kb ref --kallisto "$KALLISTO" --bustools "$BUSTOOLS" --workflow nac -i nac.idx -g nac_t2g.txt \
    -c1 c1.txt -c2 c2.txt -f1 nac_cdna.fa -f2 nac_nascent.fa genome.fa genes.gtf
mkdir bulk
for s in 1 2 3; do
    "$KALLISTO" quant -i std.idx -o bulk/s$s -b 8 bulk${s}_R1.fq.gz bulk${s}_R2.fq.gz
done
kbc() { kb count --kallisto "$KALLISTO" --bustools "$BUSTOOLS" -x 10xv3 -w whitelist.txt "$@"; }
kbc -i std.idx -g t2g.txt -o sc_std sc_R1.fq.gz sc_R2.fq.gz
NAC="-i nac.idx -g nac_t2g.txt -c1 c1.txt -c2 c2.txt --workflow nac"
kbc $NAC --filter bustools -o sc_nac sc_R1.fq.gz sc_R2.fq.gz
kbc $NAC --sum total -o sc_nac_sum sc_R1.fq.gz sc_R2.fq.gz

rm -rf "$OUT/bulk" "$OUT/sc_std" "$OUT/sc_nac" "$OUT/sc_nac_sum"
cp t2g.txt "$OUT/"
for s in 1 2 3; do
    mkdir -p "$OUT/bulk/s$s"
    cp bulk/s$s/abundance.h5 bulk/s$s/run_info.json "$OUT/bulk/s$s/"
done
for d in sc_std sc_nac sc_nac_sum; do
    mkdir -p "$OUT/$d"
    cp $d/run_info.json $d/inspect.json $d/kb_info.json "$OUT/$d/"
    cp -r $d/counts_unfiltered "$OUT/$d/"
    [ -d $d/counts_filtered ] && cp -r $d/counts_filtered "$OUT/$d/"
done
# Drop this machine's paths from the recorded command lines and kb_info.json
cd "$OUT"
python - "$WORK" "$KALLISTO" "$BUSTOOLS" "$(command -v kb)" <<'PY'
import glob, os, sys, h5py
work, kallisto, bustools, kb = sys.argv[1:]
subs = []
for path, name in [(kb, "kb"), (kallisto, "kallisto"), (bustools, "bustools"), (work, ".")]:
    # kb records resolved paths (on macOS the temporary directory is under /private)
    for v in {path, os.path.abspath(path), os.path.realpath(path)}:
        subs += [(v + "/", "" if name == "." else name + "/"), (v, name)]
subs.sort(key=lambda ab: -len(ab[0]))
def scrub(s):
    for a, b in subs:
        s = s.replace(a, b)
    return s
for p in glob.glob("*/run_info.json") + glob.glob("*/kb_info.json") + glob.glob("bulk/*/run_info.json"):
    s = open(p).read()
    open(p, "w").write(scrub(s))
for p in glob.glob("bulk/*/abundance.h5"):
    with h5py.File(p, "r+") as f:
        call = f["aux/call"][0].decode()
        del f["aux/call"]
        f["aux/call"] = [scrub(call).encode()]
PY
rm -rf "$WORK"
