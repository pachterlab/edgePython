"""Simulates the genome, annotation and reads that make_kb_files.sh quantifies.

Run by make_kb_files.sh as:  python make_kb_sim.py <workdir>
Six genes on one 60 kb chromosome, three exons each, odd-numbered genes with a
second isoform that skips exon 2, genes 3 and 6 on the minus strand. Writes
genome.fa and genes.gtf; three bulk paired-end samples (bulk<n>_R1/R2.fq.gz,
20,000 fragments); and 10x v3 reads from 50 barcodes (sc_R1/R2.fq.gz, with
whitelist.txt), 5 of them near-empty, a quarter of reads from pre-mRNA.
"""

import gzip
import os
import sys

import numpy as np

os.chdir(sys.argv[1])
rng = np.random.default_rng(7)
chrom = "".join(rng.choice(list("ACGT"), 60000))
with open("genome.fa", "w") as f:
    f.write(">chr1\n" + "\n".join(chrom[i:i + 60] for i in range(0, len(chrom), 60)) + "\n")


def attr(g, t=None):
    s = f'gene_id "ENSG{g:011d}.1"; gene_name "GENE{g}"; gene_biotype "protein_coding";'
    if t is not None:
        s += (f' transcript_id "ENST{t:011d}.2"; transcript_name "GENE{g}-20{t % 10}";'
              ' transcript_biotype "protein_coding";')
    return s


gtf, spans, t = [], [], 0
for g in range(1, 7):
    start = 1000 + (g - 1) * 9000
    exons = [(start, start + 799), (start + 2000, start + 2999), (start + 4500, start + 5699)]
    spans.append((exons[0][0], exons[-1][1]))
    strand = "+" if g % 3 else "-"
    gtf.append(f"chr1\tsim\tgene\t{exons[0][0]}\t{exons[-1][1]}\t.\t{strand}\t.\t{attr(g)}")
    for ex in ([exons, [exons[0], exons[2]]] if g % 2 else [exons]):
        t += 1
        gtf.append(f"chr1\tsim\ttranscript\t{ex[0][0]}\t{ex[-1][1]}\t.\t{strand}\t.\t{attr(g, t)}")
        for i, (a, b) in enumerate(ex, 1):
            gtf.append(f"chr1\tsim\texon\t{a}\t{b}\t.\t{strand}\t.\t{attr(g, t)} exon_number \"{i}\";")
with open("genes.gtf", "w") as f:
    f.write("\n".join(gtf) + "\n")

# Transcript sequences, spliced from the GTF (reverse-complemented on the minus strand)
comp = str.maketrans("ACGT", "TGCA")
rc = lambda s: s.translate(comp)[::-1]
tx = {}
for line in gtf:
    f = line.split("\t")
    if f[2] == "exon":
        tid = f[8].split('transcript_id "')[1].split('"')[0]
        tx.setdefault(tid, [f[6], []])[1].append(chrom[int(f[3]) - 1:int(f[4])])
tx = {k: ("".join(v) if s == "+" else rc("".join(v))) for k, (s, v) in tx.items()}
names = list(tx)
q = lambda n: "I" * n

rng = np.random.default_rng(11)
for s in range(3):
    w = rng.gamma(1.0, 1.0, len(names))
    w /= w.sum()
    with gzip.open(f"bulk{s + 1}_R1.fq.gz", "wt") as f1, gzip.open(f"bulk{s + 1}_R2.fq.gz", "wt") as f2:
        for i, k in enumerate(rng.choice(len(names), 20000, p=w)):
            seq = tx[names[k]]
            L = min(len(seq), int(rng.normal(250, 20)))
            a = rng.integers(0, len(seq) - L + 1)
            frag = seq[a:a + L]
            f1.write(f"@r{i}\n{frag[:75]}\n+\n{q(75)}\n")
            f2.write(f"@r{i}\n{rc(frag)[:75]}\n+\n{q(75)}\n")

bcs = ["".join(rng.choice(list("ACGT"), 16)) for _ in range(50)]
with open("whitelist.txt", "w") as f:
    f.write("\n".join(bcs) + "\n")
with gzip.open("sc_R1.fq.gz", "wt") as f1, gzip.open("sc_R2.fq.gz", "wt") as f2:
    i = 0
    for c, bc in enumerate(bcs):
        w = rng.gamma(0.7, 1.0, len(names))
        w /= w.sum()
        for _ in range(int(rng.integers(100, 600)) if c < 45 else 5):
            umi = "".join(rng.choice(list("ACGT"), 12))
            if rng.random() < 0.25:
                g = rng.integers(0, 6)
                p = rng.integers(spans[g][0], spans[g][1] - 90)
                r2 = chrom[p - 1:p + 89]
                if g % 3 == 2:
                    r2 = rc(r2)
            else:
                seq = tx[names[rng.choice(len(names), p=w)]]
                p = rng.integers(0, len(seq) - 90)
                r2 = seq[p:p + 90]
            f1.write(f"@s{i}\n{bc}{umi}\n+\n{q(28)}\n")
            f2.write(f"@s{i}\n{r2}\n+\n{q(90)}\n")
            i += 1
