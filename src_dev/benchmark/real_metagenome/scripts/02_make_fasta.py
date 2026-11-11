"""Split the host-filtered pairs into separate reads (R1 and R2 are independent reads from here on).

Read IDs: <ENA spot name>_1 / _2, e.g. ERR321493.17_1.
Outputs (in WORK/fasta/):
  reads_ge61.fa.gz                    reads >= 61 bp - Prodigal, FGS
  reads_lt61.fa.gz                    reads < 61 bp (stored; DeepCDS only, postprocessing later)
  deepcds/reads_all_cds_labels.fasta.gz  DeepCDS input, ALL reads (any length) in read order: every read as <id>|+
                                         and its reverse complement as <id>_rc|-; split by length in postprocessing
  read_counts.tsv                     counts per mate and length class, plus reads containing non-ACGT characters
Usage: python 02_make_fasta.py [--work /tmp/nrt204/real_metagenome_poc_work]
"""
import argparse
import gzip
import os
import subprocess
from collections import Counter

RUN = "ERR321493"
MIN_LEN = 61
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def pigz_writer(path):
    fh = open(path, "wb")
    proc = subprocess.Popen(["pigz", "-p", "8", "-c"], stdin=subprocess.PIPE, stdout=fh)
    return proc, fh


def read_fastq(path):
    proc = subprocess.Popen(["pigz", "-dc", path], stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    it = iter(proc.stdout)
    for header in it:
        seq = next(it).rstrip("\n")
        next(it)
        next(it)
        yield header[1:].split()[0], seq
    proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/nrt204/real_metagenome_poc_work")
    args = parser.parse_args()
    out = f"{args.work}/fasta"
    os.makedirs(f"{out}/deepcds", exist_ok=True)

    writers = {k: pigz_writer(p) for k, p in {
        "ge61": f"{out}/reads_ge61.fa.gz", "lt61": f"{out}/reads_lt61.fa.gz",
        "dc_all": f"{out}/deepcds/reads_all_cds_labels.fasta.gz"}.items()}
    buf = {k: [] for k in writers}

    def flush(force=False):
        for k, lines in buf.items():
            if lines and (force or len(lines) > 200_000):
                writers[k][0].stdin.write("".join(lines).encode())
                lines.clear()

    counts = Counter()
    for mate in (1, 2):
        for name, seq in read_fastq(f"{args.work}/host/{RUN}_{mate}.nohost.fastq.gz"):
            rid = f"{name}_{mate}"
            cls = "ge61" if len(seq) >= MIN_LEN else "lt61"
            counts[(mate, cls)] += 1
            if set(seq) - set("ACGT"):
                counts[(mate, cls, "non_ACGT")] += 1
            buf[cls].append(f">{rid}\n{seq}\n")
            buf["dc_all"].append(f">{rid}|+\n{seq}\n>{rid}_rc|-\n{seq.translate(COMP)[::-1]}\n")
            flush()
    flush(force=True)
    for proc, fh in writers.values():
        proc.stdin.close()
        proc.wait()
        fh.close()

    with open(f"{out}/read_counts.tsv", "w") as f:
        f.write("mate\tlength_class\treads\treads_with_non_ACGT\n")
        for mate in (1, 2):
            for cls in ("ge61", "lt61"):
                f.write(f"R{mate}\t{cls}\t{counts[(mate, cls)]}\t{counts[(mate, cls, 'non_ACGT')]}\n")
    print(open(f"{out}/read_counts.tsv").read())


if __name__ == "__main__":
    main()
