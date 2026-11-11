"""CDS extraction, translation and DIAMOND settings shared by steps 6, 7 and 8."""
import gzip
import os

DIAMOND = os.path.expanduser("~/.conda/envs/poc_env/bin/diamond")
COMP = str.maketrans("ACGTN", "TGCAN")
BASES = "TCAG"
CODE11 = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
TABLE11 = {a + b + c: CODE11[16 * i + 4 * j + k] for i, a in enumerate(BASES) for j, b in enumerate(BASES)
           for k, c in enumerate(BASES)}
TABLE4 = {**TABLE11, "TGA": "W"}
MIN_CDS = 61
TOOLS = ["DeepCDS S", "DeepCDS S+I", "MetaProdigal", "FGS (Complete)"]
CELL_NAMES = {(1, 1, 1): "all three", (1, 1, 0): "DeepCDS + MetaProdigal", (1, 0, 1): "DeepCDS + FGS",
              (1, 0, 0): "DeepCDS alone", (0, 1, 1): "MetaProdigal + FGS", (0, 1, 0): "MetaProdigal alone",
              (0, 0, 1): "FGS alone", (0, 0, 0): "none of S / MetaProdigal / FGS"}


def rc(s):
    return s.translate(COMP)[::-1]


def translate(nt, table):
    aa = "".join(table.get(nt[i:i + 3], "X") for i in range(0, len(nt) - 2, 3))
    return aa[:-1] if aa.endswith("*") else aa


def deepcds_cds(path, seqs):
    """[(read, [(start, end), ...] segments, oriented strand 'fwd'/'rc')] for DeepCDS CDS >= 61 bp (full extent)."""
    groups, out = {}, []
    with open(path) as fh:
        for line in fh:
            if line[0] == "#":
                continue
            c = line.split("\t")
            if len(c) < 9 or c[2] != "CDS":
                continue
            seqid = c[0]
            read, ori = (seqid[:-3], "rc") if seqid.endswith("_rc") else (seqid, "fwd")
            if read not in seqs:
                continue
            s, e = int(c[3]), int(c[4])
            gid = next((a[9:].rsplit(".", 1)[0] for a in c[8].split(";") if a.startswith("group_id=")), None)
            if gid is None:
                if e - s + 1 >= MIN_CDS:
                    out.append((read, [(s, e)], ori))
            else:
                groups.setdefault((seqid, gid), []).append((s, e))
    for (seqid, _), segs in groups.items():
        read, ori = (seqid[:-3], "rc") if seqid.endswith("_rc") else (seqid, "fwd")
        segs.sort()
        if segs[-1][1] - segs[0][0] + 1 >= MIN_CDS:
            out.append((read, segs, ori))
    return out


def tool_gff_cds(paths, seqs, tables=None):
    """[(read, [(start, end)], strand '+'/'-')] for Prodigal / FGS CDS >= 61 bp; tables: read -> genetic code."""
    out = []
    for path in paths:
        with gzip.open(path, "rt") as fh:
            seqhdr = None
            for line in fh:
                if line[0] == "#":
                    if tables is not None:
                        if line.startswith("# Sequence Data:"):
                            seqhdr = line.split('seqhdr="', 1)[1].split('"', 1)[0]
                        elif line.startswith("# Model Data:") and seqhdr in seqs:
                            tables[seqhdr] = int(line.split("transl_table=", 1)[1].split(";", 1)[0])
                    continue
                c = line.split("\t")
                if len(c) < 9 or c[2] != "CDS" or c[0] not in seqs:
                    continue
                s, e = int(c[3]), int(c[4])
                if e - s + 1 >= MIN_CDS:
                    out.append((c[0], [(s, e)], c[6]))
    return out


def peptide(seq, segs, strand, table):
    if strand == "rc":
        seq = rc(seq)
    if strand == "-":
        return translate(rc(seq[segs[0][0] - 1:segs[0][1]]), table)
    return "".join(translate(seq[s - 1:e], table) for s, e in segs)
