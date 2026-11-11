"""Step 7: benchmark table from the full-sample DIAMOND check (12_diamond_full_sample.py).

Part A, per tool: reads >= 61 bp with >= 1 predicted CDS >= 61 bp, predicted CDS, CDS with a UniRef90 hit.
Part B, per read group (which of DeepCDS S / MetaProdigal / FGS (Complete) predict a CDS on the read): reads, and
the share of each of these three tools' CDS in the group with a UniRef90 hit.
Also printed: hit rates without CDS containing an in-frame stop codon (table footnotes).
Inputs: RES/{detection_rates.tsv, diamond_full_cds_per_tool.tsv, diamond_full_cds_per_cell.tsv}
Outputs: RES/table_real_metagenome_{per_tool,per_group}.tsv, RES/table_real_metagenome.tex
Usage: python 13_results_table.py
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "/tmp/nrt204/FragmentPredictor/real_metagenome_poc/results"
TOOLS = ["DeepCDS N", "DeepCDS S", "DeepCDS S+I", "MetaProdigal", "FGS (Complete)", "FGS (Illumina 5)",
         "FGS (Illumina 10)"]
GROUP_TOOLS = ["DeepCDS S", "MetaProdigal", "FGS (Complete)"]
GROUPS = ["all three", "DeepCDS + MetaProdigal", "DeepCDS + FGS", "MetaProdigal + FGS", "DeepCDS alone",
          "MetaProdigal alone", "FGS alone"]


def pct(x, d=1):
    return "--" if pd.isna(x) else f"{100 * x:.{d}f}"


def main():
    n_reads = int(pd.read_csv(f"{RES}/detection_rates.tsv", sep="\t").query("category == 'all'")["n_reads"].iloc[0])
    tool = pd.read_csv(f"{RES}/diamond_full_cds_per_tool.tsv", sep="\t", index_col=0).loc[TOOLS]
    cell = pd.read_csv(f"{RES}/diamond_full_cds_per_cell.tsv", sep="\t").set_index(["cell", "tool"])

    a = pd.DataFrame({"reads_with_cds": tool["reads"], "read_rate": tool["reads"] / n_reads,
                      "cds": tool["CDS"].astype(int), "cds_with_hit": tool["with_hit"].astype(int),
                      "hit_rate": tool["hit_rate"]})
    stop = tool["CDS_in_frame_stop"].fillna(0)
    a["hit_rate_excl_in_frame_stop"] = ((tool["with_hit"] - stop * tool["hit_rate_in_frame_stop"].fillna(0))
                                        / (tool["CDS"] - stop))
    a["cds_in_frame_stop"] = stop.astype(int)

    rows = []
    for g in GROUPS:
        r = {"group": g}
        r["reads"] = int(max(cell.loc[(g, t), "reads"] for t in GROUP_TOOLS if (g, t) in cell.index))
        r["read_share"] = r["reads"] / n_reads
        for t in GROUP_TOOLS:
            r[f"hit_rate {t}"] = cell.loc[(g, t), "hit_rate"] if (g, t) in cell.index else float("nan")
            r[f"cds {t}"] = int(cell.loc[(g, t), "CDS"]) if (g, t) in cell.index else 0
        rows.append(r)
    b = pd.DataFrame(rows).set_index("group")

    a.to_csv(f"{RES}/table_real_metagenome_per_tool.tsv", sep="\t")
    b.to_csv(f"{RES}/table_real_metagenome_per_group.tsv", sep="\t")

    tex = [r"\begin{tabular}{lrrr}", r"\toprule",
           r"\multicolumn{4}{l}{\textbf{A} Per tool} \\",
           r"Tool & Reads with CDS (\%) & Predicted CDS ($\times 10^6$) & CDS with UniRef90 hit (\%) \\",
           r"\midrule"]
    for t, r in a.iterrows():
        tex.append(f"{t} & {pct(r.read_rate)} & {r.cds / 1e6:.1f} & {pct(r.hit_rate)} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", "", r"\bigskip", "",
            r"\begin{tabular}{lrrrr}", r"\toprule",
            r"\multicolumn{5}{l}{\textbf{B} Per read group} \\",
            r" & & \multicolumn{3}{c}{CDS with UniRef90 hit (\%)} \\", r"\cmidrule(l){3-5}",
            r"CDS predicted by & Reads (\%) & DeepCDS S & MetaProdigal & FGS (Complete) \\",
            r"\midrule"]
    for g, r in b.iterrows():
        rates = " & ".join(pct(r[f"hit_rate {t}"]) for t in GROUP_TOOLS)
        label = "All three" if g == "all three" else g.replace("DeepCDS", "DeepCDS S")
        tex.append(f"{label} & {pct(r.read_share)} & {rates} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(f"{RES}/table_real_metagenome.tex", "w") as fh:
        fh.write("\n".join(tex) + "\n")

    print(f"reads >= 61 bp: {n_reads:,}")
    print(a.to_string(float_format=lambda x: f"{x:.3f}"))
    print(b.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n".join(tex))


if __name__ == "__main__":
    main()
