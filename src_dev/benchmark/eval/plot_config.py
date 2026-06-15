# Shared colorblind-friendly palette for all benchmark evaluation notebooks.
# Colors are based on the Okabe & Ito (2008) palette and Paul Tol's muted palette,
# which are safe for all common forms of color vision deficiency.
#
# Usage in notebooks (one directory deeper than this file):
#   import sys
#   sys.path.insert(0, '..')
#   from plot_config import MODEL_COLORS, MODEL_DISPLAY_NAMES, MODEL_MARKERS

MODEL_COLORS = {
    # FGS family — orange (light → dark)
    "fgs_complete":              "#D55E00",  # Light orange     (Dark2-derived)
    "fgs_illumina_5":            "#EEBC4E",  # Orange           (Dark2)
    "fgs_illumina_10":           "#F0E442",  # Dark orange      (Dark2-derived)
    "prodigal":                  "#009E73",  # Teal             (Dark2)

    # DeepCDS A2 (codon only) — green family (light → dark)
    "deep_cds_a2":               "#488700",  # Blue             
    "deep_cds_a2_no_errors":     "#488700",
    "deep_cds_a2_substitution":  "#68D930",  # Mid dark blue    
    "deep_cds_a2_errors":        "#C5EF6B",  # Dark blue        

    # DeepCDS A1 (pLM only) — blue family (light → dark)
    "deep_cds_a1":               "#CC79A7",  # Blue             (ColorBrewer)
    "deep_cds_a1_no_errors":     "#CC79A7",
    "deep_cds_a1_substitution":  "#BA1F89",  # Mid dark blue    (derived)
    "deep_cds_a1_errors":        "#6F0037",  # Dark blue        (derived)

    # DeepCDS Full — pink/magenta family (light → dark)
    "deep_cds":                  "#0072B2",  # Pink             (Dark2)
    "deep_cds_no_errors":        "#0072B2",
    "deep_cds_substitution":     "#56B4E9",  # Mid magenta      (derived)
    "deep_cds_errors":           "#9FB7C4",  # Dark magenta     (derived)
}

MODEL_DISPLAY_NAMES = {
    "fgs_complete":              "FGS (Complete)",
    "fgs_illumina_5":            r"FGS (0.5% error rate)",
    "fgs_illumina_10":           r"FGS (1.0% error rate)",
    "prodigal":                  "MetaProdigal",
    "deep_cds_a2":               "DeepCDS N (Codon)",
    "deep_cds_a2_no_errors":     "DeepCDS N (Codon)",
    "deep_cds_a2_substitution":  "DeepCDS S (Codon)",
    "deep_cds_a2_errors":        "DeepCDS S+I (Codon)",
    "deep_cds_a1":               "DeepCDS N (pLM)",
    "deep_cds_a1_no_errors":     "DeepCDS N (pLM)",
    "deep_cds_a1_substitution":  "DeepCDS S (pLM)",
    "deep_cds_a1_errors":        "DeepCDS S+I (pLM)",
    "deep_cds":                  "DeepCDS N",
    "deep_cds_no_errors":        "DeepCDS N",
    "deep_cds_substitution":     "DeepCDS S",
    "deep_cds_errors":           "DeepCDS S+I",

}

MODEL_MARKERS = {
    "fgs_complete":              "s",
    "fgs_illumina_5":            "v",
    "fgs_illumina_10":           "x",
    "prodigal":                  "^",
    "deep_cds_a1":               "D",
    "deep_cds_a1_no_errors":     "D",
    "deep_cds_a1_substitution":  "D",
    "deep_cds_a1_errors":        "d",
    "deep_cds":                  "o",
    "deep_cds_no_errors":        "o",
    "deep_cds_substitution":     "P",
    "deep_cds_errors":           "*",
    "deep_cds_codon_only":       "H",
}

# Font sizes — import and apply with plt.rcParams.update(FONT_SIZES) after plt.style.use()
FONT_SIZES = {
    "axes.labelsize":  16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 13,
    "axes.titlesize":  16,
}
