"""Curated cell type marker gene database.

Sources: CellMarker, PanglaoDB, scanpy tutorials, and published literature.
Each entry maps a cell type name to a list of canonical marker genes.

Usage::
    from polariseq.markers import MARKER_DB, get_tissue_markers
    pbmc_markers = get_tissue_markers("immune")
"""

from __future__ import annotations

# Organized by tissue/system for targeted annotation.
MARKER_DB: dict[str, dict[str, list[str]]] = {
    "immune": {
        "CD4 T cell":          ["CD3D", "CD3E", "CD4", "IL7R", "CCR7", "TCF7"],
        "CD8 T cell":          ["CD3D", "CD3E", "CD8A", "CD8B", "GZMK", "CCL5"],
        "Regulatory T cell":   ["CD3D", "CD4", "FOXP3", "IL2RA", "CTLA4", "TIGIT"],
        "B cell":              ["MS4A1", "CD79A", "CD79B", "TCL1A", "CD19", "BANK1"],
        "Plasma cell":         ["MZB1", "JCHAIN", "XBP1", "SDC1", "PRDM1"],
        "NK cell":             ["NKG7", "GNLY", "KLRD1", "PRF1", "NCAM1", "FGFBP2"],
        "Monocyte (CD14+)":    ["LYZ", "S100A8", "S100A9", "CD14", "FCN1", "VCAN"],
        "Monocyte (FCGR3A+)":  ["FCGR3A", "MS4A7", "CDKN1C", "LST1", "SMIM25"],
        "Dendritic cell":      ["FCER1A", "HLA-DQA1", "CLEC10A", "CD1C", "PKIB"],
        "Plasmacytoid DC":     ["LILRA4", "GZMB", "IRF7", "CLEC4C", "SERPINF1"],
        "Platelet":            ["PPBP", "PF4", "TUBB1", "ITGA2B", "NRGN"],
        "Macrophage":          ["CD68", "CD163", "MRC1", "CSF1R", "CD14"],
        "Mast cell":           ["TPSAB1", "TPSB2", "CPA3", "KIT", "MS4A2"],
        "Erythroid":           ["HBB", "HBA1", "HBA2", "ALAS2", "SLC4A1"],
    },
    "heart": {
        "Cardiomyocyte":       ["TNNT2", "MYH7", "ACTN2", "TPM1", "MYL2", "MYH6"],
        "Fibroblast":          ["COL1A1", "COL1A2", "DCN", "LUM", "POSTN", "COL3A1"],
        "Endothelial cell":    ["PECAM1", "VWF", "CLDN5", "CDH5", "EGFL7", "AQP1"],
        "Endothelial (arterial)": ["GJA5", "HEY1", "SEMA3G", "DLL4"],
        "Endothelial (venous)":  ["NR2F2", "ACKR1", "SELP"],
        "Endothelial (lymphatic)": ["PROX1", "PDPN", "LYVE1", "CCL21"],
        "Pericyte":            ["PDGFRB", "RGS5", "CSPG4", "NOTCH3", "MCAM"],
        "Smooth muscle cell":  ["ACTA2", "MYH11", "TAGLN", "MYL9", "CNN1"],
        "Macrophage":          ["CD68", "CD163", "MRC1", "CSF1R", "LYZ"],
        "T cell":              ["CD3D", "CD3E", "CD2"],
        "B cell":              ["MS4A1", "CD79A"],
        "NK cell":             ["NKG7", "GNLY"],
        "Adipocyte":           ["ADIPOQ", "PLIN1", "LEP", "LPL"],
    },
    "brain": {
        "Excitatory neuron":   ["SLC17A7", "CAMK2A", "GRIN1", "SATB2"],
        "Inhibitory neuron":   ["GAD1", "GAD2", "SLC6A1", "VIP", "SST", "PVALB"],
        "Dopaminergic neuron": ["TH", "SLC6A3", "DDC", "SLC18A2"],
        "Astrocyte":           ["GFAP", "AQP4", "SLC1A3", "ALDH1L1", "SOX9"],
        "Oligodendrocyte":     ["MBP", "MOG", "PLP1", "OLIG1", "OLIG2"],
        "Oligodendrocyte precursor": ["PDGFRA", "CSPG4", "NKX2-2"],
        "Microglia":           ["PTPRC", "AIF1", "CX3CR1", "P2RY12", "TMEM119"],
        "Neural stem cell":    ["SOX2", "NES", "PAX6", "HES5"],
        "Ependymal cell":      ["FOXJ1", "CCDC153", "TTR"],
        "Endothelial cell":    ["PECAM1", "CLDN5", "CDH5"],
    },
    "epithelial": {
        "Epithelial cell":     ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1"],
        "Basal cell":          ["KRT5", "KRT14", "TP63", "ITGA6"],
        "Goblet cell":         ["MUC2", "TFF3", "CLCA1"],
        "Paneth cell":         ["DEFA5", "DEFA6", "REG3A", "LYZ"],
        "Enterocyte":          ["ALPI", "SLC5A1", "FABP1"],
        "Enteroendocrine":     ["CHGA", "CHGB", "HCRT"],
        "Tuft cell":           ["POU2F3", "TRPM5", "DCLK1"],
    },
    "proliferating": {
        "Cycling cell":        ["MKI67", "TOP2A", "PCNA", "CDK1", "AURKA", "BIRC5"],
    },
    "stromal": {
        "Mesenchymal cell":    ["COL1A1", "COL1A2", "DCN", "VIM"],
        "Fibroblast":          ["COL1A1", "COL3A1", "DCN", "LUM", "POSTN"],
        "Myofibroblast":       ["ACTA2", "TAGLN", "COL1A1"],
    },
}

# Tissue aliases for user convenience.
TISSUE_ALIASES = {
    "pbmc": "immune",
    "blood": "immune",
    "immune": "immune",
    "cardiac": "heart",
    "heart": "heart",
    "cardio": "heart",
    "brain": "brain",
    "neural": "brain",
    "cortex": "brain",
    "epithelial": "epithelial",
    "gut": "epithelial",
    "intestine": "epithelial",
    "stromal": "stromal",
    "proliferating": "proliferating",
}


def get_tissue_markers(tissue: str) -> dict[str, list[str]]:
    """Get marker genes for a specific tissue/system.

    Args:
        tissue: Tissue name (e.g., 'immune', 'heart', 'brain'). Aliases
            like 'pbmc', 'blood', 'cardiac' are accepted.

    Returns:
        Dict mapping cell type name → list of marker gene symbols.
    """
    key = TISSUE_ALIASES.get(tissue.lower(), tissue.lower())
    return MARKER_DB.get(key, {})


def get_all_markers() -> dict[str, list[str]]:
    """Get all markers from all tissues (flattened)."""
    all_markers: dict[str, list[str]] = {}
    for tissue_dict in MARKER_DB.values():
        for cell_type, markers in tissue_dict.items():
            if cell_type not in all_markers:
                all_markers[cell_type] = markers
            else:
                # Merge unique markers.
                existing = set(all_markers[cell_type])
                all_markers[cell_type].extend(m for m in markers if m not in existing)
    return all_markers


def score_cell_type(
    gene_scores: dict[str, float],
    markers: list[str],
) -> float:
    """Score a cell type match given per-gene scores.

    Args:
        gene_scores: Dict mapping gene name → score (e.g., mean expression).
        markers: List of marker gene names for this cell type.

    Returns:
        Mean score across markers that are present in gene_scores.
    """
    found = [gene_scores[m] for m in markers if m in gene_scores]
    if not found:
        return 0.0
    return sum(found) / len(found)
