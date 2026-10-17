"""Build the marker tables Polariseq ships, in python/polariseq/data/markers/.

    python scripts/build_marker_tables.py PROTEINATLAS_TSV C8_GMT CL_OBO

Inputs (all CC BY 4.0; attribution in THIRD_PARTY_NOTICES.md):

- proteinatlas.tsv: the Human Protein Atlas gene table (v25.1),
  https://www.proteinatlas.org/download/tsv/proteinatlas.tsv.zip
- c8.all.v2025.1.Hs.symbols.gmt: MSigDB C8 cell type signature gene sets,
  https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#C8
- cl-basic.obo: the Cell Ontology, http://purl.obolibrary.org/obo/cl/cl-basic.obo

Outputs: cell_ontology.tsv.gz (id, name, parents, synonyms), hpa.tsv.gz and
msigdb_c8.tsv.gz (one row per marker; columns in polariseq._markers.COLUMNS).
"""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "python" / "polariseq" / "data" / "markers"


def write(t: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(OUT / name, "wb", mtime=0) as f:  # mtime 0: the same bytes on every build
        f.write(t.to_csv(sep="\t", index=False).encode())
    print(f"{name}: {len(t):,} rows, {(OUT / name).stat().st_size / 1e3:.0f} kB")


# ---- Cell Ontology ---------------------------------------------------------

def cell_ontology(obo: Path) -> pd.DataFrame:
    rows, term = [], None
    for line in obo.read_text(encoding="utf-8").splitlines() + ["[Term]"]:
        if line.startswith("["):
            if term and term["id"].startswith("CL:") and not term["obsolete"]:
                rows.append((term["id"], term["name"], "|".join(term["parents"]),
                             "|".join(term["synonyms"])))
            term = {"id": "", "name": "", "parents": [], "synonyms": [], "obsolete": False} \
                if line == "[Term]" else None
        elif term is not None:
            key, _, value = line.partition(": ")
            if key == "id":
                term["id"] = value
            elif key == "name":
                term["name"] = value
            elif key == "is_a" and value.startswith("CL:"):
                term["parents"].append(value.split()[0])
            elif key == "synonym" and " EXACT" in value:
                term["synonyms"].append(value.split('"')[1])
            elif key == "is_obsolete" and value == "true":
                term["obsolete"] = True
    return pd.DataFrame(rows, columns=["id", "name", "parents", "synonyms"])


# ---- Human Protein Atlas ---------------------------------------------------

# How strongly HPA classes a gene: elevated in one type, in a group of 2 to 10,
# or above the mean of the other types.
HPA_EVIDENCE = {"Cell type enriched": 3, "Group enriched": 2, "Cell type enhanced": 1}

# The tissue each HPA single cell type was profiled in ("any": found in many).
HPA_TISSUE = {
    "Adipocytes": "adipose", "Adrenal cortex cells": "endocrine",
    "Adrenal medulla cells": "endocrine",
    "Alveolar cells type 1": "lung", "Alveolar cells type 2": "lung", "Astrocytes": "brain",
    "B-cells": "immune", "Basal keratinocytes": "skin", "Basal prostatic cells": "prostate",
    "Bergmann glia": "brain", "Brain excitatory neurons": "brain",
    "Brain inhibitory neurons": "brain", "Breast hormone-responsive cells": "breast",
    "Breast lactating cells": "breast", "Breast myoepithelial cells": "breast",
    "Breast secretory cells": "breast", "Cardiomyocytes": "heart", "Cholangiocytes": "liver",
    "Choroid plexus epithelial cells": "brain", "Colonocytes": "gut",
    "Cone photoreceptor cells": "eye", "Conjunctival goblet cells": "eye",
    "Corticotrophs": "endocrine", "Cytotrophoblasts": "placenta",
    "Decidual stromal cells": "uterus", "Differentiating spermatogonia": "testis",
    "Distal convoluted tubule cells": "kidney", "Early primary spermatocytes": "testis",
    "Early spermatids": "testis", "Endometrial ciliated cells": "uterus",
    "Endometrial glandular cells": "uterus", "Endometrial luminal cells": "uterus",
    "Endometrial secretory cells": "uterus", "Endometrial stromal cells": "uterus",
    "Enteric stem cells": "gut", "Enteric transient amplifying cells": "gut",
    "Enterocytes": "gut", "Ependymal cells": "brain", "Epicardial cells": "heart",
    "Epididymal basal cells": "testis", "Epididymal clear cells": "testis",
    "Epididymal efferent duct absorptive cells": "testis",
    "Epididymal efferent duct ciliated cells": "testis", "Epididymal principal cells": "testis",
    "Erythrocyte progenitors": "bone_marrow", "Erythrocytes": "immune",
    "Esophageal apical cells": "esophagus", "Esophageal basal cells": "esophagus",
    "Esophageal suprabasal cells": "esophagus", "Extravillous trophoblasts": "placenta",
    "Fallopian secretory cells": "ovary", "Fallopian tube ciliated cells": "ovary",
    "Fibro-adipogenic progenitors": "muscle", "Fibroblasts": "any", "Foveolar cells": "stomach",
    "Gastric chief cells": "stomach", "Gastric progenitor cells": "stomach",
    "Goblet cells": "gut", "Gonadotrophs": "endocrine", "Granulosa cells": "ovary",
    "Hematopoietic stem cells": "bone_marrow", "Hepatic stellate cells": "liver",
    "Hepatocytes": "liver", "Hofbauer cells": "placenta", "Innate lymphoid cells": "immune",
    "Kupffer cells": "liver", "Lacrimal acinar cells": "eye",
    "Lactotrophs": "endocrine", "Late primary spermatocytes": "testis",
    "Late spermatids": "testis", "Leydig cells": "testis",
    "Loop of henle epithelial cells": "kidney", "Lymphatic endothelial cells": "any",
    "Macrophages": "immune", "Mast cells": "immune",
    "Medullary thymic epithelial cells": "immune",
    "Megakaryocyte progenitors": "bone_marrow",
    "Megakaryocyte-Erythroid progenitors": "bone_marrow", "Megakaryocytes": "bone_marrow",
    "Melanocytes": "skin", "Mesothelial cells": "any", "Microglia": "brain",
    "Migrating cytotrophoblasts": "placenta", "Monocyte progenitors": "bone_marrow",
    "Mucous neck cells": "stomach", "Myonuclei": "muscle", "Myosatellite cells": "muscle",
    "Müller glia": "eye", "NK-cells": "immune", "Neuroendocrine cells": "any",
    "Neutrophil progenitors": "bone_marrow", "Neutrophils": "immune",
    "Ocular epithelial cells": "eye", "Oligodendrocyte progenitor cells": "brain",
    "Oligodendrocytes": "brain", "Oocytes": "ovary", "Other brain neurons": "brain",
    "Ovarian stromal cells": "ovary", "Pancreatic acinar cells": "pancreas",
    "Pancreatic duct cells": "pancreas", "Pancreatic islet cells": "pancreas",
    "Paneth cells": "gut", "Papillary tip epithelial cells": "kidney",
    "Parietal cells": "stomach", "Pericytes": "any", "Peritubular myoid cells": "testis",
    "Pituicytes/FSCs": "endocrine", "Pituitary stem cells": "endocrine",
    "Plasma cells": "immune", "Platelets": "immune", "Podocytes": "kidney",
    "Prostatic club cells": "prostate", "Prostatic glandular cells": "prostate",
    "Prostatic hillock cells": "prostate", "Proximal tubule cells": "kidney",
    "Renal collecting duct intercalated cells": "kidney",
    "Renal collecting duct principal cells": "kidney", "Renal connecting tubule cells": "kidney",
    "Respiratory basal cells": "lung", "Respiratory ciliated cells": "lung",
    "Respiratory deuterosomal cells": "lung", "Respiratory ionocytes": "lung",
    "Respiratory secretory cells": "lung", "Retinal amacrine cells": "eye",
    "Retinal bipolar cells": "eye", "Retinal ganglion cells": "eye",
    "Retinal horizontal cells": "eye", "Retinal pigment epithelial cells": "eye",
    "Rod photoreceptor cells": "eye", "Salivary acinar cells": "oral",
    "Salivary basal cells": "oral", "Salivary duct cells": "oral",
    "Salivary ionocytes": "oral", "Salivary myoepithelial cells": "oral",
    "Schwann cells": "any", "Sertoli cells": "testis", "Smooth muscle cells": "any",
    "Somatotrophs": "endocrine", "Submucosal glandular cells": "lung",
    "Suprabasal keratinocytes": "skin", "Syncytiotrophoblasts": "placenta", "T-cells": "immune",
    "Thymic myoid cells": "immune", "Thymocytes": "immune", "Thyrotrophs": "endocrine",
    "Transitional alveolar cells": "lung", "Tuft cells": "gut",
    "Undifferentiated spermatogonia": "testis", "Urothelial cells": "bladder",
    "Vascular endothelial cells": "any", "Vascular smooth muscle cells": "any",
    "cDC": "immune", "monocytes": "immune", "pDCs": "immune",
}
# Names a reader knows these types by.
HPA_NAMES = {"B-cells": "B cells", "T-cells": "T cells", "NK-cells": "NK cells",
             "monocytes": "Monocytes", "cDC": "Conventional dendritic cells",
             "pDCs": "Plasmacytoid dendritic cells", "Myonuclei": "Skeletal muscle fibres",
             "Pituicytes/FSCs": "Pituicytes and folliculostellate cells",
             "Loop of henle epithelial cells": "Loop of Henle epithelial cells"}


def hpa(tsv: Path) -> pd.DataFrame:
    d = pd.read_csv(tsv, sep="\t", low_memory=False,
                    usecols=["Gene", "RNA single cell type specificity",
                             "RNA single cell type specific nCPM"])
    rows = []
    for gene, cls, spec in d.itertuples(index=False):
        if cls not in HPA_EVIDENCE or not isinstance(spec, str):
            continue
        for part in spec.split(";"):
            ct, _, ncpm = part.rpartition(":")
            ct = ct.strip()
            if ct not in HPA_TISSUE:
                raise SystemExit(f"HPA cell type {ct!r} has no tissue in HPA_TISSUE: add it")
            rows.append((HPA_NAMES.get(ct, ct), gene, 1, HPA_EVIDENCE[cls], float(ncpm),
                         HPA_TISSUE[ct]))
    return pd.DataFrame(rows, columns=["cell_type", "gene", "sign", "evidence", "strength",
                                       "tissue"])


# ---- MSigDB C8 -------------------------------------------------------------

# Study prefix -> tissue; the rest of a set's name is the cell type.
C8_STUDIES = {
    "AIZARANI_LIVER": "liver", "BUSSLINGER_DUODENAL": "gut", "BUSSLINGER_ESOPHAGEAL": "esophagus",
    "BUSSLINGER_GASTRIC": "stomach", "CUI_DEVELOPING_HEART": "heart",
    "DURANTE_ADULT_OLFACTORY_NEUROEPITHELIUM": "nose", "FAN_EMBRYONIC_CTX": "brain",
    "FAN_OVARY": "ovary", "GAO_ESOPHAGUS": "esophagus", "GAO_LARGE_INTESTINE": "gut",
    "GAO_SMALL_INTESTINE": "gut", "GAO_STOMACH": "stomach", "GAUTAM_EYE": "eye",
    "HAY_BONE_MARROW": "bone_marrow", "HE_LIM_SUN_FETAL_LUNG": "lung",
    "HU_FETAL_RETINA": "eye", "JONES_OVARY": "ovary", "LAKE_ADULT_KIDNEY": "kidney",
    "MANNO_MIDBRAIN_NEUROTYPES": "brain", "MENON_FETAL_KIDNEY": "kidney",
    "MURARO_PANCREAS": "pancreas", "RUBENSTEIN_SKELETAL_MUSCLE": "muscle",
    "TRAVAGLINI_LUNG": "lung", "VANGURP_PANCREATIC": "pancreas", "ZHENG_CORD_BLOOD": "bone_marrow",
    "ZHONG_PFC": "brain", "DESCARTES_MAIN_FETAL": "any",
}
DESCARTES_ORGANS = {"ADRENAL": "endocrine", "CEREBELLUM": "brain", "CEREBRUM": "brain",
                    "EYE": "eye", "HEART": "heart", "INTESTINE": "gut", "KIDNEY": "kidney",
                    "LIVER": "liver", "LUNG": "lung", "MUSCLE": "muscle", "PANCREAS": "pancreas",
                    "PLACENTA": "placenta", "SPLEEN": "immune", "STOMACH": "stomach",
                    "THYMUS": "immune"}
# Manno et al. 2016 abbreviations of human midbrain types.
MANNO = {"HDA": "dopaminergic neurons", "HDA1": "dopaminergic neurons",
         "HDA2": "dopaminergic neurons", "HENDO": "endothelial cells",
         "HGABA": "GABAergic neurons", "HMGL": "microglia", "HNBGABA": "GABAergic neuroblasts",
         "HNBM": "medial neuroblasts", "HNBML1": "mediolateral neuroblasts",
         "HNBML5": "mediolateral neuroblasts", "HNPROG": "neuronal progenitors",
         "HOMTN": "oculomotor and trochlear nucleus neurons",
         "HOPC": "oligodendrocyte precursor cells", "HPERIC": "pericytes",
         "HPROGBP": "basal plate progenitors", "HPROGFPL": "lateral floor plate progenitors",
         "HPROGFPM": "medial floor plate progenitors", "HPROGM": "midline progenitors",
         "HRGL1": "radial glia-like cells", "HRGL2A": "radial glia-like cells",
         "HRGL2B": "radial glia-like cells", "HRGL2C": "radial glia-like cells",
         "HRGL3": "radial glia-like cells", "HRN": "red nucleus neurons",
         "HSERT": "serotonergic neurons", "BASAL": "basal plate progenitors"}
_DROP = re.compile(r"^(C\d+|\d+|\d+W|ADULT|5TH|WEEK|MAJOR|TYPES|BIG|GROUPS|CL\d+)$")


def c8_name(set_name: str) -> tuple[str, str] | None:
    """(cell type, tissue) of a C8 set, or None for one left out."""
    if set_name.startswith("SU_HO_"):  # chondrosarcoma and osteosarcoma: tumour states
        return None
    tissue, rest = None, ""
    if set_name.startswith("DESCARTES_FETAL_"):
        organ, _, rest = set_name[len("DESCARTES_FETAL_"):].partition("_")
        tissue = DESCARTES_ORGANS[organ]
    else:
        for prefix, t in sorted(C8_STUDIES.items(), key=lambda kv: -len(kv[0])):
            if set_name.startswith(prefix + "_"):
                tissue, rest = t, set_name[len(prefix) + 1:]
                break
    if tissue is None:
        raise SystemExit(f"C8 set {set_name} has no study in C8_STUDIES: add it")
    if set_name.startswith("MANNO_"):
        return MANNO[rest], tissue
    words: list[str] = []
    for w in (w for w in rest.split("_") if not _DROP.match(w)):
        if w in ("POS", "POSITIVE") and words:  # GENE_POS: cells positive for GENE
            words[-1] += "+"
        elif w in ("NEG", "NEGATIVE") and words:
            words[-1] += "-"
        else:
            words.append(re.sub(r"(?<=[A-Z0-9])POS$", "+", w))
    while words and re.fullmatch(r"\d+", words[-1]):
        words.pop()
    name = " ".join(_case(w) for w in words)
    if not name or "unknown" in name.lower() or "unspecified" in name.lower():
        return None
    return name[0].upper() + name[1:], tissue


_LOWER = {"AND", "OF", "THE", "LIKE", "PRE", "PRO", "MID", "CAP", "PIT", "TIP", "RED", "LOW",
          "HIGH", "HI", "SMG", "AEA", "DVR", "AVR", "TO", "IN", "ON", "BY", "FOR", "ITS"}


def _case(word: str) -> str:
    """A word of a set name as a reader writes it: gene symbols and short acronyms
    (CD8, NK, OPC) in capitals, other words in lower case."""
    core = word.rstrip("+-")
    if word.endswith(("+", "-")):
        return word
    if core in _LOWER:
        return word.lower()
    if re.search(r"\d", core) or len(core) <= 3:
        return word
    return word.lower()


def msigdb_c8(gmt: Path) -> pd.DataFrame:
    rows = []
    for line in gmt.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        got = c8_name(parts[0])
        if got is None:
            continue
        ct, tissue = got
        rows += [(ct, g, 1, 1, 0.0, tissue) for g in parts[2:] if g]
    t = pd.DataFrame(rows, columns=["cell_type", "gene", "sign", "evidence", "strength", "tissue"])
    # sets of one type in one tissue (clusters of one study, or several studies): a gene's
    # evidence is the number of sets naming it
    return (t.groupby(["cell_type", "gene", "sign", "tissue"], as_index=False)
            .agg(evidence=("evidence", "sum"), strength=("strength", "max")))


def main() -> None:
    hpa_tsv, gmt, obo = map(Path, sys.argv[1:4])
    on = cell_ontology(obo)
    write(on, "cell_ontology.tsv.gz")
    sys.path.insert(0, str(ROOT / "python"))
    from polariseq import _markers

    _markers.ontology.cache_clear()
    _markers._cl_index.cache_clear()
    names = dict(zip(on["id"], on["name"], strict=True))
    bad = {k: v for k, v in _markers._CL_NAMES.items() if v not in names}
    if bad:
        raise SystemExit(f"_CL_NAMES ids not in the ontology: {bad}")
    for name, build in (("hpa", lambda: hpa(hpa_tsv)), ("msigdb_c8", lambda: msigdb_c8(gmt))):
        t = build()
        types = pd.Series(t["cell_type"].unique())
        cl = dict(zip(types, types.map(_markers.match_cl), strict=True))
        t["cl_id"] = t["cell_type"].map(cl)
        t = t[["cell_type", "gene", "sign", "evidence", "strength", "tissue", "cl_id"]]
        write(t.sort_values(["cell_type", "gene"]), f"{name}.tsv.gz")
        unmatched = sorted(k for k, v in cl.items() if not v)
        print(f"  {len(types)} types, {len(types) - len(unmatched)} with a Cell Ontology term")
        if "-v" in sys.argv:
            print("  matched:", {k: f"{v} {names[v]}" for k, v in sorted(cl.items()) if v})
            print("  unmatched:", unmatched)


if __name__ == "__main__":
    main()
