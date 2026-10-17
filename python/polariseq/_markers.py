"""Marker tables for :meth:`tl.auto_annotate`, from the source a user chooses.

Every source is read into one table, a row per marker:

``cell_type``  the source's name for the cell type
``gene``       gene symbol, as the source writes it
``sign``       ``+1`` a gene the type expresses, ``-1`` one it should not
``evidence``   how many records support the row (studies, databases, ...)
``strength``   a source's own measure for ranking a type's markers (0: none)
``tissue``     one of :data:`TISSUES`, ``"any"``, or the source's own word
``cl_id``      the type's Cell Ontology term (``""`` when not matched)
``species``    ``"human"`` or ``"mouse"``

Bundled with Polariseq (open licences, attribution in THIRD_PARTY_NOTICES.md):
the Human Protein Atlas single cell types (``"hpa"``), the MSigDB C8 cell type
signatures (``"msigdb_c8"``), and Polariseq's own short list (``"curated"``).
Downloaded once on request into the cache folder (:func:`cache_dir`), since
their terms do not allow passing them on: ``"accordion"``, ``"cellmarker"``,
``"panglaodb"``, ``"sctype"``.
"""

from __future__ import annotations

import gzip
import os
import re
import sys
import urllib.request
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data" / "markers"
COLUMNS = ["cell_type", "gene", "sign", "evidence", "strength", "tissue", "cl_id", "species"]

# The tissues a table can be narrowed to, each with the words that name it in
# the sources. A word matches at the start of a word of the source's tissue
# ("uter" finds "uterus", not "outer"); the tissues are tried in this order, so
# "cortex of thymus" is immune, "renal cortex" kidney and "cerebral cortex" brain.
TISSUES: dict[str, tuple[str, ...]] = {
    "bone_marrow": ("bone marrow", "marrow", "cord blood", "hematopoietic"),
    "immune": ("blood", "pbmc", "immune", "lymph", "spleen", "splenic", "tonsil", "thymus",
               "germinal", "thoracic duct", "cisterna chyli"),
    "kidney": ("kidney", "renal", "nephr", "glomerul", "ureter", "collecting duct", "tubule",
               "thin limb", "macula densa", "juxtaglomerular", "pronephros", "loop of henle"),
    "liver": ("liver", "hepat", "bile", "biliary", "gall", "portal triad"),
    "pancreas": ("pancrea", "islet"),
    "lung": ("lung", "airway", "bronch", "alveol", "trachea", "respiratory", "pleura",
             "pulmonary", "sputum"),
    "heart": ("heart", "cardi", "myocard", "epicard", "endocard", "atrium", "atrial"),
    "endocrine": ("adrenal", "pituitar", "hypophysis", "neurohypophysis", "thyroid",
                  "parathyroid"),
    "stomach": ("stomach", "gastric", "antrum", "fundus"),
    "esophagus": ("esophag", "oesophag"),
    "gut": ("intestin", "colon", "gut", "duoden", "ileum", "jejun", "rectum", "bowel", "cecum",
            "gastrointestinal", "digestive", "gi tract", "enteric"),
    "brain": ("brain", "cortex", "cortical", "cerebr", "cerebell", "hippocamp", "midbrain",
              "forebrain", "hindbrain", "neural", "nervous system", "central nervous", "spinal",
              "striatum", "hypothalam", "thalam", "olfactory bulb", "gyrus", "colliculus",
              "ganglionic eminence", "tegment", "pons", "medulla oblongata", "amygdal",
              "putamen", "caudate", "pallidus", "nigra", "accumbens", "subiculum",
              "white matter", "gray matter", "grey matter", "corpus callosum", "diencephalon",
              "telencephalon", "limbic", "mening", "meninx", "dura mater", "choroid plexus",
              "ventricular zone", "subparaventricular", "circumventricular", "cerebrospinal",
              "entorhinal", "temporal lobe", "frontal", "somatosensory", "subfornical",
              "nucleus", "neocortex"),
    "nerve": ("nerve", "ganglion", "peripheral nervous"),
    "eye": ("eye", "retina", "cornea", "lens", "iris", "choroid", "conjunctiv", "lacrimal"),
    "skin": ("skin", "epiderm", "dermis", "keratinocyte", "hair follicle"),
    "muscle": ("skeletal muscle", "muscle", "myo"),
    "adipose": ("adipose", "fat"),
    "breast": ("breast", "mammary"),
    "prostate": ("prostat",),
    "testis": ("testis", "testes", "testic", "sperm", "epididym"),
    "ovary": ("ovar", "follic", "oocyte", "fallopian"),
    "uterus": ("uterus", "uterine", "endometri", "decidua", "cervix", "vagina"),
    "placenta": ("placenta", "trophoblast", "villous"),
    "bone": ("bone", "cartilage", "chondr", "osteo", "synovi", "suture"),
    "bladder": ("bladder", "urothel"),
    "nose": ("nose", "nasal", "olfactory"),
    "oral": ("oral", "salivary", "tongue", "tooth", "gingiva", "mouth", "taste bud"),
    "vasculature": ("vascul", "vessel", "artery", "arteri", "vein", "venule", "venous", "aorta",
                    "endotheli", "capillary"),
    "embryo": ("embryo", "fetal", "foetal", "fetus"),
}
_TISSUE_PATTERNS = {k: re.compile(r"\b(?:" + "|".join(map(re.escape, words)) + ")")
                    for k, words in TISSUES.items()}
_TISSUE_ALIASES = {"blood": "immune", "pbmc": "immune", "neural": "brain", "cardiac": "heart",
                   "intestine": "gut", "colon": "gut", "retina": "eye",
                   "airway": "lung", "marrow": "bone_marrow"}
# Groups of types a source gives for no tissue in particular.
_ANY_TISSUE = {"na", "nan", "none", "undefined", "any", "all", "connective tissue", "epithelium",
               "epithelial", "stroma", "stromal", "proliferating", "smooth muscle", "multiple",
               "various", "zygote"}


def tissue_key(text: Any) -> str:
    """The tissue of :data:`TISSUES` a source's own tissue word names (else the word, lower case)."""
    t = str(text or "").strip().lower()
    if not t or t in _ANY_TISSUE:
        return "any"
    if t in TISSUES:
        return t
    if t in _TISSUE_ALIASES:
        return _TISSUE_ALIASES[t]
    for key, pattern in _TISSUE_PATTERNS.items():
        if pattern.search(t):
            return key
    return t


# ---- sources ---------------------------------------------------------------

SOURCES: dict[str, dict[str, Any]] = {
    "hpa": {
        "what": "Human Protein Atlas single cell type atlas (v25): genes enriched or enhanced in 154 "
                "cell types across 31 tissues",
        "licence": "CC BY 4.0", "bundled": True,
        "cite": "Karlsson et al., Sci. Adv. 7, eabh2169 (2021); proteinatlas.org",
    },
    "msigdb_c8": {
        "what": "MSigDB C8 cell type signature gene sets (2025.1): 866 sets from 22 single-cell "
                "studies of human tissues, fetal and adult",
        "licence": "CC BY 4.0", "bundled": True,
        "cite": "Liberzon et al., Cell Syst. 1, 417 (2015); gsea-msigdb.org",
    },
    "curated": {
        "what": "Polariseq's short list of canonical markers (about 40 types, the earlier "
                "method='markers')",
        "licence": "Polariseq", "bundled": True, "cite": "",
    },
    "accordion": {
        "what": "Cell Marker Accordion: 13 databases harmonized to the Cell Ontology, human and "
                "mouse, with negative markers",
        "licence": "MIT (the package); the databases it gathers keep their own terms",
        "url": "https://raw.githubusercontent.com/TebaldiLab/cellmarkeraccordion/master/data/"
               "accordion_marker.rda",
        "file": "accordion_marker.rda", "needs": "pyreadr",
        "cite": "Busarello et al., Nat. Commun. 16, 11 (2025)",
    },
    "cellmarker": {
        "what": "CellMarker 2.0: markers gathered from publications, human and mouse",
        "licence": "free for academic use, terms at the source",
        "url": "http://117.50.127.228/CellMarker/CellMarker_download_files/file/"
               "Cell_marker_{Species}.xlsx",
        "file": "Cell_marker_{Species}.xlsx", "needs": "openpyxl",
        "cite": "Hu et al., Nucleic Acids Res. 51, D870 (2023)",
    },
    "panglaodb": {
        "what": "PanglaoDB markers (27 March 2020): 178 cell types, 30 organs (the earlier "
                "default)",
        "licence": "terms at the source",
        "url": "https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz",
        "file": "PanglaoDB_markers_27_Mar_2020.tsv.gz",
        "cite": "Franzen et al., Database 2019, baz046 (2019)",
    },
    "sctype": {
        "what": "ScType marker database: positive and negative markers of about 200 types in 16 "
                "tissues",
        "licence": "GPL-3.0 (downloaded by you, never shipped with Polariseq)",
        "url": "https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/"
               "ScTypeDB_full.xlsx",
        "file": "ScTypeDB_full.xlsx", "needs": "openpyxl",
        "cite": "Ianevski et al., Nat. Commun. 13, 1246 (2022)",
    },
}
_SOURCE_ALIASES = {"c8": "msigdb_c8", "msigdb": "msigdb_c8", "markers": "curated",
                   "cellmarker2": "cellmarker", "panglao": "panglaodb", "proteinatlas": "hpa",
                   "cell_marker_accordion": "accordion"}


def cache_dir() -> Path:
    """Where downloaded marker databases are kept: ``POLARISEQ_CACHE`` when set,
    else the user's cache folder (``~/.cache/polariseq``, ``%LOCALAPPDATA%\\polariseq``
    on Windows)."""
    if os.environ.get("POLARISEQ_CACHE"):
        return Path(os.environ["POLARISEQ_CACHE"]) / "markers"
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "polariseq" / "markers"
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "polariseq" / "markers"


def source_name(source: str) -> str:
    """The canonical name of a source, or a clear error listing them."""
    from polariseq import _checks

    s = str(source).strip().lower()
    s = _SOURCE_ALIASES.get(s, s)
    if s not in SOURCES:
        raise _checks.InputError(
            f"no marker source named {source!r}.{_checks.suggest(str(source), list(SOURCES))} "
            f"Sources: {_checks.listing(SOURCES)}, or pass your own table with markers=.")
    return s


def _fetch(name: str, species: str = "human") -> Path:
    """The downloaded file of an on-request source, fetched once into :func:`cache_dir`."""
    info = SOURCES[name]
    species_word = "Human" if species == "human" else "Mouse"
    file = cache_dir() / info["file"].format(Species=species_word)
    if file.exists() and file.stat().st_size > 0:
        return file
    url = info["url"].format(Species=species_word)
    file.parent.mkdir(parents=True, exist_ok=True)
    print(f"auto_annotate: downloading the {name} marker database once ({info['licence']}; "
          f"cite {info['cite']})\n  from {url}\n  to   {file}", file=sys.stderr)
    tmp = file.with_suffix(file.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (polariseq)"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(file)
    except Exception as e:  # noqa: BLE001  (any network failure gets the same advice)
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not download the {name} marker database ({type(e).__name__}: {e}). Download "
            f"it yourself from {url} and save it as {file}; it is read from there from then on. "
            f"The bundled sources need no download: source='hpa', 'msigdb_c8' or 'curated'."
        ) from None
    return file


def _needs(name: str) -> None:
    pkg = SOURCES[name].get("needs")
    if pkg is None:
        return
    try:
        __import__(pkg)
    except ImportError:
        raise ImportError(f"source={name!r} reads its database with {pkg}: pip install {pkg} "
                          "(or the extra: pip install 'polariseq[markers]')") from None


def _read_bundled(name: str) -> pd.DataFrame:
    path = DATA / f"{name}.tsv.gz"
    if not path.exists():
        raise FileNotFoundError(f"the bundled marker table {path} is missing: reinstall polariseq")
    t = pd.read_csv(path, sep="\t", keep_default_na=False, dtype={"cl_id": str, "tissue": str})
    t["species"] = "human"
    return t


def _curated() -> pd.DataFrame:
    from polariseq.markers import MARKER_DB

    rows = [(ct, g, 1, 1, 0.0, tissue_key(tissue), "", "human")
            for tissue, types in MARKER_DB.items() for ct, genes in types.items() for g in genes]
    return pd.DataFrame(rows, columns=COLUMNS)


def _panglaodb(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path, sep="\t", keep_default_na=False)
    rows = []
    for sp, code in (("human", "Hs"), ("mouse", "Mm")):
        s = t[t["species"].str.contains(code)]
        sens = pd.to_numeric(s[f"sensitivity_{sp}"], errors="coerce").fillna(0.0)
        canonical = pd.to_numeric(s["canonical marker"], errors="coerce").fillna(0).astype(int)
        rows.append(pd.DataFrame({
            "cell_type": s["cell type"], "gene": s["official gene symbol"], "sign": 1,
            "evidence": 1 + canonical, "strength": sens.to_numpy(),
            "tissue": s["organ"].map(tissue_key), "cl_id": "", "species": sp}))
    return pd.concat(rows, ignore_index=True)


def _cellmarker(path: Path, species: str) -> pd.DataFrame:
    t = pd.read_excel(path)
    t = t[(t["cancer_type"] == "Normal") & (t["cell_type"] == "Normal cell")]
    t = t.dropna(subset=["Symbol", "cell_name"])
    t = t.assign(tissue=t["tissue_class"].map(tissue_key),
                 cl_id=t["cellontology_id"].fillna("").astype(str).str.replace("_", ":"))
    g = (t.groupby(["cell_name", "Symbol", "tissue", "cl_id"])["PMID"].nunique(dropna=False)
         .reset_index(name="evidence"))
    g["evidence"] = g["evidence"].clip(lower=1)
    return pd.DataFrame({"cell_type": g["cell_name"], "gene": g["Symbol"], "sign": 1,
                         "evidence": g["evidence"], "strength": 0.0, "tissue": g["tissue"],
                         "cl_id": g["cl_id"], "species": species})


def _sctype(path: Path) -> pd.DataFrame:
    t = pd.read_excel(path)
    rows = []
    for _, r in t.iterrows():
        for col, sign in (("geneSymbolmore1", 1), ("geneSymbolmore2", -1)):
            for g in str(r[col]).split(",") if pd.notna(r[col]) else []:
                g = g.strip().upper()
                if g and g not in ("NA", "NAN"):
                    rows.append((r["cellName"], g, sign, 1, 0.0, tissue_key(r["tissueType"]), "",
                                 "human"))
    return pd.DataFrame(rows, columns=COLUMNS).drop_duplicates()


def _accordion(path: Path) -> pd.DataFrame:
    import pyreadr

    t = next(iter(pyreadr.read_r(str(path)).values()))
    # one row per (database, tissue) record: counted as evidence, as the Accordion counts its ECs
    return pd.DataFrame({
        "cell_type": t["CL_celltype"].astype(str), "gene": t["marker"].astype(str),
        "sign": np.where(t["marker_type"].astype(str) == "negative", -1, 1), "evidence": 1,
        "strength": 0.0, "tissue": t["Uberon_tissue"].map(tissue_key),
        "cl_id": t["CL_ID"].astype(str), "species": t["species"].astype(str).str.lower()})


@lru_cache(maxsize=16)
def _source_table(name: str, species: str) -> pd.DataFrame:
    if SOURCES[name].get("bundled"):
        t = _curated() if name == "curated" else _read_bundled(name)
    else:
        _needs(name)
        if name == "cellmarker":
            t = _cellmarker(_fetch(name, species), species)
        else:
            path = _fetch(name)
            t = {"panglaodb": _panglaodb, "sctype": _sctype, "accordion": _accordion}[name](path)
    t = t[COLUMNS].copy()
    missing = t["cl_id"].isin(["", "nan", "None"])
    if missing.any():
        t.loc[missing, "cl_id"] = t.loc[missing, "cell_type"].map(match_cl)
    t["source"] = name
    return t


def user_table(markers: Any) -> pd.DataFrame:
    """A table of the user's own markers: a DataFrame with ``cell_type`` and
    ``gene`` columns (``sign``, ``evidence``, ``tissue`` optional), a dict
    ``{type: [genes]}`` or ``{type: {"positive": [...], "negative": [...]}}``,
    or the path of a ``.gmt``, ``.csv`` or ``.tsv`` file."""
    from polariseq import _checks

    if isinstance(markers, (str, Path)):
        p = Path(markers)
        if not p.is_file():
            raise FileNotFoundError(f"auto_annotate markers: no file at {p}")
        if p.suffix.lower() == ".gmt":
            rows = []
            for line in p.read_text(encoding="utf-8").splitlines():
                parts = line.rstrip("\n").split("\t")
                rows += [(parts[0], g) for g in parts[2:] if g]
            markers = pd.DataFrame(rows, columns=["cell_type", "gene"])
        else:
            markers = pd.read_csv(p, sep="\t" if p.suffix.lower() in (".tsv", ".txt") else ",")
    if isinstance(markers, dict):
        rows = []
        for ct, genes in markers.items():
            if isinstance(genes, dict):
                unknown = set(genes) - {"positive", "negative"}
                if unknown:
                    raise _checks.InputError(
                        f"auto_annotate markers: {ct!r} has keys {sorted(unknown)}; a type's "
                        "markers are a list of genes or {'positive': [...], 'negative': [...]}")
                rows += [(ct, g, 1) for g in genes.get("positive", [])]
                rows += [(ct, g, -1) for g in genes.get("negative", [])]
            else:
                if isinstance(genes, str):
                    genes = [genes]
                rows += [(ct, g, 1) for g in genes]
        markers = pd.DataFrame(rows, columns=["cell_type", "gene", "sign"])
    if not isinstance(markers, pd.DataFrame):
        raise _checks.InputError(
            f"auto_annotate markers: expected a DataFrame, a dict or a file path, not "
            f"{type(markers).__name__}")
    cols = {c.lower(): c for c in markers.columns}
    need = [c for c in ("cell_type", "gene") if c not in cols]
    if need:
        raise _checks.InputError(
            f"auto_annotate markers: the table needs columns cell_type and gene; missing "
            f"{_checks.listing(need)} (columns: {_checks.listing(markers.columns)})")
    t = pd.DataFrame({"cell_type": markers[cols["cell_type"]].astype(str),
                      "gene": markers[cols["gene"]].astype(str)})
    if "sign" in cols:  # -1, "negative" or "-" mark a gene the type should not express
        raw = markers[cols["sign"]]
        word = raw.astype(str).str.strip().str.lower()
        negative = (pd.to_numeric(raw, errors="coerce") < 0) | word.isin(["negative", "neg", "-"])
        t["sign"] = np.where(negative.to_numpy(), -1, 1)
    else:
        t["sign"] = 1
    t["evidence"] = markers[cols["evidence"]].astype(float) if "evidence" in cols else 1
    t["strength"] = 0.0
    t["tissue"] = markers[cols["tissue"]].map(tissue_key) if "tissue" in cols else "any"
    t["cl_id"] = markers[cols["cl_id"]].astype(str) if "cl_id" in cols else ""
    t["species"] = "any"
    t["source"] = "markers"
    return t


# ---- Cell Ontology ---------------------------------------------------------

@lru_cache(maxsize=1)
def ontology() -> pd.DataFrame:
    """The Cell Ontology terms: ``id``, ``name``, ``parents`` (``|``-separated), ``synonyms``."""
    return pd.read_csv(DATA / "cell_ontology.tsv.gz", sep="\t", keep_default_na=False)


def _norm(name: str) -> str:
    s = str(name).lower().replace("-", " ").replace("_", " ").replace("+", " positive ")
    s = re.sub(r"[(),/]", " ", s)
    words = []
    for w in s.split():
        if w in ("cells", "cell"):
            w = "cell"
        elif len(w) > 3 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        words.append(w)
    return " ".join(words)


# Names the sources use that the ontology words differently.
_CL_NAMES = {
    "t": "CL:0000084", "b": "CL:0000236", "nk": "CL:0000623", "monocyte": "CL:0000576",
    "cdc": "CL:0000990", "pdc": "CL:0000784", "dc": "CL:0000451",
    "dendritic cell": "CL:0000451", "nk cell": "CL:0000623", "natural killer cell": "CL:0000623",
    "cd4 t cell": "CL:0000624", "cd8 t cell": "CL:0000625", "cd14 monocyte": "CL:0001054",
    "classical monocyte": "CL:0000860", "non classical monocyte": "CL:0000875",
    "nonclassical monocyte": "CL:0000875", "fcgr3a monocyte": "CL:0002396",
    "cd16 monocyte": "CL:0002396", "regulatory t cell": "CL:0000815", "treg": "CL:0000815",
    "plasmacytoid dendritic cell": "CL:0000784", "conventional dendritic cell": "CL:0000990",
    "brain excitatory neuron": "CL:0000679", "excitatory neuron": "CL:0000679",
    "brain inhibitory neuron": "CL:0000617", "inhibitory neuron": "CL:0000617",
    "other brain neuron": "CL:0000540", "neuron": "CL:0000540", "interneuron": "CL:0000099",
    "erythrocyte": "CL:0000232", "erythroid cell": "CL:0000764", "platelet": "CL:0000233",
    "megakaryocyte": "CL:0000556", "hematopoietic stem cell": "CL:0000037",
    "alveolar cell type 1": "CL:0002062", "alveolar cell type 2": "CL:0002063",
    "fibroblast": "CL:0000057", "vascular endothelial cell": "CL:0002139",
    "endothelial cell": "CL:0000115", "smooth muscle cell": "CL:0000192",
    "skeletal muscle fibre": "CL:0008002", "cardiomyocyte": "CL:0000746", "astrocyte": "CL:0000127",
    "microglia": "CL:0000129", "microglial cell": "CL:0000129", "oligodendrocyte": "CL:0000128",
    "oligodendrocyte progenitor cell": "CL:0002453", "opc": "CL:0002453",
    "macrophage": "CL:0000235", "mast cell": "CL:0000097", "neutrophil": "CL:0000775",
    "plasma cell": "CL:0000786", "hepatocyte": "CL:0000182", "kupffer cell": "CL:0000091",
    "cholangiocyte": "CL:1000488", "pericyte": "CL:0000669", "adipocyte": "CL:0000136",
    "keratinocyte": "CL:0000312", "melanocyte": "CL:0000148", "podocyte": "CL:0000653",
    "proximal tubule cell": "CL:0002306", "innate lymphoid cell": "CL:0001065",
    "thymocyte": "CL:0000893", "goblet cell": "CL:0000160", "tuft cell": "CL:0002204",
    "enterocyte": "CL:0000584", "paneth cell": "CL:0000510", "schwann cell": "CL:0002573",
    "hofbauer cell": "CL:3000001", "syncytiotrophoblast": "CL:0000525",
    "cytotrophoblast": "CL:0000523", "extravillou trophoblast": "CL:0008036",
    "lymphatic endothelial cell": "CL:0002138", "mesothelial cell": "CL:0000077",
    "ependymal cell": "CL:0000065", "bergmann glia": "CL:0000644", "muller glia": "CL:0000636",
    "müller glia": "CL:0000636", "rod photoreceptor cell": "CL:0000604",
    "cone photoreceptor cell": "CL:0000573", "retinal ganglion cell": "CL:0000740",
    "erythroblast": "CL:0000765", "eosinophil": "CL:0000771", "basophil": "CL:0000767",
    "alpha cell": "CL:0000171", "beta cell": "CL:0000169", "delta cell": "CL:0000173",
    "gamma cell": "CL:0002275", "pp cell": "CL:0002275", "epsilon cell": "CL:0005019",
    "k cell": "CL:0002278", "i cell": "CL:0002277", "ec cell": "CL:0000577",
    "ductal cell": "CL:0002079", "bipolar cell": "CL:0000748", "bipolar": "CL:0000748",
    "retinal bipolar cell": "CL:0000748", "clp": "CL:0000051", "gmp": "CL:0000557",
    "chief cell": "CL:0000155", "satellite cell": "CL:0000594", "ens neuron": "CL:0007011",
    "ens glia": "CL:4040002", "enteric glia": "CL:4040002",
    "erythrocyte progenitor": "CL:0000038",
    # today's DC1/DC2 (CD141+ and CD1c+ conventional DCs; the ontology's "DC2" is an older
    # name of plasmacytoid DCs)
    "dc1": "CL:0002394", "dc2": "CL:0002399", "cdc1": "CL:0002394", "cdc2": "CL:0002399",
    "pancreatic islet cell": "CL:0008024", "islet endocrine cell": "CL:0008024",
    # the curated list's own names
    "monocyte cd14 positive": "CL:0001054", "monocyte fcgr3a positive": "CL:0002396",
    "plasmacytoid dc": "CL:0000784", "endothelial arterial": "CL:1000413",
    "endothelial venou": "CL:0002543", "endothelial lymphatic": "CL:0002138",
}


@lru_cache(maxsize=1)
def _cl_index() -> dict[str, str]:
    on = ontology()
    on = on[~on["name"].str.contains(r"\((?:Mmus|Dmel|Hsap|Nfur|Mmul)\)", regex=True)]
    index: dict[str, str] = {}
    for tid, syns in zip(on["id"], on["synonyms"], strict=True):
        for s in str(syns).split("|"):
            if s:
                index.setdefault(_norm(s), tid)
    for tid, name in zip(on["id"], on["name"], strict=True):
        index[_norm(name)] = tid  # a term's own name wins over another's synonym
    index.update(_CL_NAMES)
    return index


# Words a term's name carries that a source's name of the same type usually leaves out.
_FILLER = {"cell", "positive", "alpha", "beta", "thymus", "derived", "mature", "of", "the", "and",
           "human", "mammalian", "lymphocyte", "type"}


def _core(normalized: str) -> frozenset[str]:
    return frozenset(w for w in normalized.split() if w not in _FILLER)


@lru_cache(maxsize=1)
def _cl_cores() -> list[tuple[frozenset[str], int, str]]:
    """Every term name and exact synonym as its core words, with its length in words."""
    return [(_core(n), len(n.split()), t) for n, t in _cl_index().items()
            if _core(n) and t not in _TOO_GENERAL]


@cache
def match_cl(name: str) -> str:
    """The Cell Ontology id of a cell type name (``""`` when none matches).

    The name as written (or an exact synonym); else the term whose words all
    appear in the name, the one with most such words ("CD4 naive T cell" ->
    naive thymus-derived CD4-positive, alpha-beta T cell; "Cornea monocytes" ->
    monocyte), words such as "alpha-beta" or "positive" aside: a term of the
    right lineage, never one too general to name a cell type by."""
    n = _norm(name)
    idx = _cl_index()
    if n in idx:
        return idx[n]
    words = _core(n)
    best: tuple[int, int, str] | None = None
    for core, length, term in _cl_cores():
        if core <= words:
            key = (len(core), -length, term)
            if best is None or key[:2] > best[:2]:
                best = key
    return best[2] if best else ""


@cache
def exact_cl(name: str) -> str:
    """The Cell Ontology id a name is the name (or an exact synonym) of, else ``""``."""
    return _cl_index().get(_norm(name), "")


@lru_cache(maxsize=1)
def _cl_terms() -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    on = ontology()
    names = dict(zip(on["id"], on["name"], strict=True))
    parents = {t: tuple(p for p in str(ps).split("|") if p)
               for t, ps in zip(on["id"], on["parents"], strict=True)}
    return names, parents


def cl_name(cl_id: str) -> str:
    return _cl_terms()[0].get(cl_id, "")


def ancestors(cl_id: str) -> dict[str, int]:
    """Every ancestor of a term with its distance (the term itself at 0)."""
    _, parents = _cl_terms()
    out = {cl_id: 0}
    frontier = [cl_id]
    while frontier:
        nxt = []
        for t in frontier:
            for p in parents.get(t, ()):
                if p not in out:
                    out[p] = out[t] + 1
                    nxt.append(p)
        frontier = nxt
    return out


# Terms too general to name a cluster by.
_TOO_GENERAL = {"CL:0000000", "CL:0000003", "CL:0000255", "CL:0000548", "CL:0002371",
                "CL:0000004", "CL:0000080", "CL:0002319", "CL:0000988", "CL:0000066",
                "CL:0000064", "CL:0000215", "CL:0000183", "CL:0000151", "CL:0000039",
                "CL:0000219", "CL:0000063", "CL:0011026", "CL:0000034", "CL:0000413", "CL:0000412",
                "CL:0002320", "CL:0000393", "CL:0000211", "CL:0000710", "CL:0002321"}


def common_ancestor(a: str, b: str) -> str:
    """The nearest ancestor two terms share, unless it is too general to name a cluster by."""
    if not a or not b:
        return ""
    aa, bb = ancestors(a), ancestors(b)
    shared = [(aa[t] + bb[t], t) for t in aa if t in bb]
    if not shared:
        return ""
    _, best = min(shared)
    return "" if best in _TOO_GENERAL else best


# ---- species ---------------------------------------------------------------

def guess_species(genes: list[str]) -> str:
    """``"mouse"`` when most gene symbols are written as mouse symbols (Cd3e), else ``"human"``."""
    sample = [g for g in genes[:5000] if g[:1].isalpha()]
    if not sample:
        return "human"
    mouse = sum(1 for g in sample if g[0].isupper() and g[1:] != g[1:].upper())
    return "mouse" if mouse > 0.5 * len(sample) else "human"


@lru_cache(maxsize=1)
def _human_to_mouse() -> dict[str, list[str]]:
    """Human gene symbol -> mouse symbols, from the MGI ortholog report (fetched once)."""
    file = cache_dir() / "HOM_MouseHumanSequence.tsv.gz"
    if not file.exists():
        url = "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"
        file.parent.mkdir(parents=True, exist_ok=True)
        print(f"auto_annotate: downloading the MGI human-mouse ortholog table once (CC BY 4.0)"
              f"\n  from {url}\n  to   {file}", file=sys.stderr)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (polariseq)"})
            with urllib.request.urlopen(req, timeout=180) as r:
                t = pd.read_csv(r, sep="\t", usecols=["DB Class Key", "NCBI Taxon ID", "Symbol"])
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"could not download the MGI ortholog table ({type(e).__name__}: {e}), which maps "
                f"human markers to mouse genes. Save {url} as {file.with_suffix('')} "
                "(gzip it to .tsv.gz), or pass mouse markers with markers=.") from None
        with gzip.open(file, "wt") as f:
            t.to_csv(f, sep="\t", index=False)
    t = pd.read_csv(file, sep="\t")
    h = t[t["NCBI Taxon ID"] == 9606].groupby("DB Class Key")["Symbol"].apply(list)
    m = t[t["NCBI Taxon ID"] == 10090].groupby("DB Class Key")["Symbol"].apply(list)
    out: dict[str, list[str]] = {}
    for key in h.index.intersection(m.index):
        for g in h[key]:
            out.setdefault(g, []).extend(m[key])
    return out


def to_species(t: pd.DataFrame, species: str) -> pd.DataFrame:
    """Rows of ``t`` written for ``species``: a source's own rows of that species,
    else its human rows mapped to mouse orthologs."""
    own = t[t["species"].isin([species, "any"])]
    if species == "human" or len(own) == len(t):
        return own
    have = set(own["source"].unique())
    human = t[(t["species"] == "human") & ~t["source"].isin(have)]
    if human.empty:
        return own
    try:
        mapping = _human_to_mouse()
    except RuntimeError as e:
        import warnings
        warnings.warn(f"{e} Mapping by letter case instead (CD3E -> Cd3e).", stacklevel=4)
        mapping = {}
    genes = human["gene"].map(lambda g: mapping.get(g) or [g[:1] + g[1:].lower()])
    mapped = human.assign(gene=genes).explode("gene")
    mapped["species"] = species
    return pd.concat([own, mapped], ignore_index=True)


# ---- the table -------------------------------------------------------------

# Genes no cell type is told by: ribosomal and mitochondrial genes, and the
# immediate-early and heat-shock genes that dissociation switches on in every
# cell (van den Brink et al., Nat. Methods 14, 935, 2017). Databases still list
# some as markers; they are left out of every source (your own markers are kept).
_NOT_MARKERS = re.compile(
    r"^(RP[LS]\d+[A-Z]?\d*$|RPLP\d$|RPSA$|MRP[LS]\d+[A-Z]?\d*$|MT-|MTRNR|"
    r"FOS$|FOSB$|JUN$|JUNB$|JUND$|EGR[1-3]$|ATF3$|IER[2-5]$|ZFP36$|ZFP36L[12]$|NR4A[1-3]$|"
    r"DUSP1$|KLF[246]$|BTG2$|PPP1R15A$|HSPA1[AB]$|HSPA8$|HSPB1$|HSPH1$|HSP90AA1$|DNAJB1$|"
    r"MALAT1$|NEAT1$|XIST$)", re.IGNORECASE)

def marker_table(source: Any = ("curated", "hpa"), *, tissue: Any = None, species: str = "human",
                 markers: Any = None) -> pd.DataFrame:
    """The markers :meth:`tl.auto_annotate` scores with, before matching to a dataset's genes.

    Args:
        source: A source name or a list of them (see ``polariseq._markers.SOURCES``):
            ``"hpa"``, ``"msigdb_c8"``, ``"curated"`` (shipped), ``"accordion"``,
            ``"cellmarker"``, ``"panglaodb"``, ``"sctype"`` (downloaded once).
            ``None`` with ``markers``: only your table.
        tissue: Keep this tissue's markers (a name or a list; see ``TISSUES``). Blood and
            immune cell types (from any study's tissue) and types found in many tissues are
            kept with every tissue.
        species: ``"human"`` or ``"mouse"``.
        markers: Your own markers, added to the source's (see :func:`user_table`).

    Returns:
        One row per marker: ``cell_type``, ``gene``, ``sign``, ``evidence``, ``strength``,
        ``tissue``, ``cl_id``, ``species``, ``source``.
    """
    from polariseq import _checks

    species = str(species).lower()
    if species not in ("human", "mouse"):
        raise _checks.InputError(f"species must be 'human' or 'mouse', not {species!r}")
    names = [] if source is None else [source] if isinstance(source, str) else list(source)
    parts = [_source_table(source_name(s), species) for s in names]
    parts = [p[~p["gene"].str.match(_NOT_MARKERS)] for p in parts]
    if markers is not None:
        parts.append(user_table(markers))
    if not parts:
        raise _checks.InputError("auto_annotate: give a source, markers=, or both")
    t = to_species(pd.concat(parts, ignore_index=True), species)
    if tissue is not None:
        wanted = [tissue] if isinstance(tissue, str) else list(tissue)
        keys = {tissue_key(w) for w in wanted}
        present = set(t["tissue"])
        unknown = [w for w, k in zip(wanted, map(tissue_key, wanted), strict=True)
                   if k not in present]
        if len(unknown) == len(wanted):
            known = sorted(present - {"any"})
            raise _checks.InputError(
                f"auto_annotate: no markers for tissue {_checks.listing(unknown)} in "
                f"{_checks.listing(names or ['markers'])}.{_checks.suggest(str(unknown[0]), known)} "
                f"Tissues there: {_checks.listing(known)}; or leave tissue=None to score them all.")
        blood = t["cl_id"].map(is_hematopoietic)
        t = t[t["tissue"].isin(keys | {"any", "immune"}) | blood.to_numpy(dtype=bool)]
    return t.reset_index(drop=True)


@cache
def is_hematopoietic(cl_id: str) -> bool:
    """Whether a type is a blood or immune cell (descends from "hematopoietic cell"),
    which every tissue holds, whatever tissue a source studied it in."""
    return bool(cl_id) and "CL:0000988" in ancestors(cl_id)
