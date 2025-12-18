from __future__ import annotations

import gzip
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# Simple directory helper
def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist and return the Path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_gtf_if_needed(cache_dir: Path, genome: str = "hg38") -> Path:
    """
    Download the GENCODE GTF file if it is not already available locally.
    Returns the path to the GTF file.
    """
    cache_dir = ensure_dir(Path(cache_dir))
    gtf_path = cache_dir / f"{genome}_gencode.gtf.gz"

    if gtf_path.exists():
        print(f"GTF file already exists: {gtf_path}")
        return gtf_path

    print("Downloading GENCODE GTF file...")
    url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz"

    import urllib.request

    urllib.request.urlretrieve(url, gtf_path)
    print(f"Downloaded GTF to: {gtf_path}")
    return gtf_path


def _parse_gtf_for_gene(gtf_path: Path, gene_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stream through the GTF and retain only entries for the requested gene.
    Returns (genes_df, exons_df) with 0-based coordinates and TSS calculated.
    """
    genes = []
    exons = []
    gene_key = gene_name.upper()
    opener = gzip.open if str(gtf_path).endswith(".gz") else open

    with opener(gtf_path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue

            chrom, source, feature, start, end, score, strand, frame, attributes = fields

            attr_dict: Dict[str, str] = {}
            for attr in attributes.split(";"):
                attr = attr.strip()
                if not attr:
                    continue
                parts = attr.split(" ", 1)
                if len(parts) == 2:
                    key, value = parts
                    attr_dict[key] = value.strip('"')

            attr_gene_name = attr_dict.get("gene_name", "")
            if attr_gene_name.upper() != gene_key:
                continue

            if feature == "gene":
                genes.append(
                    {
                        "chrom": chrom,
                        "start": int(start) - 1,
                        "end": int(end),
                        "gene_name": attr_gene_name,
                        "gene_type": attr_dict.get("gene_type", ""),
                        "strand": strand,
                    }
                )
            elif feature == "exon":
                exons.append(
                    {
                        "chrom": chrom,
                        "start": int(start) - 1,
                        "end": int(end),
                        "gene_name": attr_gene_name,
                        "strand": strand,
                    }
                )

    genes_df = pd.DataFrame(genes)
    exons_df = pd.DataFrame(exons)

    if genes_df.empty:
        raise ValueError(f"Gene '{gene_name}' not found in annotations at {gtf_path}")

    genes_df["tss"] = genes_df.apply(
        lambda row: row["start"] if row["strand"] == "+" else row["end"] - 1,
        axis=1,
    )

    return genes_df, exons_df


def load_gene_annotations_for_gene(
    gtf_path: Path,
    gene_name: str,
    shortcut_dir: Path,
):
    """
    Load gene annotations for a single gene, preferring cached shortcuts.
    Returns (genes_df, exons_df, used_shortcut).
    """
    shortcut_dir = ensure_dir(Path(shortcut_dir))
    gene_file = shortcut_dir / f"{gene_name}_gene_annotations.csv"
    exon_file = shortcut_dir / f"{gene_name}_exon_annotations.csv"

    if gene_file.exists() and exon_file.exists():
        print(f"Loading cached annotations for {gene_name} from {shortcut_dir}")
        genes_df = pd.read_csv(gene_file)
        exons_df = pd.read_csv(exon_file)
        return genes_df, exons_df, True

    print(f"No cached annotations for {gene_name}. Parsing full GTF...")
    genes_df, exons_df = _parse_gtf_for_gene(gtf_path, gene_name)

    genes_df.to_csv(gene_file, index=False)
    exons_df.to_csv(exon_file, index=False)
    print(f"Saved shortcuts: {gene_file.name}, {exon_file.name}")

    return genes_df, exons_df, False


def get_gene_info(gene_name: str, genes_df: pd.DataFrame, exons_df: pd.DataFrame) -> Dict:
    """
    Extract gene metadata and exon coordinates for the requested gene.
    """
    mask = genes_df["gene_name"].str.upper() == gene_name.upper()
    if not mask.any():
        raise ValueError(f"Gene '{gene_name}' not found in provided annotations")

    gene_info = genes_df[mask].iloc[0]
    gene_exons = (
        exons_df[exons_df["gene_name"].str.upper() == gene_name.upper()]
        .drop_duplicates(subset=["chrom", "start", "end"])
        .reset_index(drop=True)
    )

    result = {
        "gene_name": gene_info["gene_name"],
        "chrom": gene_info["chrom"],
        "start": int(gene_info["start"]),
        "end": int(gene_info["end"]),
        "strand": gene_info["strand"],
        "tss": int(gene_info["tss"]),
        "exons": gene_exons[["chrom", "start", "end"]].copy(),
    }

    return result


def load_vcf(vcf_path: str) -> pd.DataFrame:
    """
    Load VCF file into a DataFrame.
    Handles standard VCF format.
    """
    vcf_path = str(vcf_path)
    if not Path(vcf_path).exists():
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")

    opener = gzip.open if vcf_path.endswith(".gz") else open

    # Find header line to get column names
    header_line = None
    skip_rows = 0
    with opener(vcf_path, "rt") as handle:
        for line in handle:
            if line.startswith("##"):
                skip_rows += 1
            elif line.startswith("#CHROM"):
                header_line = line.strip().lstrip("#").split("\t")
                skip_rows += 1
                break
            else:
                break

    if header_line is None:
        header_line = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]

    df = pd.read_csv(
        vcf_path,
        sep="\t",
        comment="#",
        header=None,
        names=header_line[:8],
        usecols=range(min(8, len(header_line))),
    )

    df = df.rename(
        columns={"CHROM": "chrom", "POS": "pos", "ID": "rsID", "REF": "ref", "ALT": "alt"}
    )

    if not df.empty and not df["chrom"].iloc[0].startswith("chr"):
        df["chrom"] = "chr" + df["chrom"].astype(str)

    df["pos_0based"] = df["pos"] - 1

    is_snp = (df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)
    n_before = len(df)
    df = df[is_snp].reset_index(drop=True)
    print(f"Filtered {n_before - len(df)} non-SNP variants, {len(df)} SNPs remaining")
    return df
