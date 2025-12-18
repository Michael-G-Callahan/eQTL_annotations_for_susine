from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .data_ingestion import ensure_dir

FASTA_CACHE: Dict[str, str] = {}


def create_centered_window(center_pos: int, seq_len: int) -> Tuple[int, int]:
    """Create a 0-based window of length seq_len centered on center_pos."""
    half_len = seq_len // 2
    start = center_pos - half_len
    end = start + seq_len
    return start, end


def position_in_window(pos: int, window_start: int, window_end: int) -> int:
    """Return the position of pos within [window_start, window_end), or -1 if outside."""
    if window_start <= pos < window_end:
        return pos - window_start
    return -1


def is_in_editable_region(pos_in_window: int, editable_start: int, editable_end: int) -> bool:
    """Check whether the position falls inside the editable region."""
    return editable_start <= pos_in_window < editable_end


def get_output_bins_for_interval(model, interval_df: pd.DataFrame, window_start: int) -> pd.DataFrame:
    """Wrapper for model.input_intervals_to_output_bins."""
    return model.input_intervals_to_output_bins(intervals=interval_df, start_pos=window_start)


def validate_bins_in_output(bin_df: pd.DataFrame, n_output_bins: int) -> Tuple[bool, str]:
    """Ensure all bins fall inside the model output range."""
    if bin_df is None or len(bin_df) == 0:
        return False, "No bins available for validation"

    min_bin = bin_df["start"].min()
    max_bin = bin_df["end"].max()

    if min_bin < 0:
        return False, f"Bins extend before output start (min_bin={min_bin})"
    if max_bin > n_output_bins:
        return False, f"Bins extend beyond output end (max_bin={max_bin}, n_output={n_output_bins})"
    return True, f"All bins valid (range: {min_bin} - {max_bin}, output size: {n_output_bins})"


def get_all_bin_indices(bin_df: pd.DataFrame) -> List[int]:
    """Expand bin intervals into a sorted list of unique bin indices."""
    if bin_df is None or len(bin_df) == 0:
        return []
    if not {"start", "end"}.issubset(bin_df.columns):
        missing = {"start", "end"} - set(bin_df.columns)
        raise ValueError(f"Bin DataFrame missing columns: {', '.join(sorted(missing))}")

    bin_indices: List[int] = []
    for _, row in bin_df.iterrows():
        if pd.isna(row["start"]) or pd.isna(row["end"]):
            continue
        start = int(row["start"])
        end = int(row["end"])
        if end <= start:
            continue
        bin_indices.extend(range(start, end))
    return sorted(set(bin_indices))


def mutate_sequence(seq: str, pos: int, new_base: str) -> str:
    """Replace the base at `pos` with `new_base`."""
    return seq[:pos] + new_base + seq[pos + 1 :]


def _fetch_sequence_from_genome(
    chrom: str,
    start: int,
    end: int,
    genome: str = "hg38",
) -> str:
    """Fetch sequence directly from the reference genome via grelu."""
    import grelu.sequence.format

    interval_df = pd.DataFrame({"chrom": [chrom], "start": [start], "end": [end], "strand": ["+"]})
    seqs = grelu.sequence.format.convert_input_type(interval_df, output_type="strings", genome=genome)
    return seqs[0]


def read_fasta_sequence(fasta_path: Path) -> str:
    """Load FASTA sequence from disk, cached in memory for reuse."""
    key = str(Path(fasta_path).resolve())
    if key in FASTA_CACHE:
        return FASTA_CACHE[key]

    seq_parts: List[str] = []
    with open(fasta_path, "r") as handle:
        for line in handle:
            if line.startswith(">"):
                continue
            seq_parts.append(line.strip())

    sequence = "".join(seq_parts)
    FASTA_CACHE[key] = sequence
    return sequence


def ensure_fasta_shortcut(
    chrom: str,
    tss: int,
    gene_name: str,
    shortcut_dir: Path,
    genome: str = "hg38",
    flank_radius: int = 1_000_000,
) -> Dict:
    """
    Build (or load) a fasta snippet surrounding the TSS for reuse.
    Returns metadata describing the cached region.
    """
    shortcut_dir = ensure_dir(Path(shortcut_dir))
    fasta_path = shortcut_dir / f"{gene_name}_region_seq.fasta"
    meta_path = shortcut_dir / f"{gene_name}_region_seq.json"

    if fasta_path.exists() and meta_path.exists():
        with open(meta_path, "r") as handle:
            meta = json.load(handle)
        meta["path"] = str(fasta_path)
        print(f"Using cached FASTA shortcut for {gene_name}: {fasta_path}")
        return meta

    start = max(0, int(tss) - flank_radius)
    end = int(tss) + flank_radius
    sequence = _fetch_sequence_from_genome(chrom, start, end, genome=genome)

    header = f">{gene_name}_tss_flank_{start}_{end}"
    with open(fasta_path, "w") as fasta_handle:
        fasta_handle.write(f"{header}\n")
        for i in range(0, len(sequence), 80):
            fasta_handle.write(sequence[i : i + 80] + "\n")

    meta = {"chrom": chrom, "start": start, "end": end, "tss": int(tss), "radius": flank_radius, "path": str(fasta_path)}
    with open(meta_path, "w") as meta_handle:
        json.dump(meta, meta_handle, indent=2)

    print(f"Saved FASTA shortcut for {gene_name} to {fasta_path}")
    return meta


def get_window_sequence(
    chrom: str,
    start: int,
    end: int,
    genome: str = "hg38",
    fasta_meta: Optional[Dict] = None,
) -> Tuple[str, str]:
    """
    Retrieve a sequence for [start, end) either from a cached FASTA shortcut
    or directly from the reference genome. Returns (sequence, source_tag).
    """
    if fasta_meta and fasta_meta.get("chrom") == chrom:
        cached_start = int(fasta_meta.get("start", -1))
        cached_end = int(fasta_meta.get("end", -1))
        fasta_path = fasta_meta.get("path")

        if fasta_path and cached_start <= start and end <= cached_end:
            fasta_seq = read_fasta_sequence(Path(fasta_path))
            offset_start = start - cached_start
            offset_end = end - cached_start
            return fasta_seq[offset_start:offset_end], "shortcut"

    return _fetch_sequence_from_genome(chrom, start, end, genome=genome), "genome"
