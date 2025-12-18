from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .genome import (
    create_centered_window,
    get_all_bin_indices,
    get_output_bins_for_interval,
    get_window_sequence,
    is_in_editable_region,
    mutate_sequence,
)


def filter_tasks(
    model,
    assay: Optional[str] = None,
    description: Optional[str] = None,
    name: Optional[str] = None,
) -> tuple[pd.DataFrame, List[int]]:
    """
    Filter model tasks based on case-insensitive contains matching.
    Returns the filtered task dataframe and indices.
    """
    tasks = pd.DataFrame(model.data_params["tasks"])
    mask = pd.Series([True] * len(tasks))

    if assay is not None:
        mask &= tasks["assay"].astype(str).str.lower().str.contains(assay.lower(), na=False)
    if description is not None:
        mask &= tasks["description"].astype(str).str.lower().str.contains(description.lower(), na=False)
    if name is not None:
        mask &= tasks["name"].astype(str).str.lower().str.contains(name.lower(), na=False)

    filtered_tasks = tasks[mask]
    task_indices = filtered_tasks.index.tolist()
    return filtered_tasks, task_indices


def aggregate_predictions(
    preds: np.ndarray,
    task_indices: List[int],
    bin_indices: List[int],
    task_agg: str = "mean",
    length_agg: str = "sum",
) -> float:
    """Aggregate predictions across selected tasks and bins."""
    preds = np.asarray(preds)
    if preds.ndim != 3:
        raise ValueError(f"Expected preds with shape (batch, tasks, bins); got {preds.shape}")

    tasks = task_indices if len(task_indices) > 0 else list(range(preds.shape[1]))
    bins = bin_indices if len(bin_indices) > 0 else list(range(preds.shape[2]))

    selected = preds[0, tasks, :][:, bins]

    if length_agg == "mean":
        length_agg_vals = selected.mean(axis=1)
    elif length_agg == "sum":
        length_agg_vals = selected.sum(axis=1)
    else:
        raise ValueError(f"Unsupported length_agg: {length_agg}")

    if task_agg == "mean":
        return float(length_agg_vals.mean())
    if task_agg == "sum":
        return float(length_agg_vals.sum())
    raise ValueError(f"Unsupported task_agg: {task_agg}")


def predict_on_sequence(model, seq: str, device: str) -> np.ndarray:
    """
    Run model prediction on a single sequence.
    Returns array of shape (1, n_tasks, n_bins).
    Includes CUDA OOM fallback to CPU and clears cache.
    """

    def _run(dev: str) -> np.ndarray:
        with torch.no_grad():
            return model.predict_on_seqs([seq], device=dev)

    try_cuda = device == "cuda"
    if try_cuda and torch.cuda.is_available():
        try:
            preds = _run(device)
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                print("CUDA OOM; retrying on CPU.")
                torch.cuda.empty_cache()
                preds = _run("cpu")
            else:
                raise
        finally:
            torch.cuda.empty_cache()
    else:
        preds = _run("cpu")
    return preds


def analyze_variant(
    variant_row: pd.Series,
    models: Dict,
    gene_info: Dict,
    task_indices: List[int],
    seq_len: int,
    editable_start: int,
    editable_end: int,
    n_output_bins: int,
    task_agg: str,
    length_agg: str,
    device: str,
    genome: str,
    centering: str = "tss",
    fasta_meta: Optional[Dict] = None,
    precomputed_ref_preds: Optional[List[float]] = None,
) -> Dict:
    """
    Analyze a single variant across all model replicates.
    Returns a dict with status, reasons for skips, and predictions when available.
    """
    chrom = variant_row["chrom"]
    snp_pos = variant_row["pos_0based"]
    ref_allele = variant_row["ref"].upper()
    alt_allele = variant_row["alt"].upper()

    center = gene_info["tss"] if centering == "tss" else snp_pos
    window_start, window_end = create_centered_window(center, seq_len)
    snp_pos_in_window = snp_pos - window_start

    result: Dict = {
        "status": "missing",
        "reason": "",
        "centering": centering,
        "window_start": window_start,
        "window_end": window_end,
    }

    if not (0 <= snp_pos_in_window < seq_len):
        result["reason"] = "outside_input_window"
        return result

    if centering == "tss" and not is_in_editable_region(snp_pos_in_window, editable_start, editable_end):
        result["reason"] = "outside_editable_region"
        return result

    gene_exon_bins = get_output_bins_for_interval(
        list(models.values())[0],
        gene_info["exons"],
        window_start,
    )
    all_bins = get_all_bin_indices(gene_exon_bins)
    bin_indices = [b for b in all_bins if 0 <= b < n_output_bins]

    if len(bin_indices) == 0:
        result["reason"] = "no_valid_bins"
        return result

    ref_seq, seq_source = get_window_sequence(
        chrom,
        window_start,
        window_end,
        genome=genome,
        fasta_meta=fasta_meta,
    )

    if len(ref_seq) != seq_len:
        result["reason"] = "sequence_length_mismatch"
        return result

    seq_at_pos = ref_seq[snp_pos_in_window].upper()
    ref_match = seq_at_pos == ref_allele
    alt_seq = mutate_sequence(ref_seq, snp_pos_in_window, alt_allele)

    ref_preds: List[float] = []
    alt_preds: List[float] = []

    for i, (_, model) in enumerate(models.items()):
        if precomputed_ref_preds is not None:
            ref_agg = precomputed_ref_preds[i]
        else:
            ref_pred = predict_on_sequence(model, ref_seq, device=device)
            ref_agg = aggregate_predictions(
                ref_pred,
                task_indices,
                bin_indices,
                task_agg=task_agg,
                length_agg=length_agg,
            )

        alt_pred = predict_on_sequence(model, alt_seq, device=device)
        alt_agg = aggregate_predictions(
            alt_pred,
            task_indices,
            bin_indices,
            task_agg=task_agg,
            length_agg=length_agg,
        )

        ref_preds.append(ref_agg)
        alt_preds.append(alt_agg)

    result.update(
        {
            "status": "success",
            "ref_preds": ref_preds,
            "alt_preds": alt_preds,
            "ref_mean": float(np.mean(ref_preds)),
            "alt_mean": float(np.mean(alt_preds)),
            "ref_match": ref_match,
            "seq_at_pos": seq_at_pos,
            "n_bins": len(bin_indices),
            "seq_source": seq_source,
        }
    )

    return result


def run_full_analysis(
    variants_df: pd.DataFrame,
    models: Dict,
    gene_info: Dict,
    task_indices: List[int],
    seq_len: int,
    editable_start: int,
    editable_end: int,
    n_output_bins: int,
    task_agg: str,
    length_agg: str,
    device: str,
    genome: str,
    fasta_meta: Optional[Dict],
) -> pd.DataFrame:
    """
    Run analysis on all variants with both centering strategies.
    Tracks reasons for missing predictions for later validation.
    """
    results = []

    for _, row in tqdm(variants_df.iterrows(), total=len(variants_df), desc="Analyzing variants"):
        tss_result = analyze_variant(
            row,
            models,
            gene_info,
            task_indices,
            seq_len=seq_len,
            editable_start=editable_start,
            editable_end=editable_end,
            n_output_bins=n_output_bins,
            task_agg=task_agg,
            length_agg=length_agg,
            device=device,
            genome=genome,
            centering="tss",
            fasta_meta=fasta_meta,
        )

        snp_result = analyze_variant(
            row,
            models,
            gene_info,
            task_indices,
            seq_len=seq_len,
            editable_start=editable_start,
            editable_end=editable_end,
            n_output_bins=n_output_bins,
            task_agg=task_agg,
            length_agg=length_agg,
            device=device,
            genome=genome,
            centering="snp",
            fasta_meta=fasta_meta,
        )

        record = {
            "rsID": row["rsID"],
            "chrom": row["chrom"],
            "position": row["pos"],
            "ref_allele": row["ref"],
            "alt_allele": row["alt"],
            "relative_to_tss": row["pos"] - gene_info["tss"],
        }

        record["tss_status"] = tss_result.get("status", "missing")
        record["tss_missing_reason"] = tss_result.get("reason", "") if record["tss_status"] != "success" else ""
        record["tss_seq_source"] = tss_result.get("seq_source", "") if record["tss_status"] == "success" else ""

        if record["tss_status"] == "success":
            record["tss_ref_exp"] = tss_result["ref_mean"]
            record["tss_alt_exp"] = tss_result["alt_mean"]
            record["tss_ref_match"] = tss_result["ref_match"]
        else:
            record["tss_ref_exp"] = np.nan
            record["tss_alt_exp"] = np.nan
            record["tss_ref_match"] = np.nan

        record["snp_status"] = snp_result.get("status", "missing")
        record["snp_missing_reason"] = snp_result.get("reason", "") if record["snp_status"] != "success" else ""
        record["snp_seq_source"] = snp_result.get("seq_source", "") if record["snp_status"] == "success" else ""

        if record["snp_status"] == "success":
            record["snp_ref_exp"] = snp_result["ref_mean"]
            record["snp_alt_exp"] = snp_result["alt_mean"]
            record["snp_ref_match"] = snp_result["ref_match"]
        else:
            record["snp_ref_exp"] = np.nan
            record["snp_alt_exp"] = np.nan
            record["snp_ref_match"] = np.nan

        results.append(record)

    results_df = pd.DataFrame(results)
    results_df["ref_gene_exp_level"] = results_df[["tss_ref_exp", "snp_ref_exp"]].mean(axis=1)
    results_df["alt_gene_exp_level"] = results_df[["tss_alt_exp", "snp_alt_exp"]].mean(axis=1)
    results_df["diff"] = results_df["alt_gene_exp_level"] - results_df["ref_gene_exp_level"]

    return results_df
