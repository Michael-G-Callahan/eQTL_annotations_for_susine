"""
Utility package for the Borzoi variant effect notebook.

Modules:
- data_ingestion: GTF/VCF loading and shortcut management.
- genome: sequence fetching, windows, bins, and FASTA shortcuts.
- prediction: task filtering, aggregation, and variant analysis loops.
"""

from . import data_ingestion, genome, prediction

__all__ = ["data_ingestion", "genome", "prediction"]
