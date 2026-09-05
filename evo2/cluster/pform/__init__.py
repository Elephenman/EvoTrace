"""pform — protein final-state predictor CLI (EvoTrace cluster toolkit).

End-to-end: input Boltz yamls -> CHPC GPU prediction -> per-model metrics ->
final-state verdict (structure + metrics + pass/fail). Protein-agnostic;
metrics are pluggable (default: dual-lock contact geometry).
"""
__version__ = "0.1.0"
