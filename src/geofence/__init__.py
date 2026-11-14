"""Geo-Fenced Sleepers: context-conditional memory poisoning and CCS detection.

Package modules:
  common      paths, seeding, JSONL checkpointing, memguard scale helpers
  data_prep   build user mobility distributions Q, contexts, POIs, legit conditionals
  attack      construct sleepers / unconditional / legitimate-conditional entries
  agent       location-aware agent harness (Ollama / HF backends) + mocked tools + OSRM
  scope       provenance-derived scope model + out-of-scope divergence
  defenses    content filters + provenance gating baselines
  stats       SPRT, sample-complexity, Wilson CI, calibration  (pure math, unit-tested)
  ccs         Counterfactual Context Sampling detector
  evaluate    experiment driver (stealth, ASR, detection, FPR, cost)
  aggregate   results/summary.json
  plots       figures
"""
