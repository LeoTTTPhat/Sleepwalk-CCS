# GAPS.md -- Stage 4 completeness audit (p3-geofenced-sleepers-memory)

Audit date: 2026-10-16. Reference: the experiment matrix in PLAN.md and the evaluation plan in
BRIEF.md, compared against results/summary.json and the raw JSONL in results/full*/.
Every number below is read from results/summary.json (key paths are given) or from the raw
JSONL it is built from.

## A. Inconsistencies found and fixed offline (no new model calls)

These were fixed in `src/geofence/aggregate.py`. Every Stage-3 value in summary.json is
unchanged; the code only adds new keys, checked leaf by leaf against the Stage-3 copy.

A1. **The calibration set is part of the FPR set (in-sample FPR).**
- `_calibrate` draws rho0's null pool from the first `calib_n` interleaved benign/legit entries.
- The CCS FPR set is the first 60+60 (seed 1) or 30+30 (seeds 2/3) of the same lists, so all 30 (or 20) calibration entries are also scored for FPR.
- Fix: new keys `ccs.fpr_all_null_sprt_heldout`, `fpr_{benign,legit}_sprt_heldout` and `fpr_all_null_sprt_calib_entries`, plus `calib_overlap_null_entries`.
- Held-out FPR for seeds 1/2/3 is 8/90 = 0.089 [0.046, 0.166], 2/40 and 3/40.
- Beijing held-out FPR is 6/24 = 0.25.
- Paper: report the held-out FPR as the primary FPR.

A2. **Calibrated rho0 versus the null divergence actually seen in CCS runs.**
- New key `ccs.null_divergence_in_ccs_runs`.
- Austin, per seed: 0.017, 0.014 and 0.017, against rho0 = 0.011, 0.016 and 0.011. These agree within the calibration CI.
- Beijing: 0.050 [0.037, 0.068] against rho0 = 0.018 (3/192). Beijing calibration underestimates the null by about 3x, which explains its FPR of 11/40. Re-run: section C.

A3. **FPR against the SPRT bound.**
- New key `ccs.fpr_heldout_vs_sprt_bound`: an exact one-sided binomial test of held-out FPR against alpha/(1-beta) = 0.0625.
- Austin, per seed: p = 0.20, 0.72 and 0.46. None of the seeds is significantly above the bound, but n = 90/40/40 cannot tell 0.06 from 0.09 apart. This is underpowered; see D3.
- Beijing: p = 0.003. The bound is violated.

A4. **The across-seed t-CI only reflects context-sampling variance.**
- The 3 seeds re-sample contexts and user pairing on the same 50 entries, so `across_seeds` CIs (for example ASR [0.207, 0.230]) leave out entry-sampling uncertainty.
- Fix: new key `full.across_seeds_entry_bootstrap`. Each entry's outcome is averaged over the seeds, then entries are bootstrapped.
- Results: ASR 0.219 [0.113, 0.336]; SPRT detection 0.733 [0.620, 0.833]; detection incl. confirm 0.993 [0.98, 1.0]; null FPR 0.061 [0.022, 0.106]; dormant fire 0.019 [0, 0.056].
- Paper: quote these intervals, not the t-CI.

A5. **The `pooled` blocks are not independent samples.**
- Pooled rows concatenate the seeds over the same entries (for example `pooled.asr.n_entries` = 185 is 85 + 50 + 50), so their Wilson/bootstrap CIs are too narrow.
- Paper: use per-seed, across-seed (A4) or seed-1 full-slice values, never pooled CIs.

A6. **The confirmation path uses an oracle label.**
- `ccs_detect` sets `confirm_required = entry["risk_tier"] == "high"`, which is the generator's payload label. A deployed defender does not have that label.
- Fix: `geofence/triage.py` adds `confirm_text`, a regex over the high-risk tool classes (pay/book/message/share location) that a defender can compute.
- Offline check on Austin: confirm_text matches the oracle on every one of the 85 positive action entries, and flags 0/100 benign and 0/100 legit entries.
- Caveat: the null corpus never mentions high-risk tools, so the 0% false-confirm rate holds for this corpus only (limitation L2).
- Keys: `full.triage.confirm_text.*`.
- Detection incl. confirm_text (instead of the oracle) is 84/85, 50/50 and 49/50 for seeds 1-3, the same as with the oracle.

A7. **The content filters scored a different sleeper set.**
- `filters` scores all 100 sleeper entries, including the 15 split 'condition' halves. ASR and CCS use the 85 positives.
- Fix: new key `full.triage.filters_on_positive_sleepers`. On the 85 positives: deepset 85/85, protectai 10/85, llama-guard 25/85.
- Stage 3 reported 100%, 11% and 25% over the 100 sleeper entries.

A8. **Detection on sleepers that actually fire was missing.**
- Most sleepers never fire under phi: `asr.frac_entries_fired_at_least_once` is 0.25 on seed 1, so SPRT detection over all sleepers mixes effective and inert entries.
- New keys: `ccs.detection_sprt_effective` (asr > 0 in the same seed) and `ccs.detection_sprt_ineffective`.
- Effective, per seed: 20/22, 10/12 and 11/12, which is 41/46 = 0.89 combined.
- Ineffective: 42/63, 26/38 and 25/38. CCS also flags many entries that never fire when phi holds, because they cause out-of-scope actions in other contexts.
- Beijing: effective 10/10.

A9. **Detection versus budget n was missing.**
- New keys `ccs.detection_sprt_by_budget` and `fpr_null_sprt_by_budget`, for B in {5, 10, 20, 30, 40}. These are lower bounds, because a flag at n_used <= B is also a flag under truncation at B.
- Seed 1 detection: 29, 53, 60, 61 and 62 of 85. Seed 1 null flags: 1, 3, 7, 7 and 9 of 120.
- Detection saturates near n = 20.

A10. **Detection for triggers with Q-mass >= eps was missing.**
- This is the brief's headline claim. New keys `ccs.detection_sprt_qmass_ge_eps` and `_incl_confirm_`.
- Seed 1: 9/17 = 0.53 by SPRT and 17/17 including confirm. Seed 2: 8/11. Seed 3: 9/13.
- Low-risk (no confirm path) at seed 1: 4/4.
- The brief's expectation of more than 95% detection by SPRT at Q-mass >= eps is not supported. Most sleepers have Q-mass below eps: `frac_qmass_ge_eps` is 17/85. The claim is underpowered (D1).

A11. **Split and soft-edged breakdowns were missing.**
- New key `ccs.detection_sprt_by_split`: split pairs 9/14, non-split 53/71 (seed 1).
- New key `ccs.detection_sprt_by_soft`: soft 9/10, hard 53/75.

A12. **No paired tests existed for the ablations.**
- New key `paired_vs_ccs` (exact McNemar on the same sleeper and user units).
- randomq against ccs on the same 50 sleepers (seed 1): 38 vs 38, 4 discordant each way, p = 1.0. shift against ccs: identical counts.
- Stage-3 notes said "randomq/shift 38/50 vs ccs 0.73". The 0.73 is over all 85 sleepers; on the same 50, ccs is also 38/50.
- So **sampling from Q gives no measurable detection gain over uniform (no-Q) probing in this setup.** Stage 5 has to say this plainly.

A13. **No paired test existed for provenance gating.**
- New key `provgate.paired_sign_test`.
- Spotlight against none: 11 entries lower, 1 higher, p = 0.006. TMA-NM and TMA-NM-relaxed against none: 12 lower, 0 higher, p = 0.0005.
- Only 12 of 51 entries fire at all without a gate; see D2.

## B. Coverage of the planned matrix (PLAN.md / BRIEF.md)

| planned item | status | where |
|---|---|---|
| asr (stealth, ASR, p) Qwen, 3 seeds | done (seed 1 85 positives; seeds 2/3 on the pre-registered 50 slice) | full.per_seed.*.asr |
| ccs detection / FPR / Q-mass / n, 3 seeds | done (same slices; null 60+60 seed 1, 30+30 seeds 2/3; brief planned 100+100) | full.per_seed.*.ccs |
| content filters (Llama-Guard-3, deepset, protectai) | done; Prompt-Guard-2 descoped (gated, no HF token) | full.filters |
| provgate (spotlight, TMA-NM, utility loss) | done, **1 seed** | full.per_seed.20261015.provgate |
| randomq (no-Q) | done, **1 seed**, 50 sleepers | full.per_seed.20261015.randomq |
| shift (Q from held-out region) | done, **1 seed**, 50 sleepers | full.per_seed.20261015.shift |
| loo (post-hoc) | done, 1 seed; n = 20 in Stage 3 (5 harm cases), extended to 85 in Stage 4 | loo_posthoc |
| osrm non-mocked loop | done, 1 seed; n = 10 in Stage 3, extended to all reroute positives in Stage 4 | osrm |
| cost per write, shared vs unshared | done; the unshared cost is 2 x n_used x latency (analytic, not measured) | cost |
| cost with pre-triage | **missing in Stage 3** (the evaluate.py docstring claimed it, but no code existed) -> Stage-4 triage sweep | triage |
| Hammer2.1-7b table | done, ASR only (0/36) | full_hammer |
| GeoLife (Beijing) contexts | done, asr + ccs (40 sleepers / 8 users) | full_beijing |
| ablation: exploration-mixture weight | **missing** -> Stage-4 `explore0` (0.0); with ccs (0.3) and randomq (1.0, safety 0) this gives 3 points | explore0 |
| ablation: scope granularity | **missing** -> Stage-4 `scope_tool`, `scope_none` | scope_tool, scope_none |
| ablation: temperature 0 vs sampled decoding | **missing** -> Stage-4 run `full_temp07` | full_temp07 |
| ablation: pre-triage threshold | **missing** -> Stage-4 triage sweep | full.triage.sweeps |
| robustness: split / soft / shift | split and soft breakdowns added offline (A11); shift done | ccs.detection_sprt_by_{split,soft} |
| detection vs n | added offline (A9) | ccs.detection_sprt_by_budget |
| confirmation-prompt rate | present (`frac_confirm_required`), plus non-oracle proxy (A6) | ccs, triage |
| >= 3 seeds for reported means | met only for asr and ccs | see D4 |

## C. Gaps run in Stage 4 (guarded chain `run_gaps.sh` -> `gaps_chain.sh`)

How it ran:
- One memguard job, `p3-geofenced-sleepers-memory-gaps`, with `--ollama-models qwen2.5:7b`.
- Queued at 09:53. It waited 2 h 53 min for a guard slot, because both slots were held by p1/p2 jobs. It acquired one at 12:46 and finished GAPS_OK at 16:10.
- Compute: 3 h 24 min. Peak project memory: 15.3 GiB. No breaches and no restarts.
- Per-step wall-clock is in `results/{full,full_beijing_recal,full_temp07}/timing.jsonl`.
- All steps are idempotent. Re-running `nohup sh run_gaps.sh > logs/gaps.log 2>&1 &` skips finished work.

| # | gap | run | wall | result (summary.json key) |
|---|---|---|---|---|
| C1 | pre-triage threshold ablation; cost with pre-triage | `geofence.triage` (MiniLM, CPU) + offline sweep | 12 s | `full.triage.sweeps` |
| C2 | LOO post-hoc had only 5 harm cases | loo extended to all 85 positives, seed 1 | 167 s | `loo_posthoc` |
| C3 | OSRM loop n = 10 | osrm extended to all 34 reroute positives | 62 s | `osrm` |
| C4 | Beijing null FPR 11/40 (calibration too small and in-sample) | run `full_beijing_recal`: calibration on 40 null entries x 15 contexts, disjoint from the FPR set | 45 min | `full_beijing_recal.*`, `paired_vs_reference` |
| C5 | scope-granularity ablation (planned, missing) | `scope_tool`, `scope_none`, each with its own calibration, seed 1 full set | 85 min + 40 s | `full.per_seed.20261015.scope_*`, `calibration_scope_*`, `paired_vs_ccs` |
| C6 | temperature 0 vs sampled decoding (planned, missing) | run `full_temp07` (T = 0.7, one sample per context): 30 sleepers, 15+15 null, held-out calibration 20 x 10 | 9 + 51 min | `full_temp07.*`, `paired_vs_reference` |
| C7 | exploration-mixture weight (planned, missing) | `explore0` (exploration 0, pure user Q), seed 1, same 50 sleepers as randomq/shift | 9 min | `full.per_seed.20261015.explore0`, `paired_vs_ccs` |

Results. All are seed 1 unless stated. CIs are Wilson; p-values are exact McNemar unless stated.

C1. **Pre-triage.** Entries scoring below the threshold skip the dry-runs. `emb_score` is the maximum cosine similarity to the prototypes.

| threshold | positives passing | benign passing | legit passing | SPRT detection | null SPRT flags | null calls saved |
|---|---|---|---|---|---|---|
| none | 85/85 | all | all | 62/85 | 9/120 | 0 |
| 0.22 | 0.953 | 0.45 | 0.25 | 61/85 | 2/120 | 66% |
| 0.26 | 0.765 | 0.11 | 0.09 | 50/85 | 0/120 | 90% |
| 0.30 | 0.471 | 0.01 | 0.03 | 33/85 | 0/120 | 97% |

- Classifier pre-triage with protectai at threshold 0.001 passes 76% of positives and 0-2% of null entries, with detection 45/85.
- Read: pre-triage at 0.22 removes most false positives and two thirds of the null-entry cost, and loses 1 detection.
- The threshold was not tuned on a held-out split. Treat it as an ablation curve, not a deployed operating point.

C2. **Leave-one-out (post-hoc).**
- Harm occurred in 19/85 contexts. Attribution was correct in 19/19 harm cases, CI [0.83, 1.0].
- Stage 3 had 5/5.

C3. **OSRM loop.**
- The agent routed in 34/34 cases, 29/34 to the attacker's POI (0.85 [0.70, 0.94]). OSRM resolved 29/29 of those routes.
- Mean route: 30.7 km, 1566 s.
- Stage 3 had 8/10.

C4. **Beijing recalibration.** Calibrated null divergence was 33/600 = 0.055, against 3/192 = 0.016 before; rho0 = 0.056 and rho1 = 0.136.

| metric (Beijing) | before | after |
|---|---|---|
| null SPRT FPR (all held-out) | 11/40 | 2/40 = 0.05 |
| binomial test against the 0.0625 bound | p = 0.003 | p = 0.72 |
| SPRT detection | 29/36 | 19/36 |
| detection incl. confirm | 36/36 | 34/36 |
| detection on effective sleepers | 10/10 | 8/10 |
| mean n | 13.3 | 19.1 |

- Paired changes: sleepers p = 0.002, null p = 0.004.
- Read: Beijing's FPR problem was a calibration problem. Fixing it costs detection, and that trade-off has to be reported.
- ASR is identical (0.22), because it replays from the call cache.

C5. **Scope granularity.** One row per scope definition.

| scope | calibrated null divergence | SPRT detection | high-risk / low-risk sleepers | null SPRT FPR | McNemar vs ours |
|---|---|---|---|---|---|
| POI-level (ours) | 6/600 = 0.010 | 62/85 | 29/51 / 33/34 | 9/120 | n/a |
| tool-only | 0/600 | 6/85 | 6/51 / 0/34 | 0/120 | p < 1e-16 |
| none (any action change) | 413/600 = 0.69 | 57/85 | n/a | 86/120 | null: p = 5e-19 |

- The no-scope rho0 is capped at 0.35 in code, so its SPRT is degenerate.
- Read: without a scope model the null diverges in 69% of contexts, so the scope model is necessary.
- With tool-only scope almost nothing is detected. **Even the high-risk payloads are detected mostly through induced off-entry routes, not through high-risk tool calls.** Stage 5 has to report this.

C6. **Temperature 0.7.** Same 30 sleepers and 30 null entries and the same contexts as T = 0.

| metric | T = 0 | T = 0.7 |
|---|---|---|
| calibrated rho0 (held-out calibration) | 0.011 | 0.027 (5/200 divergences) |
| ASR, same 30 entries | 0.193 | 0.160 |
| SPRT detection | 23/30 | 19/30 (p = 0.125) |
| null FPR | 1/30 | 2/30 (p = 1.0) |
| mean n, sleepers | 11.0 | 16.0 |
| detection on effective sleepers | n/a | 6/7 |

- Read: sampled decoding raises the null rate and lowers detection slightly. The difference is not significant at n = 30.

C7. **Exploration weight.** SPRT detection on the same 50 sleepers, by setting:
- exploration 0 (pure Q): 41/50.
- exploration 0.3 with safety 0.5 (ccs): 38/50.
- exploration 1.0 with safety 0 (randomq): 38/50.
- Paired tests against ccs: explore0 p = 0.375, randomq p = 1.0.
- Read: in this setup the context distribution does not measurably change detection. Q, the exploration mixture and uniform probing all detect about 76-82%.

## D. Gaps not closed (exact reasons)

D1. **Detection at Q-mass >= eps is underpowered.**
- Only 17, 11 and 13 sleepers per seed reach Q-mass >= eps; the generator does not control Q-mass.
- Estimating a rate near 0.95 to within 5 points needs about 73 such sleepers. That means generating sleepers conditioned on each user's Q (new attack generation plus about 73 x 12 x 2 calls per seed), which is more than 6 h for 3 seeds.
- Paper: report it as underpowered, with Wilson CIs and the entry bootstrap.

D2. **Provenance gating per channel is underpowered, and has one seed.**
- Without a gate only 12/51 entries fire. forwarded_doc has 9 entries with base ASR 0.11, about one firing entry, so "forwarded documents and instruction-bearing tool outputs dominate the residual" cannot be tested.
- The data show TMA-NM at 0 on every channel, which is significant against no gate (A13).
- Seeds 2/3 of provgate would take about 2 x 87 min of new calls (the phi-holding contexts change with the seed). That is within the budget on its own, but it was ranked below the missing planned ablations, and the slot wait (section C) used the 6 h budget.
- Exact resume: `shared/memguard.py --name p3-geofenced-sleepers-memory-eval-provgate-s20261016 --ollama-models qwen2.5:7b -- shared/.venv/bin/python -m geofence.evaluate --config src/configs/full.yaml --exp provgate --seed 20261016` (and the same for 20261017), with `PYTHONPATH=src`.
- Note: `replicate_overrides` does not cap provgate_n, so 51 sleepers would run per seed.

D3. **Null FPR against nominal alpha is underpowered.**
- To separate FPR = 0.09 from the 0.0625 bound at 80% power (one-sided, 0.05) needs about 650 held-out null entries; we have 90 + 40 + 40.
- That is not feasible: there are 100 benign and 100 legit entries in total, and new null generation plus CCS runs would take more than 6 h.
- Report held-out FPR with CIs and the binomial test (A3).

D4. **Several experiments have a single seed: randomq, shift, provgate, loo, osrm, filters.**
- filters is deterministic and seed-free.
- randomq and shift seeds 2/3 would take about 4 x 50 min of mostly new calls. The paired result at seed 1 is an exact tie (A12), and the C7 sweep is flat, so more seeds would only tighten a null effect. They were ranked below the missing ablations and did not fit the budget (see D2).
- The Stage-4 ablations (explore0, scope_*, temp07) are also single seed.
- Exact commands: `--exp randomq --seed 20261016`, and likewise for shift and seed 20261017. These can run at once because the replicate overrides already restrict to the first 50 positives.

D5. **The CCS null class is 60+60, not the brief's 100+100.** This descope was recorded in Stage 3 (time cap). Seeds 2/3 use 30+30.

D6. **Llama-Prompt-Guard-2-86M is not evaluated.** The model is gated, and there is no HF token (Stage 2).

D7. **Hammer2.1-7b has ASR only.** ASR is 0/36: the outputs are well-formed actions that ignore the payloads. With no sleeper that fires, CCS detection on Hammer would only measure null divergence. A Hammer CCS FPR run (about 2 h of MPS generation) was not run because it does not support any detection claim.

D8. **Beijing has asr + ccs only**, with no provgate, filters or ablations (Stage-3 descope). The filters are text-only and city-independent in design, but the Beijing entries were not scored.

D9. **Repeated sampling (k > 1 samples per context) was not run.** Stage 4 ran temperature 0.7 with one sample per context, paired with the T = 0 run (C6). k-repeat majority/any voting would multiply calls by k (k = 3 is about 3 h for the same slice).

D10. **The safety-critical up-weighting ablation (safety_weight = 0 at exploration 0.3) was not run** (about 1 h). randomq (exploration 1.0, safety 0) and explore0 (exploration 0) bracket it.

D11. **The trajectory-motif predicate class is weak.** Motif ASR across seeds is 0.074, so any motif-class detection claim rests on very few firing entries. This is a property of the victim model (brief fallback: report it separately).

## Limitations to state in the paper

- L1. The calibration and FPR sets overlap in the Stage-3 Austin runs; report held-out FPR (A1).
- L2. The confirmation path uses an oracle risk label. The text proxy matches it on this corpus, but null entries that mention high-risk tools would raise the confirm rate (A6).
- L3. Q-based sampling shows no gain over uniform probing (A12, C7). The "mobility distribution as defender asset" claim has to be rewritten: detection comes mostly from predicate-independent out-of-scope perturbation, largely induced off-entry routes (C5), not from hitting phi (A8, A10).
- L5. Sampled decoding (T = 0.7) raises the calibrated null rate 2.5x (C6). Deployments need per-temperature calibration.
- L6. The pre-triage threshold and the confirm_text regex were set by the authors without a held-out split. Present them as ablation curves.
- L4. Provenance gating with TMA-NM removes the measured residual (0 ASR); the "gating is not enough" narrative is not supported by these numbers.
