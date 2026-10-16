# Correlated Pauli Noise in Rotated Surface-Code Memories

**Research report — consolidated draft**
**Simulation setting:** `P_BASE = 0.001`; rotated surface-code Z-memory circuits; code distances 3, 5, 7, and 9 across different experiments.
**Tools:** Stim, PyMatching, NumPy, and selected SciPy/LDPC tooling for additional decoder tests.

> **Scope note.** This report consolidates results reported in the experiment logs available during this project. Several tables come from separate runs with different seeds and shot counts; they should not be interpreted as a single pooled experiment unless explicitly labeled pooled. Before publication, verify each reported number against the archived CSV for that run and record the exact package versions and Git commit.

## Abstract

This project studies how injected spatially local correlated Pauli errors affect logical-memory performance in rotated surface-code circuits, and how detector-error-model (DEM) representation and matching-decoder choices influence measured results. The physical circuit uses a background noise level of `P_BASE = 0.001` and adds adjacent vertical data-qubit `XX` pair events with probability parameters including `p_corr = 0.0001` and `0.0002`. Matched-independent control noise is used to help separate the effect of explicit pair correlations from an increase in single-qubit error marginals.

Three findings stand out. First, the distance-3 correlated DEM contains parallel graphlike mechanisms with conflicting logical-observable labels. Because PyMatching merges parallel edges and can retain labels from the first added edge, DEM instruction order can strongly affect distance-3 predictions. Second, at distances 5 and 7, decoder ablations and paired decoder comparisons show that correlated-pair information is useful under the tested noise models: removing pair information increases logical failures, and correlation-aware matching improves on ordinary MWPM in the tested conditions. Third, decoder optimizations can trade runtime for accuracy, but distance-9 screening-based risk budgets do not guarantee the same error increase on fresh samples. A 1.25× correlated-pair DEM-strength adjustment also failed to establish a statistically reliable improvement in independent-seed validation.

These are simulation results for a particular circuit family and decoder stack. They are not a hardware demonstration, a threshold estimate, or proof that any heuristic is optimal for the physical circuit distribution.

## 1. Questions

1. How does local `XX` pair noise affect logical failure rates compared with a matched-independent control?
2. How much can DEM representation and edge-label handling distort the apparent distance-3 result?
3. Does correlation-aware matching usefully exploit correlated-noise information at larger distances?
4. Can a lightweight gate selectively invoke correlation-aware decoding to reduce runtime?
5. Does scaling the pair-event probabilities in the decoder DEM improve held-out performance?

## 2. Simulation design

### 2.1 Circuit and noise model

The base circuit family is Stim's `surface_code:rotated_memory_z`, using the code distance as the number of syndrome-extraction rounds in the main runs. Background noise is parameterized by `P_BASE = 0.001`, using depolarizing and reset/measurement-flip channels. Explicit correlated events apply `XX` to adjacent vertical data-qubit pairs with per-event probability `p_corr`.

The comparison model is a matched-independent control. Its single-qubit error probabilities are adjusted to account for the marginal contribution of pair events, rather than injecting the joint pair mechanism. This control is useful but is not identical to the correlated model: the models intentionally differ in joint error structure.

The experiment family includes distances `d = 3, 5, 7, 9`, but not every analysis was run at every distance. The exact sample sizes vary by experiment and are stated alongside the relevant results.

### 2.2 Decoder variants

The investigation compared:

- **Ordinary MWPM:** PyMatching loaded from a DEM without correlation-aware decoding.
- **Correlation-aware MWPM (CA):** PyMatching configured and called with `enable_correlations=True`.
- **Probability-prioritized matching heuristic:** an order-aware audit and graph-construction procedure used to stabilize the distance-3 result where parallel-edge classes carry competing logical labels.
- **Gated CA:** a low-cost score is computed first, then correlation-aware matching is used only when a detector-weight or matching-cost threshold is reached.
- **Scaled pair-strength DEMs:** physical circuits retain the target `p_corr`; only the correlated-pair probabilities in the decoder DEM are multiplied by a candidate scale.
- **Exact DEM-MAP at distance 3:** an exhaustive posterior calculation over all `2^24` detector patterns for the DEM-defined distribution.

### 2.3 Statistical conventions

Many decoder comparisons use the same sampled shots for both decoders, allowing paired differences in logical failures. Independent-seed validations were used for selected gate and DEM-strength candidates. Where the result log reports a confidence interval, it is reproduced below. A confidence interval that includes zero does not establish equality; it means the tested data did not resolve the difference at the reported precision.

## 3. Main findings

### Finding A — distance-3 DEM order sensitivity is a material confounder

An audit found the following numbers of potential parallel-edge logical-label conflicts:

| Code distance | Matched-independent control | Correlated model |
|---:|---:|---:|
| 3 | 0 | 12 |
| 5 | 0 | 0 |
| 7 | 0 | 0 |

At distance 3, the 12 conflicting singleton detector signatures were `D0`, `D1`, `D2`, `D3`, `D5`, `D7`, `D8`, `D10`, `D13`, `D15`, `D16`, and `D18`. The same detector signature could be produced by one mechanism with no logical-observable label and another carrying `L0`. The audit traced examples to a mixture of injected correlated `XX` events and background Pauli-error mechanisms.

In the tested PyMatching graph-loading path, the order of equivalent DEM instructions changed predictions when a merged parallel-edge class had competing logical labels. Across the complete distance-3 DEM, the audit found prediction changes under reordering; repeated correlated-circuit tests showed the observed logical-failure estimate could shift from roughly 0.90% under the original order to roughly 0.25% under the revised/prioritized construction. One 500,000-shot comparison recorded 4,527 failures (0.9054%) for ordinary matching in the original order and 1,275 (0.2550%) for reversed order; the probability-prioritized graph also recorded 1,275 failures, with zero prediction changes in that sample under order reversal.

**Interpretation:** this is a decoder-representation sensitivity, not evidence that changing DEM text order changes the underlying physical noise. PyMatching's documentation notes that parallel edges are merged and that the logical-observable indices for merged parallel edges are retained from the first added edge. That behavior is material when the input contains conflicting labels ([PyMatching source/documentation](https://github.com/oscarhiggott/PyMatching/blob/master/src/pymatching/matching.py)).

A probability-prioritized graph is a useful stabilizing heuristic for this experiment, but it is not an exact circuit maximum-likelihood decoder. The distance-3 result must therefore be reported separately from distances 5 and 7 rather than treated as an uncomplicated scaling point.

### Finding B — correlated-pair information matters at distances 5 and 7

A grouped decoder-model ablation removed pair mechanisms from the decoder's model while leaving the physical circuits correlated. Removing all pair information increased logical failures in each tested distance/noise-strength condition. Reported excess failures relative to the full pair-information decoder model were:

| Distance | `p_corr` | Excess failures when pair information was removed |
|---:|---:|---:|
| 5 | 0.0001 | +0.02460 percentage points |
| 5 | 0.0002 | +0.05600 percentage points |
| 7 | 0.0001 | +0.00720 percentage points |
| 7 | 0.0002 | +0.02500 percentage points |

The report from this ablation indicates that these four comparisons passed its family-wise Bonferroni confidence-interval criterion. Removing either the interior or boundary subset also increased failures in the tested conditions. Individual pair effects were not robustly distinguished after global multiple-comparison correction.

A separate same-shot decoder comparison measured the following ordinary-MWPM versus correlation-aware-MWPM failure rates:

| Distance | `p_corr` | Model | Ordinary MWPM | Correlation-aware MWPM |
|---:|---:|---|---:|---:|
| 5 | 0.0001 | Matched-independent control | 0.01520% | 0.01260% |
| 5 | 0.0001 | Fully correlated | 0.05520% | 0.04580% |
| 5 | 0.0002 | Matched-independent control | 0.01870% | 0.01400% |
| 5 | 0.0002 | Fully correlated | 0.10020% | 0.07810% |
| 7 | 0.0001 | Matched-independent control | 0.00155% | 0.00090% |
| 7 | 0.0001 | Fully correlated | 0.01180% | 0.00785% |
| 7 | 0.0002 | Matched-independent control | 0.00300% | 0.00135% |
| 7 | 0.0002 | Fully correlated | 0.02185% | 0.01460% |

These are results from the dedicated decoder-comparison run, not from the later gate validation. Rates will differ slightly across independent seeds and sample sizes. Across these tested conditions, correlation-aware matching performed better at distances 5 and 7. The distance-3 case is an exception because of the DEM parallel-edge label issue described above.

Stim provides stabilizer-circuit simulation and DEM generation; PyMatching's correlated mode uses a two-pass matching procedure to exploit decomposable correlated/hyperedge information ([Stim project](https://github.com/quantumlib/Stim), [PyMatching documentation](https://github.com/oscarhiggott/PyMatching)).

### Finding C — exact distance-3 DEM-MAP provides a useful benchmark

An exhaustive posterior convolution considered all `2^24 = 16,777,216` detector states for the flattened DEM-defined distribution. The computed distribution normalized to `0.9999999999999970`. For one distance-3 setting at `p_corr = 0.0001`, the results were:

| Decoder / benchmark | Estimated logical failure risk |
|---|---:|
| Exact MAP Bayes risk under the DEM distribution | 0.23669523% |
| Probability-prioritized MWPM, circuit validation | 0.2657% |
| Exact DEM-MAP, circuit validation | 0.2463% |

In the corresponding million-shot circuit-vs-DEM validation, the DEM-sampled exact-MAP rate was 0.2356%, close to the calculated DEM Bayes risk of 0.2367%. The circuit-sampled exact-MAP rate was 0.2463%; the reported circuit-minus-DEM difference was +0.0107 percentage points with 95% CI `[−0.00289, +0.02429]` percentage points, which includes zero. This test did not detect a statistically resolved difference for that logical-failure metric, but it does not prove equality of the complete circuit and DEM distributions.

The probability-prioritized decoder was measurably above exact DEM-MAP in that validation: 0.2657% versus 0.2463%, a paired excess of 0.0194 percentage points with reported 95% CI `[+0.01552, +0.02328]` percentage points.

**Scope:** exact MAP is optimal for the probabilistic distribution represented by the complete DEM mechanisms. It is not automatically optimal for every detail of the physical circuit if the DEM is an approximation or changes relevant dependencies.

### Finding D — gated correlation-aware decoding trades runtime for accuracy

The gate first computes a cheap score, then invokes correlation-aware matching only above a fixed threshold. Thresholds were screened and then evaluated on fresh seeds. This is an engineering experiment, distinct from the main physical-noise findings.

#### Distances 5 and 7: detector-weight gates

Frozen thresholds were `detector weight >= 5` for `d=5` and `>= 8` for `d=7`.

| Distance | `p_corr` | Ordinary MWPM | Full CA | Weight-gated CA | Routed to CA | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.0001 | 0.05607% | 0.04567% | 0.04837% | ~10% | ~1.62× |
| 5 | 0.0002 | 0.09860% | 0.07603% | 0.07973% | ~10% | ~1.66× |
| 7 | 0.0001 | 0.01137% | 0.00665% | 0.00697% | ~26.2% | ~1.43× |
| 7 | 0.0002 | 0.02238% | 0.01520% | 0.01585% | ~26.7% | ~1.41× |

The distance-5 pooled gate validation used three replicates of one million shots per condition; distance 7 used three replicates of two million shots per condition. The gates retained most of the full-CA accuracy gain while reducing decoder time in these tested settings. At distance 7, the gate/full-CA differences for the correlated conditions were small but statistically resolved in the pooled paired comparisons.

#### Distance 9: matching-cost gates

The distance-9 matching-cost validation used three independent replicates of two million shots per condition (six million shots per pooled condition). Its results are summarized below. A separate standardized timing benchmark of the detector-weight gate `weight >= 12` was conducted on October 15, 2026, and is reported in the additional subsection below. The two experiments used different sample sizes and timing protocols and should not be conflated.

The latest independent validation gives:

| `p_corr` | Physical model | Decoder | Logical failure rate | Routed to CA | Speedup |
|---:|---|---|---:|---:|---:|
| 0.0001 | Fully correlated | Full CA | 0.00088% | 100% | Reference |
| 0.0001 | Fully correlated | Cost-per-detector gate, 5% screen budget | 0.00095% | 44.4% | 1.22× |
| 0.0001 | Fully correlated | Total-cost gate, 10% screen budget | 0.00092% | 17.2% | 1.55× |
| 0.0002 | Fully correlated | Full CA | 0.00198% | 100% | Reference |
| 0.0002 | Fully correlated | Cost-per-detector gate, 5% screen budget | 0.00205% | 45.6% | 1.20× |
| 0.0002 | Fully correlated | Total-cost gate, 10% screen budget | 0.00253% | 17.8% | 1.55× |

For the cost-per-detector 5% setting, the pooled gate-minus-full-CA difference was +0.000067 percentage points at `p_corr=0.0001` (95% CI `[−0.000013, +0.000147]`) and +0.000067 percentage points at `p_corr=0.0002` (95% CI `[−0.000046, +0.000180]`). Both intervals include zero: these tests did not establish a difference, but they do not prove equivalence.

For the total-cost 10% setting, the pooled gate-minus-full-CA difference at `p_corr=0.0002` was +0.000550 percentage points (95% CI `[+0.000341, +0.000759]`). This is a statistically supported increase in failures. At `p_corr=0.0001`, the observed increase was smaller and its confidence interval included zero.

**Interpretation:** a 5% screening budget is not a guarantee of a 5% relative increase on fresh samples. Of the tested distance-9 options, the cost-per-detector score at its 5% setting is the more defensible compromise for further work; full CA remains the reference when accuracy is the priority.

#### Standardized detector-weight-gate benchmark (2026-10-15)

A separate distance-9 benchmark used nine rounds, `p_base = 0.001`, and a
detector-weight threshold of 12. Each physical-model/strength condition used
three replicates of 100,000 shots (300,000 shots per condition; 1.2 million
shots total). Syndromes with detector weight below 12 were decoded by ordinary
MWPM; those at or above 12 were decoded by correlation-aware MWPM.

Each decoding path was warmed on a separate 2,048-shot batch, excluded from
timing. Sampling and failure bookkeeping were excluded; weight calculation and
selective decoding were included. The benchmark used 25,000-shot chunks and
alternated decoder call order by chunk.

| `p_corr` | Physical model | Ordinary failures | Full CA failures | Gated failures | Routed to CA | Full CA time (s) | Gated time (s) | Speedup |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0001 | Fully correlated | 9/300,000 | 4/300,000 | 4/300,000 | 50.826% | 4.258682 | 3.364562 | 1.266x |
| 0.0001 | Matched-independent control | 1/300,000 | 1/300,000 | 1/300,000 | 51.923% | 3.443126 | 3.136762 | 1.098x |
| 0.0002 | Fully correlated | 9/300,000 | 4/300,000 | 4/300,000 | 51.779% | 4.152254 | 3.553308 | 1.169x |
| 0.0002 | Matched-independent control | 1/300,000 | 1/300,000 | 1/300,000 | 53.497% | 4.109446 | 3.385870 | 1.214x |

Across the four conditions, full correlation-aware decoding took 15.963508
seconds and the gated path took 13.440502 seconds: a combined measured
speedup of 1.188x and a 15.8% reduction in decoding time under this protocol.

No gate-only or full-CA-only logical-failure classifications were observed
among the 1.2 million paired shots. This is empirical agreement on the sampled
data, not proof of statistical equivalence. Each condition had only one or
four logical failures per decoder. These timing results are separate from the
six-million-shot matching-cost-gate validation above.

Raw evidence:
`results/paper_evidence/d9_gate_standardized_20261015_024110/replicates.csv`,
`summary.csv`, and `run_metadata.txt`.


### Finding E — increasing decoder pair strength to 1.25× was inconclusive

The physical circuit always retained its target pair-event probability. Only the pair-event probabilities in the decoder DEM were scaled. The two best screen-ranked scales were 1.00× (683 aggregate screening failures) and 1.25× (689), and 1.25× was compared with 1.00× on fresh samples with one million shots in each of four conditions.

| Distance | `p_corr` | Scale 1.00× | Scale 1.25× | Delta (candidate − baseline) | 95% CI for delta |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.0001 | 0.04830% | 0.04770% | −0.000600 pp | [−0.001519, +0.000319] pp |
| 5 | 0.0002 | 0.07950% | 0.07940% | −0.000100 pp | [−0.001631, +0.001431] pp |
| 7 | 0.0001 | 0.00750% | 0.00720% | −0.000300 pp | [−0.000639, +0.000039] pp |
| 7 | 0.0002 | 0.01600% | 0.01590% | −0.000100 pp | [−0.000296, +0.000096] pp |

The pooled difference was −0.000275 percentage points. The reported Bonferroni interval was `[−0.000857, +0.000307]` percentage points, which includes zero. Thus, the 1.25× adjustment is **not a demonstrated improvement**. The evidence currently supports retaining 1.00× as the default and treating further calibration as exploratory.

### Finding F — DEM-fidelity checks were reassuring but not exhaustive

For distances 5 and 7, a circuit-vs-DEM fidelity test used two million shots per source condition. After Bonferroni correction, none of the tested detector-observable covariance, detector-pair covariance, or raw logical-observable differences was statistically resolved. This supports using the DEM as a working reference for the measured quantities and conditions. It does not establish equality of the full joint distributions or rule out higher-order mismatches.

## 4. Practical conclusions

1. **Treat distance 3 separately.** Its correlated DEM has a documented parallel-edge logical-label ambiguity under the tested graph-construction pathway. Report the order audit and exact DEM-MAP benchmark alongside any distance-3 rate.
2. **Keep full correlation-aware MWPM as the main reference decoder at distances 5 and 7.** Correlated-pair ablations and same-shot comparisons support using pair information under the simulated conditions.
3. **Describe gated decoding as an engineering trade-off, not the main scientific claim.** Distance 5 and 7 weight gates improved throughput in the reported tests; distance 9 favored matching-cost gating over detector weight.
4. **Do not claim a calibrated optimum at 1.25×.** The independent validation did not establish a significant improvement over the nominal 1.00× pair strength.
5. **Separate statistical non-significance from equivalence.** With very low logical-failure counts, intervals can be too wide to resolve small differences.

## 5. Limitations

- The project uses a specific rotated Z-memory circuit family and a chosen Pauli-noise model. Results need not generalize to other stabilizer codes, circuit schedules, hardware noise, leakage, coherent errors, or non-Pauli channels.
- Different results were collected in multiple scripts, with different seeds and shot budgets. Cross-table differences may reflect both genuine settings and Monte Carlo variation.
- The distance-3 priority heuristic is a representation-handling strategy, not proof of globally optimal decoding.
- Exhaustive exact MAP is feasible here because there are 24 detector bits; its `2^n` state growth makes this method impractical at larger distances.
- Gating thresholds were selected by screening, so independent validation is essential. A finite screening budget does not guarantee the same relative risk on new samples.
- Very low event counts, especially in matched-independent controls at distance 9, limit inference about small relative changes.
- Timing speedups depend on software versions, machine, batch size, and implementation; report the benchmark environment with any final performance claim.

## 6. Reproducibility checklist

Before publishing the repository, archive or record:

- Exact Python, Stim, PyMatching, NumPy, SciPy, and `ldpc` versions where used.
- Git commit or source hashes for every script used to generate a table.
- All CSV outputs referenced in the report.
- Each experiment's random seed, shot count, distance, round count, `P_BASE`, `p_corr`, decoder type, and threshold-selection split.
- The distinction between screening data and fresh validation data.
- The exact DEM construction and error-instruction ordering strategy.
- Machine and timing setup for any speedup claim.

The `create_reproducibility_manifest_qec.py` script can help inventory source files and their hashes. Run it in the project root, then confirm that its output includes the archived CSVs used in the report.

## 7. References

1. Stim project documentation and source: [https://github.com/quantumlib/Stim](https://github.com/quantumlib/Stim). Stim is a high-performance stabilizer-circuit simulator with detector-error-model support.
2. PyMatching source/API documentation: [https://github.com/oscarhiggott/PyMatching](https://github.com/oscarhiggott/PyMatching). The matching API describes correlation-aware two-pass decoding and the handling of parallel edges when constructing a matching graph from a DEM.
3. PyMatching correlated matching implementation details and release notes: [https://github.com/oscarhiggott/PyMatching/releases](https://github.com/oscarhiggott/PyMatching/releases).

---

**Suggested paper title:** *Correlated Pauli Noise and Decoder-Representation Effects in Rotated Surface-Code Memory Simulation*.
