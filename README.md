# A 9/8 lower bound for the ring loading problem

The ring loading problem routes each demand on a cycle, unsplit, either clockwise or counterclockwise, and minimises the maximum edge load L.
L\* is the optimum when demands may be split, and D is the largest demand.
This repository contains explicit instances, exact computer-checked proofs and the search code for two new lower bounds:

| Constant | Previous lower bound | This work |
|---|---|---|
| Ring loading: smallest ρ with L ≤ L\* + ρD on every ring | 11/10 (Skutella, SIAM J. Discrete Math. 2016) | **9/8** |
| Schrijver–Seymour–Winkler sequence constant β (Problem 6.61 of Georgiev, Gómez-Serrano, Tao and Wagner, arXiv:2511.02864) | 47/42 (AlphaEvolve, 2025) | **9/8** |

The known upper bounds are ρ ≤ 13/10 (Däubel, 2022) and β ≤ 19/14 (Skutella, 2016).

The write-up is [`paper/main.pdf`](paper/main.pdf).

## The instance

The sequence has m = 15 terms, all multiples of 1/16:

    u = 3/4 5/16 1/4 11/16 1/8 1/2 3/4 5/16 1/4 1/4 3/4 1/16 3/4 1/2 0
    v = 1/8 11/16 1/4 5/16 3/4 1/2 1/8 11/16 1/4 3/4 0 15/16 1/8 1/2 7/8

For every z ∈ ∏{v_k, −u_k}, some k has |Σ_{i≤k} z_i − Σ_{i>k} z_i| ≥ 9/8, and some z attains exactly 9/8.

Lifting the sequence to a ring gives one instance in two forms, both with D = 1, **L\* = 15/2** and **L = 69/8**:

- [`instances/ring_instance_n30.json`](instances/ring_instance_n30.json): 30 nodes, 15 crossing demands and 40 adjacent filler demands. On 13 edges, two filler pieces join the same pair of nodes.
- [`instances/ring_instance_distinct_pairs_n43.json`](instances/ring_instance_distinct_pairs_n43.json): 43 nodes, with all 55 demands on distinct node pairs. It is obtained by subdividing those 13 edges.

## Reproduce

```
pip install -r requirements.txt
python certificate.py
```

`certificate.py` takes about 2 s and uses only integers and Fractions in every decision. It checks:

- the sequence value 9/8 under both index conventions;
- Table 1 of the paper;
- the ring proof: all 2^15 crossing routings, a necessary condition that leaves 8 of them, and an exact subset-sum DP that rules out all 8;
- the 43-node instance;
- the lift of the polished AlphaEvolve sequence, whose gap is 23/21.

Independent checks:

| Script | What it checks | Time |
|---|---|---|
| `python verify.py instances/instance_m15_den16_a.json` | Brute force of the sequence value | under 1 s |
| `python ring_exact.py instances/instance_m15_den16_a.json` | Second implementation of the ring proof (numba) | about 10 s |
| `python check_minsum.py instances/instance_m15_den16_a.json 9 8` | Ring bound by a different argument, without the filter lemma | a few minutes |
| `python ae_raw_lift.py` | AlphaEvolve's published floating-point sequence, taken exactly: α = 1.1190475…, ring-lift gap ≤ 1.0823 | under 1 min |

## Search code

The search code is what found the instance:

- `ringlib.py`: exact branch and bound for α, pruning on the range of the prefix sums.
- `ascent.py`: LP active-set ascent.
- `ils.py`: iterated local search with elite restarts.
- `milp_ascent.py`: exact test for a local ascent direction.
- `milp_cg.py`: exact maximisation by constraint generation, which works for small m only.
- `check.py`: cross-checks branch and bound against brute force.

Example run: `python ils.py <seed> <seconds> <m_lo> <m_hi> out elite`.

## Other files in `instances/`

- `instance_m15_den16_b.json` and `instance_m16_den48.json`: two further sequences with value exactly 9/8. Their ring lifts lose part of the gap.
- `instance_m13_fullk.json`: value 9/8 when the end term k = m is included.
- `alphaevolve_raw.json`: AlphaEvolve's published sequence.
- `alphaevolve_47_42_polished.json`: our LP-polished version, with value exactly 47/42.

## Use of AI tools

The search code, the verifiers and the first draft of the note were written with Claude (Anthropic) in Claude Code, at the author's direction. `check_minsum.py` was written by a separate Claude agent acting as referee. Every claim rests on the exact computations above.

## Citation

Thanh Van Nguyen, *A 9/8 lower bound for the ring loading problem*, 2026. Preprint, with the arXiv identifier to be added.

## License

MIT (see `LICENSE`).
