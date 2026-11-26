# The 9/8 lower bound for ring loading: an independent rediscovery

> **Priority correction (26 Nov 2026).** The bound 9/8 is **not** new, and an earlier version of this README wrongly called it new.
> It was first posted by **Maks23** on 8 Oct 2026, as a 14-term exact certificate for the sequence constant β, in [issue #12 of DeepMind's AlphaEvolve problem repository](https://github.com/google-deepmind/alphaevolve_repository_of_problems/issues/12).
> In the same thread, **Karl Däubel** gave a 12-term instance and pointed out that Lemma 7 of his 2022 paper turns such a certificate into a real ring with L − L\* ≥ 9/8·D.
> **rozetyp** then checked those rings directly with CP-SAT ([repo](https://github.com/rozetyp/ring-loading-47-42)).
> The exact 47/42 certificate is due to V. Yadav ([issue #11](https://github.com/google-deepmind/alphaevolve_repository_of_problems/issues/11), 6 Oct 2026).
> Our literature check (Semantic Scholar and arXiv) missed that issue thread.
> The instance below was found independently. This repository is a second, independent set of exact proofs; it does not claim a new bound.

The ring loading problem routes each demand on a cycle, unsplit, either clockwise or counterclockwise, and minimises the maximum edge load L.
L\* is the optimum when demands may be split, and D is the largest demand.
This repository contains explicit instances, exact computer-checked proofs and the search code for the following bounds:

| Constant | Bound before Oct 2026 | 9/8, first shown by | This repository |
|---|---|---|---|
| Ring loading: smallest ρ with L ≤ L\* + ρD on every ring | 11/10 (Skutella, SIAM J. Discrete Math. 2016) | Maks23's certificate + Däubel's Lemma 7 (Oct 2026) | independent 9/8 instance |
| Schrijver–Seymour–Winkler sequence constant β (Problem 6.61 of Georgiev, Gómez-Serrano, Tao and Wagner, arXiv:2511.02864) | ≈1.119 (AlphaEvolve, 2025); 47/42 exact (V. Yadav, Oct 2026) | Maks23 (8 Oct 2026) | independent 9/8 sequence (m = 15) |

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
| `python ae_raw_lift.py` | AlphaEvolve's published floating-point sequence, taken exactly: α = 1.1190475…; *our* ring lift gives a gap ≤ 1.0823 (Däubel's Lemma 7 construction does not lose the gap) | under 1 min |

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

Please cite the original posts first: Maks23, issue #12 of the AlphaEvolve problem repository (8 Oct 2026), and K. Däubel, SIAM J. Discrete Math. 36 (2022), Lemma 7.
This repository: Thanh Van Nguyen, *The 9/8 lower bound for ring loading: an independent rediscovery with exact proofs*, 2026 (note in `paper/`, not submitted to arXiv).

## License

MIT (see `LICENSE`).
