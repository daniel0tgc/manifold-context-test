# Manifold‑Context Simulation — README

This project implements a minimal, testable cognitive dynamics model inspired by the idea that human thought and memory live on a latent manifold, and that recent high‑salience experiences distort this manifold by creating temporary attractor basins that bias later recall and mind‑wandering.

It is a deliberately interpretable toy system for probing hypotheses about:

- **episodic memory as templates** in latent space
- **context as a decaying trace** that biases retrieval
- **salience/arousal as a gain** on memory consolidation or access
- **cue‑driven attractor dynamics** producing daydream‑like transitions

The immediate motivating example: after a conversation with someone (high salience), later exposure to a weakly related cue disproportionately triggers recall/daydreams of specific episodes — because the recent event reshaped the energy landscape. Nuerons activated during conversation remain in the context of a person for a short period, triggering future cues/thoughts.

The system approximates this via:

```
x(t) : current internal state on latent manifold
m_i  : stored episodic memory templates
h(t) : recent‑history context trace (decaying)
u(t) : external cue projected into latent space
```

Dynamics:

```
h <- alpha*h + beta*salience*event_embedding
attn_i ∝ exp( <u,m_i> + rho <h,m_i> )         # history biases retrieval
bias(x) ≈ weighted mixture of near memories    # forms attractors
x_{t+1} = x_t + ( -γx + bias + u_proj )/τ + noise
```

This explicit construction lets us **experimentally vary** parameters like
`alpha (recency), beta (update strength), rho (context coupling), noise, tau` and measure their effect on recall likelihood and trajectory behavior.

---

## Repo structure (target)

```
manifold-context-sim/
├── model.py            # autoencoder to define latent manifold
├── memory.py           # episodic store + context trace
├── dynamics.py         # attractor-like latent dynamics
├── train.py            # train AE + build synthetic memory
├── simulate_events.py  # run scenarios (e.g. tickle-demo)
├── tests/              # basic correctness tests
├── requirements.txt
├── README.md (this)
└── saved/              # serialized models/memory
```

---

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0
- numpy, matplotlib, scikit‑learn, tqdm

```
pip install -r requirements.txt
```

---

## Minimal usage flow

1. **Train latent space & create memory**

```
python train.py
```

This trains the autoencoder on synthetic events and stores encoded templates.

2. **Run a scenario**

```
python simulate_events.py --scenario tickle_demo
```

Produces:

- PCA/t‑SNE plot of trajectory vs memories
- recall probability difference pre/post high‑salience event

3. **Run tests**

```
pytest -q
```

---

## Why this is biologically aligned (brief)

- **Episodic memory templates** approximate hippocampal pattern templates
- **History trace h(t)** approximates short‑lived synaptic eligibility traces
- **Salience scaling** reflects neuromodulatory gain (NE/ACh)
- **Attractor drift** mimics content‑addressable completion and mind‑wandering

The simplicity is intentional: it is a _generative, manipulable testbed_ for cognitive hypotheses, not an engineering stack.

---

## Next planned extensions

- dual timescale contexts (short vs long)
- consolidation rule for w_i
- learned E0(x) instead of fixed quadratic
- inject real embeddings (text or audio)
- fit α,ρ,β to human diary/mind‑wandering data

---

## License

TBD.

---

## Contact / Intent

This repo exists to explore whether simple, low‑parameter latent dynamics can reproduce human‑like bias in spontaneous thought as a function of recency, salience, and context coupling.
