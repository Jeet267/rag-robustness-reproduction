# RAG Robustness to Spurious Features: Reproduction and a Prompt-Level Intervention

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![Model](https://img.shields.io/badge/model-Llama--3.1--8B--Instruct-orange)

**Can an explicit instruction reduce a RAG model's sensitivity to spurious features, without any training?**

NLP Research Project (M1–M6), Newton School of Technology, Rishihood University

| Author | Roll No. |
|---|---|
| Abhijeet Kumar Shah | 230036 |
| Aditya Raj Sharma | 230123 |
| Pintu Singh | 230105 |

**Anchor paper:** S. Yang et al., *Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data*, ACL 2026 ([arXiv:2503.05587](https://arxiv.org/abs/2503.05587), [code](https://github.com/maybenotime/RAG-SpuriousFeatures))

---

## TL;DR

1. **Reproduction (M2):** On SIG_Wiki, Llama-3.1-8B-Instruct (4-bit) is **33–49 RR points more robust** than the paper reports on four of five categories; only Meta is close (90.0 vs 94.0).
2. **Experiment (M3–M4):** Adding one instruction sentence to the prompt raises overall Robustness Rate significantly: **69.0 → 72.8** (generic, p = 0.002) and **69.0 → 71.9** (specific, p = 0.03). HTML gains most (47 → 61).
3. **Ablation:** Naming the spurious features gives **no benefit** over a generic instruction (p = 0.53).
4. **Failure analysis (M5):** The specific instruction's gain comes **entirely from more pairs answered wrongly on both documents** (more NO-RES refusals). Conditional RR, the share of correct answers that survive the perturbation, is unchanged (66.2 → 66.8). **RR can rise while the model only becomes more cautious.**

---

## Background

A RAG model answers from retrieved documents. The same facts can arrive in different surface forms: raw HTML or JSON, simple or elaborate writing, shuffled sentence order, a "Source: Twitter" tag or a timestamp. The anchor paper calls these **spurious features** and shows they can flip a model's answer.

**Metrics** (computed per question–document pair; Y = 1 if the normalised gold answer is contained in the prediction):

| Metric | Meaning |
|---|---|
| **RR**, Robustness Rate | Correctness is the same on original and perturbed document (both right **or both wrong**) |
| **WR**, Win Rate | Wrong on original, right on perturbed |
| **LR**, Lose Rate | Right on original, wrong on perturbed |
| **Conditional RR** (ours) | Of the pairs answered correctly on the original document, the share still correct on the perturbed one |

---

## M1: Literature survey and anchor paper

Five papers surveyed: Yang et al. (ACL 2026), Cuconasu et al. (SIGIR 2024), Yoran et al. (ICLR 2024), Shi et al. (ICML 2023), Liu et al. (TACL 2024). Yang et al. was chosen as anchor because its benchmark, code, scoring and model are public.

| Reproducibility checklist | |
|---|---|
| Public dataset (SIG_Wiki) | ✅ |
| Public code and scoring script | ✅ |
| Public model weights | ✅ |
| Metric clearly defined | ✅ |
| Evaluation runs on a free GPU (Kaggle T4) | ✅ |
| SFT/DPO training (8×A100) | ❌ not feasible, not reproduced |

## M2: Baseline reproduction

**Setup:** SIG_Wiki, 16 perturbations × 100 pairs; 1,500 used (the Mistral self-paraphrase subset is excluded when evaluating Llama). Llama-3.1-8B-Instruct, 4-bit NF4, greedy decoding, authors' prompt and scoring, Kaggle T4. No training.

| Category | Paper RR (Table 6) | Our RR | Gap |
|---|---|---|---|
| Style | 10.0 | 56.5 | +46.5 |
| Source | 15.5 | 49.0 | +33.5 |
| Logic | 20.0 | 68.7 | +48.7 |
| Format | 24.0 | 64.5 | +40.5 |
| Meta | 94.0 | 90.0 | −4.0 |

![Reproduction](report/fig_repro.png)

**Possible causes of the gap (not isolated):** (1) SIG_Wiki keeps pairs on which the authors' runs were *not* robust, so an independent run is likely to look more robust; (2) 4-bit NF4 vs bf16; (3) greedy decoding vs temperature 0.1.
**Most fragile perturbation:** HTML (RR 47%, accuracy 69% → 28%). 28.1% of answers on original documents were NO-RES.

## M3: Research gap and hypothesis

**Gap.** The anchor paper improves robustness only through training (SFT/DPO on 8×A100). Whether a zero-cost, inference-time instruction helps was never tested.

**Hypothesis.** *Appending an instruction that tells the model to ignore format, style, reasoning order, source and metadata will increase RR, because instruction-tuned models follow explicit constraints.*

| Category | Predicted Δ (specific vs baseline) |
|---|---|
| Format | +5 to +10 |
| Style | +5 to +10 |
| Source | +3 to +5 |
| Logic | 0 to +3 |
| Meta | ≈ 0 (ceiling) |

Generic instruction: at most half of the specific gain. Predictions were recorded in the first cell of the Kaggle notebook version saved before the run (25 Nov 2026, ~21:40 IST); see [`M3_hypothesis.md`](M3_hypothesis.md).

## M4: Controlled experiment and ablation

**One variable:** a sentence appended to the authors' unchanged prompt.

| Arm | Extra sentence |
|---|---|
| Baseline | none (= M2) |
| Generic | "Base your answer only on the information contained in the document." |
| Specific | "The document may be written in a different format (HTML, JSON, Markdown, YAML), a different writing style or reasoning order, and may include source or timestamp metadata. These surface features do not change the facts: extract the answer from the factual content only." |

**Held fixed:** weights, 4-bit NF4, greedy decoding, 64 new tokens, batch 8, the same 1,500 pairs, authors' scoring. 3 arms × 3,000 answers; 82 minutes on Kaggle 2×T4. Significance: exact McNemar test on paired robust / non-robust outcomes.

**Determinism check:** the baseline arm reproduces M2 exactly in every category.

| Category | Baseline | Generic | Specific | Predicted Δ | Actual Δ | p (spec.) |
|---|---|---|---|---|---|---|
| Style | 56.5 | 57.0 | 58.5 | +5..+10 | +2.0 | 0.71 |
| Source | 49.0 | 53.5 | 56.5 | +3..+5 | +7.5 | 0.07 |
| Logic | 68.7 | 74.7 | 69.0 | 0..+3 | +0.3 | 1.00 |
| Format | 64.5 | 70.8 | 69.8 | +5..+10 | +5.3 | 0.07 |
| Meta | 90.0 | 91.0 | 90.8 | ≈0 | +0.8 | 0.79 |
| **All 1,500 pairs** | **69.0** | **72.8** (p = 0.002) | **71.9** (p = 0.03) | | **+2.9** | **0.03** |

![Results by arm](report/fig_arms.png)

**Ablation (generic vs specific):** p = 0.53, no difference; the prediction that generic would give at most half the gain is **not supported**.
**Per perturbation:** HTML 47 → 55 → 61 (+14 specific); JSON 65 → 80 (generic); Markdown falls 80 → 75.

## M5: Failure analysis

**Where the RR gain comes from** (all 1,500 pairs, %):

| | Baseline | Generic | Specific |
|---|---|---|---|
| RR | 69.0 | 72.8 | 71.9 |
| correct on both | 33.1 | 34.6 | 30.3 |
| wrong on both | 35.9 | 38.2 | 41.7 |
| Conditional RR | 66.2 | 69.7 | 66.8 |
| Accuracy on original document | 49.9 | 49.7 | 45.3 |
| NO-RES rate | 30.0 | 34.4 | 34.5 |
| Answers longer than 5 words | 39.5 | 51.7 | 42.6 |

![RR decomposition](report/fig_decomp.png)

- **Specific instruction:** all of its RR gain comes from pairs wrong on both documents; correct-on-both falls and refusals rise. The model became more cautious, not more robust.
- **Generic instruction:** a small real gain in conditional RR (+3.5), but median answer length rises from 3 to 7 words, and containment-based scoring rewards longer answers.
- **Floor effect on Meta:** RR 90.0 with 32.8% accuracy; 62% of Meta pairs are wrong on both documents.

**Failure types in 10 sampled losses (specific arm):**

| Type | Count | Example |
|---|---|---|
| False refusal: answer present, model says NO-RES | 4 | HTML: "2017" is in the `<title>` tag; model says the document has no information about the Crows |
| Distractor span: first plausible entity in rewritten text | 6 | Random order: "anaphase" instead of "metaphase"; complex style: "New Found Glory" instead of "the Ramones" |

**Limitations:** one model, 4-bit; one wording per instruction; specific prompt longer than generic (length confound); string-match scoring, no LLM judge; one greedy run; SIG_Wiki is filtered for hard cases. Classroom-scale experiment: no high-stakes claims.

## M6: Paper and presentation

- [`report/paper.pdf`](report/paper.pdf): IEEE-format paper; LaTeX source in [`report/paper.tex`](report/paper.tex)
- [`report/RAG_Robustness_M1_M6.pptx`](report/RAG_Robustness_M1_M6.pptx): presentation slides

---

## Repository structure

```
├── README.md
├── RAG_M1_M2_Report.pdf              # M1 literature survey + M2 report
├── M2_Baseline_Reproduction.ipynb    # M2 reproduction notebook (with outputs)
├── nlp_m3_m4_m5.ipynb                # M3–M5 experiment notebook (with outputs)
├── M3_hypothesis.md                  # M3 gap, hypothesis, predictions, plan
├── requirements.txt
├── rag_m2/                           # M2 artifacts
│   ├── rr_comparison.png
│   ├── table6_comparison_*.csv
│   ├── per_perturbation_*.csv
│   └── generations_*.jsonl
├── rag_m4/                           # M4–M5 artifacts
│   ├── generations_*_{baseline,generic,specific}.jsonl   # raw outputs, 3 arms × 1,500 pairs
│   ├── m4_rr_by_category.csv         # main results table
│   ├── m4_rr_per_perturbation.csv    # 15 perturbations × 3 arms
│   ├── m4_mcnemar.csv                # significance tests
│   ├── m4_wr_lr_by_category.csv      # win / lose rates
│   ├── m4_accuracy_by_category.csv   # accuracy on original / perturbed docs
│   ├── m4_rr_three_arms.png
│   ├── m5_floor_effect.csv           # standard vs conditional RR, wrong-on-both share
│   ├── m5_answer_behaviour.csv       # NO-RES rate, answer length
│   └── m5_failure_cases_specific.csv # failure cases with document evidence
└── report/                           # M6
    ├── paper.tex / paper.pdf
    ├── RAG_Robustness_M1_M6.pptx
    └── fig_*.png
```

## Reproduce

```bash
git clone https://github.com/Jeet267/rag-robustness-reproduction.git
cd rag-robustness-reproduction
pip install -r requirements.txt
```

Open `nlp_m3_m4_m5.ipynb` on Kaggle (GPU T4 ×2, Internet on) and add your Hugging Face token as the secret `HF_TOKEN` (accept the Llama-3.1 licence first). Set `DRY_RUN = True` for a 2-minute pipeline test; with `DRY_RUN = False`, *Save & Run All* takes about 82 minutes for all three arms. To save time, set `ARMS_TO_RUN = ["baseline", "specific"]`.

## References

1. S. Yang et al., "Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data," ACL 2026. arXiv:2503.05587
2. F. Cuconasu et al., "The Power of Noise: Redefining Retrieval for RAG Systems," SIGIR 2024.
3. O. Yoran et al., "Making Retrieval-Augmented Language Models Robust to Irrelevant Context," ICLR 2024.
4. F. Shi et al., "Large Language Models Can Be Easily Distracted by Irrelevant Context," ICML 2023.
5. N. F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts," TACL 2024.
6. A. Grattafiori et al., "The Llama 3 Herd of Models," arXiv:2407.21783, 2024.

## License


