# CAPE reproduction — Milestones 3 & 4: the evaluation fix and the mask-convention experiment

Paper: Chowdhury et al., *CAPE: CAM as a Probabilistic Ensemble for Enhanced DNN Interpretation*, CVPR 2024.
Setting: ResNet-50, CUB-200-2011, post-fitted (PF) CAPE head, 14x14 explanation maps.

> Items marked **[TODO]** need a number or confirmation from our own runs before submission.

---

## 1. Summary

1. Our first full evaluation was wrong. It scored CAPE and mu-CAPE maps with the *vanilla* head's confidence. Scoring them with the *CAPE* head, which is the head they explain, moved CAPE's Average Drop (AD) from 57.7 to 33.6 and mu-CAPE's from 22.8 to 11.2. The CAM-family rows and every mIoU value did not change.
2. The learning-rate explanation we gave for the original gap is retracted. The lr = 1e-4 rerun went through the same faulty evaluator, so it is inconclusive.
3. With the fix, some of the paper's claims hold in direction and some do not. Section 5 gives the claim-by-claim verdicts.
4. The experiment asks how much AD rankings depend on the mask convention. AD is mostly a function of how much of the image a mask keeps. At matched kept area, CAPE sits on roughly the same AD curve as the CAM family.
5. The paper's CAPE-vs-CAM AD comparison pairs each method with a different head. The same CAM maps score 24.3 on the vanilla head and 13.0 on the CAPE head, so that pairing mixes a head effect with a method effect.

---

## 2. The fix

**What was wrong.** `metrics.py` read `out['orig']['outcome'].softmax(...)` for every method. CAPE and mu-CAPE explain the CAPE head's output, so their confidence should come from `out['cape']['outcome']`, which is already a probability and must not be softmaxed again.

**What changed.** The rewritten `metrics.py` picks the head per method: the CAPE head for CAPE and mu-CAPE, and the vanilla head for CAM, Grad-CAM and Grad-CAM++. Nothing else substantive changed.

**Why we believe the fix is the cause and not just a lucky change:**

- CAM, Grad-CAM and Grad-CAM++ rows are identical before and after (for example CAM AD 24.40 both times).
- mIoU is identical for all five methods, as expected, because it depends only on the maps.
- Only the CAPE and mu-CAPE rows moved.
- The mask-convention experiment (Section 4) contains an independent regression check. Under the soft mask, the vanilla-head CAPE AD is 57.3 (old evaluator: 57.7) and the CAPE-head AD is 34.1 (corrected evaluator: 33.55).
- The paper says the CAPE explanation should not be used to explain the vanilla classifier (Discussion, "CAPE explains itself"). Scoring CAPE maps with the vanilla head went against that.

---

## 3. What the earlier results implied, and why we retract it

The first evaluation showed CAPE far less faithful than the paper (AD 57.7 vs 22.2, IC 6.7 vs 26.5). We blamed the mismatch between the paper's PF learning rate (1e-4) and the released config (1e-3), and a rerun at 1e-4 appeared to "rule out" the learning rate while pointing at the masking convention.

Both parts of that story were wrong:

- The 1e-4 rerun was scored with the faulty evaluator. All it showed was that the learning rate had no detectable effect *under that evaluation*.
- The "masking convention" suspicion was in the right area but vague. The real cause was a specific bug (wrong head).
- **[TODO]** If the lr = 1e-4 checkpoint still exists, re-evaluate it with the corrected `metrics.py` and report the numbers here. That turns the question into a real result.

---

## 4. Corrected results

### 4.1 Full test set (n = 5,794), corrected evaluator

| Method | AD ↓ | IC ↑ | ADD ↑ | ADCC ↑ | mIoU ↓ | BC |
|---|---|---|---|---|---|---|
| CAM (paper) | 21.2 | 27.9 | 67.4 | 78.8 | 75.9 | n/a |
| CAM (ours) | 24.40 | 24.46 | 64.96 | 78.00 | 74.91 | 4 |
| Grad-CAM (ours) | 24.40 | 24.42 | 64.96 | 77.99 | 74.91 | 2 |
| Grad-CAM++ (ours) | 22.18 | 26.23 | 67.66 | 78.17 | 90.40 | 8 |
| CAPE PF (paper) | 22.2 | 26.5 | 68.7 | 73.7 | 13.4 | 3 |
| CAPE PF, first evaluation | 57.70 | 6.70 | 40.03 | 52.90 | 26.97 | n/a |
| CAPE PF (ours, corrected) | 33.55 | 25.61 | 52.04 | 74.66 | 26.97 | 4 |
| mu-CAPE PF (paper) | 15.9 | 30.9 | 69.6 | 83.0 | 66.6 | 5 |
| mu-CAPE PF, first evaluation | 22.80 | 25.80 | 67.42 | 78.41 | 83.29 | n/a |
| mu-CAPE PF (ours, corrected) | 11.24 | 46.96 | 71.48 | 85.74 | 83.29 | 12 |

How to read this:

- **CAM family.** AD, IC, ADD and ADCC are within a few points of the paper. CAM and Grad-CAM matching each other is a correctness check on our Grad-CAM, because on a global-average-pooling head the two are mathematically the same up to scale. The paper lists mIoU 100.0 for Grad-CAM and Grad-CAM++, which class-specific gradients cannot produce for two different classes. We got 74.9 and 90.4. We do not know how the paper computed it and did not pursue it.
- **CAPE.** IC and ADCC are close to the paper. AD (33.6 vs 22.2) and ADD (52.0 vs 68.7) are not. mIoU is about twice the paper's (27.0 vs 13.4), although it is still by far the lowest of our five methods.
- **mu-CAPE.** ADD and ADCC are close to the paper. AD and IC are better than the paper's (IC 47.0 vs 30.9, a large gap we cannot fully explain), and mIoU is higher.
- **Borda Count is not comparable to the paper's.** The paper ranks 12 methods per dataset and we rank 5. CAM and Grad-CAM tie on several metrics and our sort breaks ties by list order, so their BC difference (4 vs 2) is tie-breaking noise.
- **Sanity checks. [TODO]** Report (a) test accuracy of both heads (the paper's Table 2 gives 83.34 vanilla and 82.12 for CAPE (PF)), and (b) the output of the sum-to-confidence check below.

```python
with torch.no_grad():
    out = net(image_tensor.unsqueeze(0))
c = pred_class
print('sum of CAPE map :', out['cape']['weighted_contribution'][0, c].sum().item())
print('cape outcome[c] :', out['cape']['outcome'][0, c].item())
print('cape outcome sum:', out['cape']['outcome'][0].sum().item())
```

CAPE's defining claim is that region contributions for a class add up to the model's confidence in that class. If the first two numbers match, that is direct evidence our implementation reflects the method. **[TODO: paste result.]**

---

## 5. Experiment: how much does the AD ranking depend on the mask?

**Question.** How much does the ranking of explanation methods under AD depend on how a heatmap is turned into a mask?

**Setup** (`experiments/mask_convention_experiment.py`; n = 1,000 test images, a nested prefix of the 5,794; seed 123; commit **[TODO: hash]**):

- Three mask families: the soft map itself (what the paper's definition uses), binary thresholds (keep cells >= theta x max, theta from 0.05 to 0.70), and same-area top-k (keep exactly the top K% of the 14x14 cells, K from 5 to 70).
- A Random control keeps K% of cells chosen at random.
- Every masked image is scored by both heads, so methods can be compared under the same head.
- 95% paired-bootstrap intervals over images. Treat these as descriptive, because many unadjusted comparisons are involved.
- Outputs: `ad_summary.csv`, `paired_differences.csv`, `ad_vs_mask_convention.png`, `kept_area_vs_theta.png`.

### 5.1 Findings

**AD is mostly about how much image the mask keeps.** Under the soft mask, CAPE keeps 9.0% of the image against 21.4% for CAM, because CAPE's maps are concentrated. Matching on kept area with binary-threshold masks, CAPE and CAM sit on nearly the same curve:

| Head | CAPE (theta, kept area, AD) | CAM at similar area (theta, kept area, AD) |
|---|---|---|
| Vanilla | 0.05, 28.7%, 27.9 | 0.30, 28.4%, 27.4 |
| Vanilla | 0.10, 22.1%, 35.4 | 0.40, 21.6%, 34.6 |
| Vanilla | 0.30, 11.2%, 58.0 | 0.60, 11.0%, 57.5 |
| CAPE | 0.05, 28.7%, 6.7 | 0.30, 28.4%, 7.8 |
| CAPE | 0.10, 22.1%, 9.7 | 0.40, 21.6%, 10.6 |
| CAPE | 0.30, 11.2%, 26.2 | 0.60, 11.0%, 24.6 |

These pairs were matched by hand. A full AD-vs-kept-area plot from `ad_summary.csv` would show the same thing more cleanly **[TODO: add this figure]**.

**With exactly equal area (top-k), CAPE's gap mostly disappears.** CAPE minus Grad-CAM, CAPE head:

| K | 5% | 10% | 20% | 30% | 50% | 70% |
|---|---|---|---|---|---|---|
| Difference (AD points) | +4.2 | +2.9 | +0.3 | +0.1 | +0.2 | +0.1 |

The intervals include 0 at K = 20, 30 and 70. At K = 5-10 CAPE's top cells are slightly worse than CAM's, so the effect is not purely an area artifact. At large K every method's AD falls toward 2% (a floor effect), so K = 5-30 is the informative range. At large K CAPE's sparse maps also run out of non-zero cells, and ties are then broken in raster order.

**Every method is far better than random.** At K = 30%, AD is about 26-28% for the methods and 97.7% for Random (vanilla head).

**Rankings change with the convention.** On the vanilla head Grad-CAM++ is best under the soft mask and most thresholds, while CAM is best under top-k for K >= 10. CAPE is last under every soft and threshold condition on both heads, but third of five under top-k at K >= 30.

**Head choice matters about as much as method choice.** The same CAM maps score AD 24.3 on the vanilla head and 13.0 on the CAPE head. Same-head comparisons of mu-CAPE and CAM:

| Head | mu-CAPE | CAM |
|---|---|---|
| CAPE head | 11.5 | 13.0 |
| Vanilla head | 22.7 | 24.3 |

mu-CAPE's AD edge over CAM is therefore about 1.5 points on either head. The 13-point gap in our main table (11.2 vs 24.4) comes mostly from pairing each method with a different head. A plausible reason is that the CAPE head's outputs are softer (the paper notes CAPE's "soft prediction confidence"), which would shrink relative drops, but we did not test this.

---

## 6. Does this uphold or break the paper's claims?

| Paper claim | Verdict |
|---|---|
| Per-region CAPE values sum to the class confidence | **[TODO]** Pending the sum-to-confidence check. |
| CAPE is more class-discriminative than CAM (low mIoU) | **Reproduced in direction.** 27.0 vs 74.9 (paper: 13.4 vs 75.9). Magnitude differs. |
| CAPE's faithfulness is comparable to CAM's (AD 22.2 vs 21.2) | **Not reproduced under the soft mask** (CAPE 33.6, CAM 24.4). At matched kept area the two are close, so the gap mostly reflects how little image CAPE's soft mask keeps. This does not explain why the paper's own CAPE does better than ours (see Limitations). |
| mu-CAPE is the strongest method | **Weakly supported.** It has the top BC among our five, but BC is not comparable to the paper's. Its same-head AD edge over CAM is about 1.5 points. In the paper's Table 1, the strongest result belongs to mu-CAPE (TS); mu-CAPE (PF), which we reproduce, scores BC 5 on CUB, below Score-CAM's 6. |
| CAM and Grad-CAM behave almost identically on this architecture | **Reproduced.** |

Overall, our evidence neither breaks nor confirms the paper. What it shows is that conclusions about CAPE's faithfulness depend on the mask convention and on which head scores the mask.

---

## 7. Limitations

- **One checkpoint**, one seed, and n = 1,000 for the experiment (nested within the 5,794). Intervals are over images only.
- **Section 4.3 does not say which head Ψ refers to.** Pairing CAPE-family methods with the CAPE head and the CAM family with the vanilla head is our reading, supported by the paper's statement that CAPE should not be used to explain the vanilla classifier.
- **Masking details.** We mask in ImageNet-normalized space (so "removed" means mean colour, not black), normalize the 14x14 map and then upsample it bilinearly, while the paper describes upsampling and then min-max normalizing. Top-k masks are blocky because of nearest-neighbour upsampling. Any of these could contribute to the remaining gap to the paper.
- **The remaining CAPE gap is unresolved.** The paper's CAPE has a lower mIoU and a better AD than ours. Sparsity alone cannot explain that, so something else differs (pipeline details, temperature, or training).
- **mIoU thresholds.** mIoU uses 0.2 x max on the 14x14 map. We did not sweep it.
- **`best.pth` selection.** It is chosen by test accuracy and then evaluated on the same test set.
- **PF initialization.** **[TODO: confirm.]** `train_pf.py` loads the full TS state dict, so the CAPE head starts from its TS-trained weights. The paper describes initializing the CAPE layer from the vanilla classifier. State which one our runs did.

---

## 8. Reproducing

```bash
python src/metrics.py --checkpoint <best.pth> --processed_dir <dir with test.pkl> --num_images 5794
python experiments/mask_convention_experiment.py --checkpoint <best.pth> \
    --processed_dir <dir with test.pkl> --output_dir results/mask_convention --num_images 1000
```

Checkpoints and logs: Hugging Face (links in the main README). Provenance: see `PROVENANCE.md`. Metrics code is written by us from the paper's Section 4.3. CAM, CAPE and mu-CAPE maps come from the authors' model code, and Grad-CAM and Grad-CAM++ are our implementations.
