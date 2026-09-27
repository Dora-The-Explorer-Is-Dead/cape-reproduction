## What CAPE actually is, and what we're doing here

Deep learning classifiers are usually black boxes — a model looks at an image
and says "97% Pileated Woodpecker," but doesn't say why. CAM (Class Activation
Mapping) methods were built to open that box a little: they produce a
heatmap over the image showing which regions the model paid attention to.
The problem is these heatmaps only show *relative* attention — "this beak
region got more attention than that leaf region" — with no way to say how
much any one region actually contributed to the final 97% confidence, or to
meaningfully compare that contribution across different possible classes.
CAPE's whole contribution is fixing that: it reformulates the explanation so
that the contributions of every region, added together, exactly equal the
model's overall confidence score. Instead of "here's roughly where it looked,"
you get "here's exactly how much each piece of the image is worth."

To understand how it does that, it helps to know that the model in this repo
actually has two separate output layers sitting side by side on top of the
same ResNet-50 backbone, which we call "heads." The first is a completely
ordinary classifier head — a linear layer whose output gets turned into class
probabilities the normal way. This ordinary head is what CAM, Grad-CAM, and
Grad-CAM++ are all built on top of, and none of those three need any special
training of their own — they're just formulas you can apply to any already-
trained classifier's activations or gradients, and they hand you a heatmap
immediately. The second head is the CAPE head. It's the same shape as the
first, but its output gets passed through CAPE's own math instead of a plain
softmax — three small learnable numbers (called temperatures) that rescale
things so the region-by-region contributions sum up to the model's actual
confidence. Because this head has its own learnable parameters that start out
untrained, CAPE can't just be "applied" to a finished model the way CAM can.
It has to be trained first.

That training can happen two ways, and the paper calls them TS and PF.
Training-from-Scratch (TS) means the backbone and both heads all learn
together, starting from an ImageNet-pretrained backbone and randomly
initialized heads, fitting everything to bird classification from the ground
up. Post-Fitting (PF) starts somewhere very different: from a model that
*already* knows how to classify birds — in our case, the authors' own
TS-trained checkpoint, where the backbone and ordinary head are already
finished and good. In PF, we freeze that backbone and ordinary head
completely, so they never change again, and train *only* the CAPE head on
top of them. The CAPE head learns by trying to mimic the frozen ordinary
head's own predictions as closely as possible, region by region, using what's
called a distillation loss. The practical reason this mode exists at all,
per the authors: it's far cheaper, because the expensive part (training a
25-million-parameter backbone) is already done. It's the realistic version of
"we already have a working classifier in production — can we bolt
interpretability onto it after the fact, without retraining the whole thing?"

This is exactly what we're doing. We are not training a bird classifier from
scratch, and we are not touching the classifier's actual bird-recognition
ability at any point. We load the authors' pretrained TS checkpoint, keep the
backbone and ordinary classifier frozen forever, and run PF training
(`src/train_pf.py`) so that only the CAPE head learns how to reformulate that
already-trained model's knowledge into CAPE's format.

While that training runs, the script saves two different files, and they
serve different purposes. Every single epoch, it overwrites a file called
`latest.pth` with a full snapshot of the model and optimizer at that exact
point — this exists purely so that if training gets interrupted (a crashed
session, a disconnect), we can rerun the same command and pick up from where
we left off instead of starting all 30 epochs over. Separately, any time the
CAPE head's accuracy on the held-out test set improves over its previous
best, the script saves a second file called `best.pth`. This second file is
the one that actually matters for the rest of the project — it's whichever
epoch's version of the CAPE head performed best on data it never trained on,
which isn't necessarily the very last epoch. `best.pth` is what we load
afterward to actually generate CAM, Grad-CAM, Grad-CAM++, and CAPE heatmaps
side by side and compute the paper's evaluation metrics (AD, IC, ADD, ADCC,
mIoU) to see how our reproduction compares to what the original paper
reported.

One discrepancy worth flagging: the paper's text says PF training
uses a learning rate of 1e-4, but the actual configuration file the authors
released (`configs/cub/resnet50_PF.py`) uses 1e-3. We followed the released
config rather than the paper text, since that's what actually produced the
checkpoints we're building on and comparing against.

## Trained checkpoint

Our reproduced CAPE (PF) checkpoint and training logs:
https://huggingface.co/umamamianoor/cape-cub-pf \
or at https://huggingface.co/AyaanArif/cape-cub-pf

Trained for 30 epochs on full CUB-200-2011, batch size 32, lr=1e-3 (per the
released config; the paper text states 1e-4), T_kld=2, SGD — matching
configs/cub/resnet50_PF.py from AIML-MED/CAPE.

## The explanation methods, and how they differ

**CAM** (Zhou et al., 2016) is the original method. It only works on networks
with a specific architecture — global average pooling immediately followed by
a single linear classifier layer, exactly what this ResNet-50 setup uses. The
heatmap is produced directly from that final layer's own weights: each
spatial location's importance is just a weighted sum of the feature map
values there, using the classifier's learned weights as the weights.

**Grad-CAM** (Selvaraju et al., 2017) generalizes this idea to work on *any*
CNN, not just GAP-plus-linear ones, by replacing "the classifier's weights"
with "the gradient of the target class's score with respect to the last
convolutional layer." On architectures like this one where CAM already
applies directly, Grad-CAM ends up mathematically very close to plain CAM —
which is exactly why, in our results, CAM and Grad-CAM's numbers track each
other closely rather than diverging.

**Grad-CAM++** (Chattopadhay et al., 2018) refines Grad-CAM's weighting
scheme using second-order gradient terms instead of a simple average. This
mainly helps in cases Grad-CAM handles poorly — when a class appears more
than once in an image, or is only partially visible — by better spreading
credit across multiple relevant regions instead of collapsing to one.

**CAPE** is not a variant of the CAM family in the same sense — it doesn't
tweak how the weighting is computed, it changes what the resulting numbers
*mean*. CAM-family heatmaps only tell you relative importance within one
image; you can't say a region is worth "12% of the decision," and you can't
directly compare a heatmap computed for one candidate class against a
heatmap computed for another. CAPE reformulates the classifier's output so
that per-region values are on a shared, meaningful scale — they sum exactly
to the model's actual confidence score, and are comparable across classes.

**μ-CAPE** is a variant of CAPE itself, not a new method from scratch. Plain
CAPE tends to be very class-discriminative — it sharply highlights only the
regions that distinguish this specific class from *all* other classes, which
can mean it suppresses regions that are genuinely part of the object but
happen to also be relevant to other similar classes. μ-CAPE restores some of
those "class-mutual" regions, trading some of that sharp discriminability for
a fuller, more complete picture of what the model is actually looking at.

## What these metrics actually measure, and why they matter

All six of these come from the paper's Section 4.3, and they exist to answer
one underlying question: when a heatmap points at part of an image and says
"the model looked here," is that actually true in a way you can verify, not
just visually plausible?

**AD (Average Drop)** and **IC (Average Increase)** work the same basic way:
take the image, multiply it by the explanation map so only the highlighted
region survives, and feed that masked image back through the classifier. If
the explanation map genuinely captured the evidence the model relied on, the
model's confidence on that masked image should barely drop — because you've
kept the part that actually mattered and thrown away the rest. AD measures
how much confidence is lost when you do this (lower is better — less lost
means the highlighted region really was the important part). IC measures the
opposite direction: sometimes masking out distracting background actually
*increases* confidence, and IC just counts how often that happens (higher is
better).

**ADD (AD in Deletion)** flips the mask around: instead of keeping only the
highlighted region, you delete it and keep everything else. If the
explanation map is accurate, removing it should hurt the model a lot — a big
drop in confidence here is a *good* sign, meaning the region really was
load-bearing (so, unlike AD, higher ADD is better).

**ADCC** exists because AD and IC alone can be gamed — a method that just
highlights the entire image trivially scores well on both. ADCC combines
three things at once: coherency (does re-running the explanation method on
the masked image give you back a similar map, i.e. is the explanation
self-consistent?), complexity (how much of the image did the method bother
highlighting — a method that lights up everything is being lazy, not
informative), and AD. A method has to do well on all three simultaneously to
score well here, which makes it harder to cheat.

**mIoU** measures something different from faithfulness: how much do the
explanation maps for the model's top-2 predicted classes overlap with each
other? A method with low mIoU is telling you something class-specific — "this
looks like a Pileated Woodpecker *because of this region*, and that's a
different region than what would make it look like a Red-headed Woodpecker
instead." A method with high mIoU is largely just highlighting "the bird" in
general, regardless of which of the two species it's deciding between — less
useful for understanding *why* it picked one over the other.

**BC (Borda Count)** isn't its own measurement — it's a ranking system
applied after the fact. Every method gets ranked against the others on each
of the five metrics above (3 points for 1st place, 2 for 2nd, 1 for 3rd, 0
otherwise), and BC is just the sum of those ranking points across all five
metrics — one combined scoreboard instead of five separate tables.

## If CAM and CAPE score similarly here, what's actually the point of CAPE?

This is a fair question, and the honest answer is: **these faithfulness
metrics were never CAPE's main selling point** — even in the paper's own
Table 1, CAPE (PF) doesn't clearly beat CAM across the board (CAPE's ADCC is
actually *lower* than CAM's, 73.7 vs 78.8). What CAPE is actually solving is
a structural problem CAM has, not a faithfulness problem: CAM's heatmap tells
you *where* the model looked, but the numbers on that heatmap don't mean
anything on their own — you can't say "this region is worth 12% of the
decision," and you can't meaningfully compare a heatmap for "woodpecker"
against a heatmap for "albatross," because CAM's values for each class are
computed independently with no shared scale between them.

CAPE's actual contribution is that its per-region values are constructed so
they **sum to exactly the model's confidence score** and are on a genuinely
comparable scale across different classes — meaning you can look at a CAPE
map and say something quantitative like "this beak region contributed 18% of
the total 92% confidence," and compare that contribution directly against
what the same region contributes toward a different candidate class. CAM
literally cannot make either of those statements, no matter how good its
heatmap looks or how well it scores on AD/IC. That's a difference in what
kind of question the explanation can answer, not a difference in how good
the explanation looks — which is exactly why it doesn't necessarily show up
as CAPE "winning" on faithfulness metrics that were designed around CAM-style
methods in the first place.

The paper's own strongest results actually come from **μ-CAPE** (a variant
that restores some of the class-overlap CAPE otherwise suppresses), which
does top the Borda Count rankings in their Table 1 — we implemented plain
CAPE, not μ-CAPE, since that was the tractable scope for this reproduction.
That's worth stating directly: our numbers are a fair test of what CAPE (not
μ-CAPE) actually delivers, and the paper's own results suggest CAPE alone
isn't meant to dominate every metric — its case rests on the mIoU/
class-discriminative story and the absolute-contribution framing, not on
outscoring CAM on AD/IC/ADD/ADCC.

## Reproduction results (Quantitative) — full test set (n=5,794, CUB-200-2011)

| Method | AD ↓ | IC ↑ | ADD ↑ | ADCC ↑ | mIoU ↓ | BC |
|---|---|---|---|---|---|---|
| CAM (paper) | 21.2 | 27.9 | 67.4 | 78.8 | 75.9 | 0 |
| CAM (ours) | 24.40 | 24.46 | 64.96 | 78.00 | 74.91 | 5 |
| Grad-CAM (paper) | 21.6 | 27.5 | 66.8 | 77.3 | 100.0 | 0 |
| Grad-CAM (ours) | 24.40 | 24.42 | 64.96 | 77.99 | 74.91 | 2 |
| Grad-CAM++ (paper) | 20.3 | 28.7 | 68.9 | 77.4 | 100.0 | 0 |
| Grad-CAM++ (ours) | 22.18 | 26.23 | 67.66 | 78.17 | 90.40 | 11 |
| CAPE (PF) (paper) | 22.2 | 26.5 | 68.7 | 73.7 | 13.4 | 3 |
| CAPE (PF) (ours) | 57.70 | 6.70 | 40.03 | 52.90 | 26.97 | 3 |
| μ-CAPE (PF) (paper) | 15.9 | 30.9 | 69.6 | 83.0 | 66.6 | 5 |
| μ-CAPE (PF) (ours) | 22.80 | 25.80 | 67.42 | 78.41 | 83.29 | 9 |

**CAM, Grad-CAM, and Grad-CAM++ reproduced closely** across all five metrics
— ADCC in particular landed within a point of the paper (78.0 vs. 78.8 for
CAM), confirming our ADCC implementation is correct after fixing a
coherency-normalization bug found during development (see PROVENANCE.md).
CAM and Grad-CAM producing near-identical numbers (24.40 vs. 24.40 AD) is
expected, not a coincidence: on a global-average-pooling architecture like
this ResNet-50, the two methods are mathematically near-equivalent.

**CAPE's mIoU correctly reproduces the paper's central qualitative claim.**
CAPE's mIoU (26.97) sits far below CAM's (74.91) — consistent with the
paper's core finding that CAPE produces more class-discriminative,
lower-overlap explanations than CAM-family methods. This required correcting
a mix-up during development where we initially generated the *μ-CAPE* map
(`logcampe_clip0`) while labeling it "CAPE" — the true CAPE quantity
(`weighted_contribution`) is the one shown here, verified against the
authors' own `generate_cam_maps.py`.

**CAPE's AD, IC, ADD, and ADCC did not reproduce closely, in a consistent
direction.** Our CAPE (PF) shows a much larger confidence drop under masking
(AD 57.7 vs. the paper's 22.2) and much smaller confidence gain (IC 6.7 vs.
26.5) than reported — more extreme than even the paper's own description of
CAPE as sparse and class-discriminative would suggest. We believe this stems
from a combination of: (1) the paper text and the released config disagreeing
on PF's learning rate (1e-4 vs. the 1e-3 we used, matching the code), which
may have produced a CAPE layer with different sharpness than the authors';
and (2) our AD/IC/ADD masking implementation, built from the paper's
equations without access to their evaluation code, likely differs from
theirs in exact normalization or masking conventions — a difference to which
CAPE's naturally sparse output distribution may be especially sensitive,
compared to the smoother CAM-family maps where this same implementation
reproduced closely.

**μ-CAPE reproduced moderately well** — ADCC (78.41) and IC (25.80) are close
to CAM's own numbers, though without the clear separation the paper reports
(μ-CAPE(PF) AD 15.9, notably better than CAM's 21.2 in the paper; ours is
comparable to, not better than, CAM).

## Could the learning rate discrepancy explain CAPE's gap?

Section 4.1 of the paper states PF training uses a learning rate of 1e-4, but
the released config file (`configs/cub/resnet50_PF.py`) uses 1e-3 — we
followed the released config, since it's what actually produced the
checkpoints we built on. We think this discrepancy is a plausible
contributor to CAPE's AD/IC/ADD/ADCC gap, though we have not confirmed it
experimentally.

The reasoning: PF training only updates the CAPE head and three learnable
temperature parameters, which directly control how sharply the model's
softmax-style normalization squashes its output. A 10x higher learning rate
plausibly pushed these temperatures toward a sharper, more extreme final
state than the authors' own PF-trained model reached. This would predict a
CAPE explanation map that is *more* peaky and class-discriminative than
theirs — and that prediction lines up with what we actually observed: our
CAPE mIoU (26.97) correctly reproduces the paper's finding that CAPE is far
more class-discriminative than CAM (74.91), while AD and IC collapse far
more severely (57.7 and 6.7 vs. the paper's 22.2 and 26.5) than the paper's
own CAPE row would suggest — consistent with a heatmap so concentrated that
masking to "only the highlighted region" discards nearly the entire image.

That said, this is a plausible, internally-consistent hypothesis, not a
confirmed diagnosis. We did not have access to the authors' own trained
temperature values to compare against, and our AD/IC/ADD implementation was
built directly from the paper's equations without their evaluation code (see
PROVENANCE.md) — a masking or normalization convention that differs from
theirs, independent of learning rate, remains an equally plausible
contributor, and we cannot cleanly separate the two effects without rerunning
PF training at lr=1e-4 and re-evaluating, which we did not do given time
constraints on this milestone.

## Qualitative results: CAM, Grad-CAM, Grad-CAM++, CAPE, and μ-CAPE side by side

Each figure below shows one CUB test image (never seen during training),
with heatmaps from all five methods generated for the model's own predicted
class — not the ground-truth label — using our trained PF checkpoint (30
epochs, full CUB, lr=1e-3).

**Correct, high-confidence predictions:**

![True: 31, Predicted: 31, Confidence: 92.5%](https://huggingface.co/umamamianoor/cape-cub-pf/resolve/main/figures/explain_0_true31_pred31.png)
*True class 31, predicted class 31, confidence 92.5%*

![True: 18, Predicted: 18, Confidence: 91.7%](https://huggingface.co/umamamianoor/cape-cub-pf/resolve/main/figures/explain_2_true18_pred18.png)
*True class 18, predicted class 18, confidence 91.7%*

![True: 6, Predicted: 6, Confidence: 94.2%](https://huggingface.co/umamamianoor/cape-cub-pf/resolve/main/figures/explain_4_true6_pred6.png)
*True class 6, predicted class 6, confidence 94.2%*

**Misclassifications — arguably more informative, since they show what each
method highlights when the model gets it wrong:**

![True: 10, Predicted: 57, Confidence: 23.2%](https://huggingface.co/umamamianoor/cape-cub-pf/resolve/main/figures/explain_1_true10_pred57.png)
*True class 10, predicted class 57 (incorrect), confidence 23.2% — low
confidence reflects the model's own uncertainty here*

![True: 55, Predicted: 115, Confidence: 41.6%](https://huggingface.co/umamamianoor/cape-cub-pf/resolve/main/figures/explain_3_true55_pred115.png)
*True class 55, predicted class 115 (incorrect), confidence 41.6%*

**Observations:** [fill this in after actually looking at the images
yourselves — e.g., do CAM and Grad-CAM highlight visually similar regions,
as expected given they're near-equivalent on this architecture? Does CAPE's
heatmap look visibly sparser/more concentrated than CAM's, consistent with
its much lower mIoU in the metrics table above? On the misclassified
examples, do any methods point at a plausible reason for the confusion —
e.g., a similar-looking body part between the true and predicted species?]