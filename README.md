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
https://huggingface.co/umamamianoor/cape-cub-pf

Trained for 30 epochs on full CUB-200-2011, batch size 32, lr=1e-3 (per the
released config; the paper text states 1e-4), T_kld=2, SGD — matching
configs/cub/resnet50_PF.py from AIML-MED/CAPE.

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