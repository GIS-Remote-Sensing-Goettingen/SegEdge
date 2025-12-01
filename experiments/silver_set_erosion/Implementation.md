## 1️⃣ DenseCRF / graph-cut on top of your current masks

**Why high priority**

* Classic and still very effective for sharpening segmentation boundaries and removing pepper noise, especially around object edges. Used as post-processing in DeepLab and many weakly-supervised seg methods. ([viso.ai][1])
* Works *exactly* in your setting: you already have a probability/score map per pixel; CRF uses RGB + spatial proximity to pull boundaries to real edges.

**Expected improvement**

* **Moderate to high**: particularly helps where noisy buffered labels or kNN scores bleed into background or miss tight curves.
* Big gain in boundary F-score / edge IoU, less effect on coarse region coverage.

**Difficulty**

* **Low**:

  * Python bindings: `pydensecrf` implements the standard fully-connected DenseCRF from Krähenbühl & Koltun (2011).
  * Pipeline is just: logits / probs → CRF → refined mask.
* You’d just need a small wrapper like `refine_with_crf(img, coarse_mask)` and plug it into your existing `use_crf` flag.

**Verdict**:
👉 **Do this first.** Extremely cheap to add, clearly fits your “label refinement” objective, and plugs in with almost no changes to your code.

---

## 2️⃣ Superpixel-based refinement (e.g. SLIC + voting)

**Why it fits**

* Superpixels (SLIC etc.) group pixels into small, edge-aware regions. ([arXiv][2])
* Very natural for label refinement: enforce that each superpixel is either mostly LWF or mostly background → clean, coherent shapes and fewer salt-and-pepper artifacts.
* Especially good for hedgerows: long skinny superpixels can follow linear canopy structures.

**Expected improvement**

* **Moderate**:

  * De-noises your masks and aligns them better to intensity edges.
  * Can be combined with CRF: superpixel majority vote → CRF smoothing.

**Difficulty**

* **Low–medium**:

  * Implementation: `skimage.segmentation.slic` (or `felzenszwalb`, `quickshift` etc.). ([arXiv][2])
  * Algorithm:

    1. Compute superpixels on RGB tile.
    2. For each superpixel, compute mean DINO-kNN score or majority of coarse mask.
    3. Threshold per superpixel to get a cleaner binary segmentation.
  * Need to handle tiles + stitching, but you already have tiling infrastructure.

**Verdict**:
👉 **Second thing to try.** Easy to integrate, conceptually simple, and very compatible with your “one-class positive bank” idea.

---

## 3️⃣ Feature-space label propagation / graph-based refinement

**What this is**

* Build a graph where:

  * Nodes = patches or pixels (using your DINO features).
  * Edges = similarity (cosine in DINO space, maybe k-NN).
* Propagate labels from high-confidence positives into nearby, similar unlabeled nodes.
* Closely related to **Label Propagation / Label Spreading** in semi-supervised learning. ([scikit-learn.org][3])

**Why it fits the objective**

* You already have:

  * A positive bank (from SH_2022 / Planet).
  * Scores for unlabeled pixels.
* You can:

  * Treat high-score pixels as “seed” positive labels.
  * Let them diffuse through the graph to refine uncertain areas and carve away noisy positives that are feature-wise dissimilar.

**Expected improvement**

* **Moderate**, but more on *global consistency* than edge crispness:

  * Helps ensure that all pixels belonging to the same LWF are consistently labeled.
  * Can fix holes in the interior of features and remove isolated weird patches.

**Difficulty**

* **Medium**:

  * Simple-ish version:

    * Do it at **patch level**, not pixel level.
    * Use scikit-learn’s `LabelSpreading` / `LabelPropagation` on a subset of DINO features (e.g. PCA reduce) with positive / “unknown”. ([scikit-learn.org][3])
  * trickiness:

    * Graph sizes can explode; you will want to run per-tile + maybe sub-sample.
    * Need to choose seeding strategy (e.g. score > 0.9 = positive, < 0.1 = negative, rest unlabeled).

**Verdict**:
👉 **Third in line.** Very aligned with “refine noisy labels using DINO features” and still implementable with off-the-shelf tools, but needs more engineering choices than CRF/superpixels.

---

## 4️⃣ Prototypes + EM-style refinement (DINO prototypical clustering)

**Idea**

* Learn one or few **prototypes** for LWF and for background in the DINO embedding space (similar to Prototypical Networks / few-shot segmentation). ([Medium][4])
* Iteratively:

  1. Assign each patch to the closest prototype.
  2. Update prototypes based on current assignments (EM-style).
  3. Optionally reweight/protect high-confidence positives.
* End result: cleaned/clustered labels with consistent prototypes.

**Why it fits**

* You’re already doing something “prototype-like” with k-NN to a positive bank; this formalizes it:

  * Instead of many noisy positive vectors, compress them into a small set of prototypes.
  * Ambiguous patches that don’t match any prototype well can be dropped or flagged for manual review.

**Expected improvement**

* **Potentially high**, but more uncertain:

  * Can strongly denoise positives and make them morphologically & semantically more coherent.
  * Works well if LWF class is reasonably compact in feature space.

**Difficulty**

* **Medium–high**:

  * You’d implement clustering (k-means / GMM / simple EM) on DINO features for positive & negative.
  * Then reassign labels based on distances / responsibilities.
  * Extra design choices: number of prototypes, weighting, convergence criteria.
* The math is simple, but debugging the behavior on real geodata takes time.

**Verdict**:
👉 **Fourth.** Worth trying once you have CRF + superpixels + possibly simple label propagation in place, especially if you want an “embedding space view” for your thesis.

---

## 5️⃣ Teacher–student self-training / pseudo-labeling loop

**What it is**

* Take your current zero-shot + refinement pipeline as a **teacher**.
* Generate refined pseudo-labels on a large pool of DOP20 tiles.
* Train a **segmentation student** (U-Net / Mask2Former head on top of DINO) on these pseudo-labels.
* Iterate:

  * Student predicts on unlabeled data.
  * Filter high-confidence pixels → new pseudo-labels.
  * Retrain or fine-tune student.
* This is the core of many modern **semi-supervised segmentation** approaches, with lots of variants (class-wise thresholds, consistency regularization, etc.). ([OpenReview][5])

**Why it fits your thesis**

* Perfect for your “human-in-the-loop, scalable refinement” story:

  * Teacher: DINO + kNN + CRF + SH_2022 clipping.
  * Student: simple U-Net / Mask2Former head trained end-to-end on your VHR imagery.
* Can also plug into your XAI section very naturally (attention maps, feature embeddings etc.).

**Expected improvement**

* **Potentially very high**, especially over **time**:

  * Once you have a decent initial soft mask, a trained student typically learns sharper, more coherent boundaries than hand-crafted kNN + CRF alone. ([OpenReview][5])
  * You also get a fast feed-forward model for large-scale inference (instead of per-tile kNN).

**Difficulty**

* **High**:

  * Need a full training pipeline:

    * Dataset abstractions, tiles, augmentations.
    * Checkpointing, validation on a golden set.
  * Need to design pseudo-label filtering rules (confidence thresholds, ignore index).
  * Risk of confirmation bias / drift if teacher is bad.

**Verdict**:
👉 **Last step**, but also the most “research-grade” and aligned with a thesis-level contribution once the rest of the refinement stack is stable.

[1]: https://viso.ai/deep-learning/deeplab/?utm_source=chatgpt.com "DeepLab: Pioneering Semantic Segmentation Techniques"
[2]: https://arxiv.org/abs/2210.16829?utm_source=chatgpt.com "Self-Regularized Prototypical Network for Few-Shot ..."
[3]: https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_digits.html?utm_source=chatgpt.com "Label Propagation digits: Demonstrating performance"
[4]: https://medium.com/%40MarouaMaru7/prototypical-ne-5425ee8b25d3?utm_source=chatgpt.com "Prototypical networks. 1. Introduction : | by MAROUA CHANBI"
[5]: https://openreview.net/forum?id=-TwO99rbVRu&utm_source=chatgpt.com "Designing Pseudo Labels for Semantic Segmentation"
