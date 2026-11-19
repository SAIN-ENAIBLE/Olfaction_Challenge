# Predicting Smell from Molecular Structure

Three prediction tasks on human odor perception data:

| Task | Input | Target | Type |
|---|---|---|---|
| **T1** odor labels | one molecule (SMILES) | 49 binary odor descriptors | multi-label classification |
| **T2** odor profile | one molecule (SMILES) | 41 continuous odor intensities | multi-output regression |
| **T4** mixture similarity | two mixtures of molecules | perceptual distance between them | regression |

## Folder layout

```
T1_odor_labels/
    fold0/ train.csv  valid.csv  test.csv
    ...
    fold4/
T2_odor_profile/
    fold0/ ... fold4/
T4_mixture_similarity/
    fold0/ ... fold4/
    molecules.csv          # CID → SMILES for every molecule used in a mixture
```

All files are comma-separated. `CID` is the PubChem Compound ID of a molecule.

## Splits and how to use them

The splits are **structure-based**. Molecules were grouped into clusters of near-identical structures (molecules differing by about one bond edit). Whole clusters were then assigned to train, validation or test. So test molecules are structurally different from the training molecules, and the score measures how well a model generalizes to *new chemistry*, not how well it recognizes close variants of molecules it has already seen.

The splits are **coordinated across tasks**: within a fold, a T1 test molecule never appears in T2 train/valid, and vice versa. So you may pretrain on T1 and fine-tune on T2, as long as you use the same fold for both.

## T1: odor labels (multi-label classification)

**Columns:** `CID`, `SMILES`, then 49 binary label columns (`Alcoholic`, `Animal`, …, `Woody`). A 1 means the molecule has been described with that odor descriptor in at least one perfumery or psychophysics source. Every molecule has at least one label.

**Metrics** (higher is better; your model should output a score per label, e.g. a probability):

- **LRAP** (label ranking average precision). For each molecule, rank the 49 labels by predicted score. LRAP measures how high the true labels are ranked, averaged over molecules. A per-molecule metric: "are the right odors at the top of the list?"
- **mAP** (macro mean average precision). For each label, compute the average precision (area under the precision–recall curve) over all test molecules, then average across labels. A per-label metric: every label counts equally, including rare ones.

```python
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score, average_precision_score

# Y_true: (n_molecules, 49) array of 0/1;  Y_score: (n_molecules, 49) predicted scores
lrap = label_ranking_average_precision_score(Y_true, Y_score)
mAP = np.mean([average_precision_score(Y_true[:, j], Y_score[:, j])
               for j in range(Y_true.shape[1]) if Y_true[:, j].sum() > 0])
```

**Notes:**
- Labels are very imbalanced: `Fruity` has ~2,600 positives, `Rotten.Decay` only 26. Some labels have only 3–7 positives in a test fold, so their AP is noisy. Look at per-label AP as well as the average.
- A useful sanity check is a baseline that predicts each label's training frequency for every molecule.

## T2: odor profile (multi-output regression)

**Columns:** `CID`, `SMILES`, then 41 descriptor columns (`green`, `fruity`, `sweet`, …). Each value is the **average rating of a panel of human participants**: how strongly that odor quality was perceived, from 0 (not at all) to 5 (very strong).

**Metrics:**

- **NDCG@7** (higher is better). For each molecule, rank the 41 descriptors by predicted value and check whether the 7 top-ranked ones are the descriptors people rated highest, weighting by the true ratings. Averaged over molecules. "Does the model get the dominant notes of the smell right?"
- **Per-descriptor Spearman ρ** (higher is better). For each descriptor, compute the rank correlation between predicted and true values across test molecules, then average over descriptors. "For a given odor quality, does the model rank molecules correctly?"
- **Cosine distance** (lower is better). For each molecule, compute 1 − cos(predicted profile, true profile), then average over molecules. "Is the overall shape of the predicted profile right?" It ignores overall scale.

```python
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from sklearn.metrics import ndcg_score

# Y_true, Y_pred: (n_molecules, 41) arrays
ndcg7 = ndcg_score(Y_true, Y_pred, k=7)
rho = np.nanmean([spearmanr(Y_pred[:, j], Y_true[:, j])[0]
                  for j in range(Y_true.shape[1])
                  if Y_true[:, j].std() > 0 and Y_pred[:, j].std() > 0])
cos_dist = np.mean([cosine(p, t) for p, t in zip(Y_pred, Y_true)])
```

## T4: mixture similarity (regression)

Each row is a **pair of odor mixtures** that people smelled and compared. Each mixture contains 1 to 30 molecules.

**Columns:**

| Column | Meaning |
|---|---|
| `pair_id` | unique ID of the pair |
| `source_dataset` | which experiment the pair comes from (e.g. `Bushdid`, `Snitz 1`, `Ravia 2`). It is an experiment name; `validation` is just the name of one experiment, not a hint about which split to use |
| `mix_A_label`, `mix_B_label` | IDs of the two mixtures **within** their experiment (identifiers, **not** targets) |
| `mixture_A_cids`, `mixture_B_cids` | the molecules in each mixture, as a list of CIDs. Look up SMILES in `molecules.csv` |
| `n_A`, `n_B` | number of molecules in each mixture |
| `distance` | **target**: perceptual distance between the two mixtures, from 0 (smell the same) to 1 (smell completely different) |

**Metrics:**

- **Spearman ρ** (higher is better) between predicted and true distance over all test pairs.
- **RMSE** (lower is better).

```python
import numpy as np
from scipy.stats import spearmanr

rho = spearmanr(y_pred, y_true)[0]
rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
```


# References:
- B. Sanchez-Lengeling, J.N. Wei, B.K. Lee, R.C. Gerkin, A. Aspuru-Guzik, and A.B.
Wiltschko. Machine learning for scent: Learning generalizable perceptual representations
of small molecules. 2019. https://arxiv.org/abs/1910.10685
- B.K. Lee, E.J. Mayhew, B. Sanchez-Lengeling, J/N. Wei, W.W. Qian, K.A. Little,
M.Andres, B.B. Nguyen, T. Moloy, J. Yasonik, J.K. Parker, R.C. Gerkin, J.D. Mainland,
and A.B. Wiltschko. A principal odor map unifies diverse tasks in olfactory perception.
Science, 2023. https://www.science.org/doi/10.1126/science.ade4401
- OpenPOM. https://github.com/biomachinelearning/openpom
- G. Tom, C.T. Ser, E.M. Rajaonson, S. Lo, H.S. Park, B.K. Lee, and B. Sanchez-Lengeling. From Molecules to Mixtures: Learning Representations of Olfactory Mixture Similarity using Inductive Biases. 2025. https://arxiv.org/abs/2501.16271
- Satarifard et al., A semantic-based community model for high-fidelity tuning of olfactory mixture distances. PNAS 2026, https://www.pnas.org/doi/abs/10.1073/pnas.2611057123
- https://github.com/Satarifard/DREAM-olfactory-mixtures-prediction-challenge
- DREAM Olfactory Mixtures Prediction Challenge 2025, https://www.synapse.org/Synapse:syn64743570/wiki/630800
