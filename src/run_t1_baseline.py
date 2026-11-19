"""T1 fold-0 baseline: CheMeleon SMILES embeddings and three classifiers.

Train on data/T1_odor_labels/fold0/train.csv, choose hyperparameters on
valid.csv, and report LRAP and macro mAP on test.csv.

The three trained models are logistic regression, histogram gradient boosting,
and a multilayer perceptron. A label-frequency predictor is scored alongside
them as the task's sanity-check reference.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import torch
from rdkit import Chem
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemeleon_fingerprint import CheMeleonFingerprint  # noqa: E402

DATA_DIR = ROOT / "data" / "T1_odor_labels" / "fold0"
OUT_DIR = ROOT / "results" / "t1_fold0"
FIG_DIR = OUT_DIR / "figures"
SEED = 0

MODEL_ORDER = [
    "label_frequency",
    "logistic_regression",
    "gradient_boosting",
    "mlp",
]
MODEL_LABELS = {
    "label_frequency": "Label frequency",
    "logistic_regression": "Logistic regression",
    "gradient_boosting": "Gradient boosting",
    "mlp": "Multilayer perceptron",
}
MODEL_COLORS = {
    "label_frequency": "#7F7F7F",
    "logistic_regression": "#0072B2",
    "gradient_boosting": "#E69F00",
    "mlp": "#009E73",
}


def load_split(name: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    frame = pd.read_csv(DATA_DIR / f"{name}.csv")
    label_cols = [c for c in frame.columns if c not in ("CID", "SMILES")]
    labels = frame[label_cols].to_numpy(dtype=np.int8)
    return frame, labels, label_cols


def embed_unique(smiles: list[str], batch_size: int = 256) -> np.ndarray:
    cache_path = OUT_DIR / "chemeleon_embeddings.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_smiles = cached["smiles"].astype(str)
        if list(cached_smiles) == list(smiles):
            print(f"Loaded cached embeddings {cached['X'].shape} from {cache_path}")
            return cached["X"]

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    print(f"Loading CheMeleon fingerprint model on {device}")
    fingerprinter = CheMeleonFingerprint(device=device)

    chunks = []
    n = len(smiles)
    started = time.perf_counter()
    for start in range(0, n, batch_size):
        batch = smiles[start : start + batch_size]
        chunks.append(fingerprinter(batch))
        done = min(start + batch_size, n)
        print(f"  embedded {done}/{n}", flush=True)
    matrix = np.vstack(chunks).astype(np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, smiles=np.array(smiles), X=matrix)
    print(f"Embeddings {matrix.shape} saved in {time.perf_counter() - started:.1f}s")
    return matrix


def scores_for(model, x: np.ndarray) -> np.ndarray:
    if isinstance(model, np.ndarray):
        return np.broadcast_to(model, (len(x), model.shape[0])).copy()
    probabilities = model.predict_proba(x)
    if isinstance(probabilities, list):
        return np.column_stack([probs[:, 1] for probs in probabilities])
    return np.asarray(probabilities)


def ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, np.ndarray]:
    lrap = float(label_ranking_average_precision_score(y_true, y_score))
    per_label = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() > 0:
            per_label[j] = average_precision_score(y_true[:, j], y_score[:, j])
    return lrap, float(np.nanmean(per_label)), per_label


def selection_score(lrap: float, macro_ap: float) -> float:
    return 0.5 * (lrap + macro_ap)


def fit_candidates(x_train, y_train, x_valid, y_valid) -> tuple[dict, list[dict], dict]:
    searches = {
        "logistic_regression": logistic_grid(),
        "gradient_boosting": boosting_grid(),
        "mlp": mlp_grid(),
    }
    best_models = {}
    best_params = {}
    rows = []
    for name, candidates in searches.items():
        print(f"\nTuning {MODEL_LABELS[name]} ({len(candidates)} settings)", flush=True)
        best_key = -np.inf
        for spec in candidates:
            started = time.perf_counter()
            model = spec["build"]()
            model.fit(x_train, y_train)
            valid_scores = scores_for(model, x_valid)
            lrap, macro_ap, _ = ranking_metrics(y_valid, valid_scores)
            key = selection_score(lrap, macro_ap)
            elapsed = time.perf_counter() - started
            row = {
                "model": name,
                "params": json.dumps(spec["params"]),
                "valid_lrap": lrap,
                "valid_map": macro_ap,
                "valid_score": key,
                "seconds": elapsed,
            }
            rows.append(row)
            print(
                f"  score={key:.4f}  LRAP={lrap:.4f}  mAP={macro_ap:.4f}  "
                f"{elapsed:.1f}s  {spec['params']}",
                flush=True,
            )
            if key > best_key:
                best_key = key
                best_models[name] = model
                best_params[name] = spec["params"]
    return best_models, rows, best_params


def logistic_grid() -> list[dict]:
    grid = []
    for c in (1e-4, 1e-3, 1e-2, 1e-1):
        for class_weight in (None, "balanced"):
            params = {"C": c, "class_weight": class_weight}

            def build(c=c, class_weight=class_weight):
                estimator = LogisticRegression(
                    C=c,
                    class_weight=class_weight,
                    solver="lbfgs",
                    max_iter=400,
                    random_state=SEED,
                )
                return Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", OneVsRestClassifier(estimator, n_jobs=-1)),
                    ]
                )

            grid.append({"params": params, "build": build})
    return grid


def boosting_grid() -> list[dict]:
    grid = []
    for learning_rate in (0.05, 0.1):
        for max_depth in (3, None):
            params = {
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "min_samples_leaf": 20,
                "l2_regularization": 0.1,
                "class_weight": "balanced",
            }

            def build(learning_rate=learning_rate, max_depth=max_depth):
                estimator = HistGradientBoostingClassifier(
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    max_iter=200,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    class_weight="balanced",
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=12,
                    random_state=SEED,
                )
                return OneVsRestClassifier(estimator, n_jobs=-1)

            grid.append({"params": params, "build": build})
    return grid


def mlp_grid() -> list[dict]:
    grid = []
    for hidden in ((256,), (512, 128)):
        for alpha in (1e-4, 1e-3):
            params = {"hidden_layer_sizes": list(hidden), "alpha": alpha}

            def build(hidden=hidden, alpha=alpha):
                estimator = MLPClassifier(
                    hidden_layer_sizes=hidden,
                    alpha=alpha,
                    activation="relu",
                    solver="adam",
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=60,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=8,
                    random_state=SEED,
                )
                return Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", estimator),
                    ]
                )

            grid.append({"params": params, "build": build})
    return grid


def save_figures(summary: pd.DataFrame, per_label: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    plot_metric_bars(summary)
    plot_per_label(per_label)
    plot_ap_against_prevalence(per_label)


def plot_metric_bars(summary: pd.DataFrame) -> None:
    test = summary[summary["split"] == "test"].set_index("model").loc[MODEL_ORDER]
    valid = summary[summary["split"] == "valid"].set_index("model").loc[MODEL_ORDER]
    labels = [MODEL_LABELS[name] for name in MODEL_ORDER]
    colors = [MODEL_COLORS[name] for name in MODEL_ORDER]
    x = np.arange(len(MODEL_ORDER))
    width = 0.36

    legend_handles = [
        Patch(facecolor="#4C4C4C", alpha=0.4, edgecolor="#4C4C4C", label="Validation"),
        Patch(facecolor="#4C4C4C", edgecolor="black", linewidth=0.4, label="Test"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    for ax, metric, title in (
        (axes[0], "lrap", "LRAP"),
        (axes[1], "map", "Macro mAP"),
    ):
        valid_vals = valid[metric].to_numpy()
        test_vals = test[metric].to_numpy()
        ax.bar(x - width / 2, valid_vals, width, color=colors, alpha=0.4, edgecolor=colors)
        bars = ax.bar(x + width / 2, test_vals, width, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_title(title)
        ax.legend(handles=legend_handles, frameon=False, loc="upper right")
        for bar, value in zip(bars, test_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("T1 fold 0 — CheMeleon embeddings", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "metric_comparison.png", dpi=160)
    plt.close(fig)


def plot_per_label(per_label: pd.DataFrame) -> None:
    trained = ["logistic_regression", "gradient_boosting", "mlp"]
    ordered = per_label.sort_values("train_prevalence", ascending=True)
    values = ordered[[f"test_ap_{name}" for name in trained]].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 10.5))
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(ordered)), ordered["label"], fontsize=8)
    ax.set_xticks(np.arange(len(trained)), [MODEL_LABELS[name] for name in trained], rotation=20, ha="right")
    ax.set_title("Test average precision by odor label\n(sorted by training prevalence)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Average precision")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_label_ap.png", dpi=160)
    plt.close(fig)


def plot_ap_against_prevalence(per_label: pd.DataFrame) -> None:
    trained = ["logistic_regression", "gradient_boosting", "mlp"]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.scatter(
        per_label["train_prevalence"],
        per_label["test_ap_label_frequency"],
        s=22,
        alpha=0.7,
        label=MODEL_LABELS["label_frequency"],
        color=MODEL_COLORS["label_frequency"],
        marker="x",
    )
    for name in trained:
        ax.scatter(
            per_label["train_prevalence"],
            per_label[f"test_ap_{name}"],
            s=28,
            alpha=0.85,
            label=MODEL_LABELS[name],
            color=MODEL_COLORS[name],
        )
    ax.set_xlim(0, max(0.05, per_label["train_prevalence"].max() * 1.08))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Training prevalence")
    ax.set_ylabel("Test average precision")
    ax.set_title("Test average precision by label frequency")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ap_vs_prevalence.png", dpi=160)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {}
    label_cols = None
    frames = []
    for name in ("train", "valid", "test"):
        frame, labels, cols = load_split(name)
        label_cols = cols
        splits[name] = {"frame": frame, "y": labels}
        frames.append(frame)
        print(f"{name}: {len(frame)} molecules, {labels.shape[1]} labels, {int(labels.sum())} positives")

    missing = [s for frame in frames for s in frame["SMILES"] if not isinstance(s, str) or not s]
    if missing:
        raise SystemExit(f"{len(missing)} rows have an empty SMILES string")

    all_smiles = []
    seen = set()
    for frame in frames:
        for smiles in frame["SMILES"]:
            if smiles not in seen:
                seen.add(smiles)
                all_smiles.append(smiles)
    invalid = [smiles for smiles in all_smiles if Chem.MolFromSmiles(smiles) is None]
    if invalid:
        raise SystemExit(f"{len(invalid)} SMILES could not be parsed. First: {invalid[:5]}")
    matrix = embed_unique(all_smiles)
    index = {smiles: i for i, smiles in enumerate(all_smiles)}
    features = {
        name: matrix[[index[s] for s in splits[name]["frame"]["SMILES"]]]
        for name in splits
    }

    prevalence = splits["train"]["y"].mean(axis=0).astype(np.float64)
    dead = [label_cols[j] for j, rate in enumerate(prevalence) if rate == 0 or rate == 1]
    if dead:
        raise SystemExit(f"Train labels with no variation: {dead}")

    best_models, search_rows, best_params = fit_candidates(
        features["train"],
        splits["train"]["y"],
        features["valid"],
        splits["valid"]["y"],
    )
    best_models["label_frequency"] = prevalence
    best_params["label_frequency"] = {"rule": "training positive rate of each label"}

    metric_rows = []
    per_label = pd.DataFrame(
        {
            "label": label_cols,
            "train_prevalence": prevalence,
            "train_positives": splits["train"]["y"].sum(axis=0),
            "valid_positives": splits["valid"]["y"].sum(axis=0),
            "test_positives": splits["test"]["y"].sum(axis=0),
        }
    )
    score_store = {}
    for name in MODEL_ORDER:
        model = best_models[name]
        for split in ("valid", "test"):
            y_score = scores_for(model, features[split])
            lrap, macro_ap, aps = ranking_metrics(splits[split]["y"], y_score)
            metric_rows.append(
                {
                    "model": name,
                    "split": split,
                    "lrap": lrap,
                    "map": macro_ap,
                    "score": selection_score(lrap, macro_ap),
                }
            )
            per_label[f"{split}_ap_{name}"] = aps
            if split == "test":
                score_store[name] = y_score
            print(f"{MODEL_LABELS[name]:<24} {split:<5}  LRAP={lrap:.4f}  mAP={macro_ap:.4f}")

    summary = pd.DataFrame(metric_rows)
    summary.to_csv(OUT_DIR / "metrics.csv", index=False)
    pd.DataFrame(search_rows).to_csv(OUT_DIR / "hyperparameter_search.csv", index=False)
    per_label.to_csv(OUT_DIR / "per_label_ap.csv", index=False)
    (OUT_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2))
    np.savez_compressed(
        OUT_DIR / "test_scores.npz",
        labels=np.array(label_cols),
        y_true=splits["test"]["y"],
        **{f"score_{name}": score_store[name] for name in MODEL_ORDER},
    )
    save_figures(summary, per_label)
    print(f"\nWrote metrics and figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
