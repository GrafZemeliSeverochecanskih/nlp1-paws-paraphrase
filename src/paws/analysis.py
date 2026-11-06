"""Results tables, box plots, Bayesian ROPE comparisons (part1.analysis.BenavoliCorrelatedT)."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paws.config import Config
from paws.tutorial import part1


def load_scores(cfg: Config) -> dict[str, dict]:
    """All saved systems, keyed by name."""
    return {path.stem: json.loads(path.read_text()) for path in sorted((cfg.results_dir / "cv").glob("*.json"))}


def results_table(scores: dict[str, dict]) -> pd.DataFrame:
    """Mean and standard deviation over folds per system, best first."""
    rows = [
        {
            "system": name,
            "accuracy_mean": np.mean(s["accuracies"]),
            "accuracy_std": np.std(s["accuracies"], ddof=1),
            "f1_mean": np.mean(s["f1"]),
        }
        for name, s in scores.items()
    ]
    return pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False, ignore_index=True)


def comparison_pairs(scores: dict[str, dict]) -> list[tuple[str, str]]:
    """Log-linear encoders against each other, then the best log-linear system against both MLPs."""
    logistic = [name for name in scores if name.endswith("+loglin")]
    mlps = [name for name in scores if name.endswith(("+mlp", "+mlp2"))]
    pairs = [(a, b) for i, a in enumerate(logistic) for b in logistic[i + 1 :]]
    return pairs + [(mlp.split("+")[0] + "+loglin", mlp) for mlp in mlps] + [tuple(sorted(mlps))]


def rope_table(scores: dict[str, dict], cfg: Config, pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Posterior of mean accuracy difference (B - A) per pair, at the chosen ROPE and the tutorial's 0.005."""
    rows = []
    for a, b in pairs or comparison_pairs(scores):
        posterior = part1.analysis.BenavoliCorrelatedT(
            scores[a]["accuracies"], scores[b]["accuracies"], runs=len(cfg.seeds), names=[a, b]
        )
        row = {"A": a, "B": b, "mean_diff_B_minus_A": posterior.mean_diff, "effect_size": posterior.effect_size}
        for rope in (cfg.rope, 0.005):
            row |= dict(zip((f"P(A)@{rope}", f"P(tie)@{rope}", f"P(B)@{rope}"), posterior.probs(rope)))
        rows.append(row)
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = posterior.plot(cfg.rope)
        ax.set_xlabel("Mean accuracy difference (B - A)")
        ax.get_lines()[0].set_label("posterior")
        ax.legend()
        fig.savefig(cfg.results_dir / f"rope_{a}_vs_{b}.png", dpi=200, bbox_inches="tight")
        plt.close("all")
    return pd.DataFrame(rows)


def box_plot(scores: dict[str, dict], cfg: Config) -> None:
    """Fold accuracies per system, using the tutorial's plot."""
    runs = len(cfg.seeds)
    data = {
        name: [{"accuracies": dict(enumerate(np.array_split(s["accuracies"], runs)[run]))} for run in range(runs)]
        for name, s in scores.items()
    }
    part1.analysis.plot_box_and_points(part1.analysis.make_data_frame(data))
    plt.gcf().savefig(cfg.results_dir / "accuracy_box.png", dpi=200, bbox_inches="tight")
    plt.close("all")
