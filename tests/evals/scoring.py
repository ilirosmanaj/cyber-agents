"""Scoring functions for eval golden datasets."""

from __future__ import annotations


def classification_accuracy(
    predicted: dict[str, tuple[str, bool | None]],
    expected: dict[str, tuple[str, bool | None]],
) -> float:
    """Fraction of endpoint_key -> (category, requires_auth) that match."""
    if not expected:
        return 1.0

    correct = sum(1 for key, exp in expected.items() if predicted.get(key) == exp)
    return correct / len(expected)


def vuln_precision_recall(
    predicted: list[str],
    expected: list[str],
) -> tuple[float, float, float]:
    """Precision, recall, F1 over pattern identifier strings."""
    pred_set = set(predicted)
    exp_set = set(expected)

    if not pred_set and not exp_set:
        return 1.0, 1.0, 1.0

    true_positives = len(pred_set & exp_set)
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(exp_set) if exp_set else 0.0

    if precision + recall == 0:
        return precision, recall, 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def risk_rank_correlation(
    predicted: list[str],
    expected: list[str],
) -> float:
    """Spearman-ish rank correlation, normalized to [0, 1]."""
    if not expected:
        return 1.0

    pred_ranks = {item: rank for rank, item in enumerate(predicted, 1)}
    exp_ranks = {item: rank for rank, item in enumerate(expected, 1)}

    common = set(pred_ranks) & set(exp_ranks)
    if not common:
        return 0.0

    n = len(common)
    d_squared_sum = sum(
        (pred_ranks[item] - exp_ranks[item]) ** 2 for item in common
    )

    if n <= 1:
        return 1.0

    # Spearman's formula: ρ = 1 − 6Σd² / n(n²−1)
    rho = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1))
    return max(0.0, (rho + 1) / 2)
