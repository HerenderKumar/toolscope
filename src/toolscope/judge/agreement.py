"""Judge vs human agreement (Phase 5 Step 7): raw agreement (as LiveMCPBench reports) and Cohen's kappa (our trust
rule: the judge is used only if kappa >= 0.7, D-072)."""

import random
from collections import Counter

KAPPA_MIN = 0.7
BOOTSTRAP = 2000


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two raters over the same items; 1.0 when both agree perfectly on a single class."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equally long, non-empty label lists")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def kappa_interval(pairs: list[tuple[str, str]], n: int = BOOTSTRAP, seed: int = 2027) -> tuple[float, float]:
    """95% percentile bootstrap interval for Cohen's kappa (resampling the labelled items, seeded)."""
    rng = random.Random(seed)
    values = []
    for _ in range(n):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        values.append(cohen_kappa([h for h, _ in sample], [j for _, j in sample]))
    values.sort()
    return values[int(0.025 * n)], values[int(0.975 * n) - 1]


def agreement_report(pairs: list[tuple[str, str]]) -> dict:
    """pairs = [(human, judge)]. Returns n, raw agreement, kappa, the 2x2 counts and the pass flag."""
    human, judge = [h for h, _ in pairs], [j for _, j in pairs]
    kappa = cohen_kappa(human, judge)
    return {"n": len(pairs), "agreement": sum(h == j for h, j in pairs) / len(pairs), "kappa": kappa,
            "kappa_ci": kappa_interval(pairs),
            "confusion": dict(Counter(f"human={h}/judge={j}" for h, j in pairs)), "passes": kappa >= KAPPA_MIN}
