"""T2V retrieval primitives for T3.

Pure-Python (torch + numpy) helpers extracted from the cleaned eval script so
they can be imported by external consumers without going through the full
`evaluate.run()` flow.

Heavy deps (torch) are imported lazily inside the functions that need them.
"""

from __future__ import annotations

import json
import os
import pathlib
from collections import defaultdict
from typing import Any


HN_BIN_ORDER = ["0-100", "100-200", "200-300", "300-400", "400-500"]
RECALL_KS = [1, 5, 10, 50, 100, 200, 300, 400, 500]


def compute_ranks(text_emb, video_emb, txt2neg: dict):
    """T2V rank for each query within {positive} ∪ txt2neg[i] candidates.

    Args:
        text_emb: [N_txt, D] tensor, L2-normalized.
        video_emb: [N_vid, D] tensor, L2-normalized.
        txt2neg: dict mapping text_idx -> list of negative video indices.

    Returns:
        np.ndarray of shape [N_txt], 0-indexed rank of the positive per query.
    """
    import numpy as np  # lazy

    sim = (text_emb @ video_emb.T).numpy()
    n_txt = text_emb.shape[0]
    ranks = np.zeros(n_txt, dtype=np.int64)
    for i in range(n_txt):
        candidate_indices = [i] + list(set(txt2neg[i]))
        candidate_scores = sim[i, candidate_indices]
        sorted_order = np.argsort(-candidate_scores)
        ranks[i] = int(np.where(sorted_order == 0)[0][0])
    return ranks


def compute_metrics(ranks) -> dict[str, Any]:
    """R@K (for K in RECALL_KS) and median rank, given a 1-D rank array."""
    import numpy as np  # lazy

    if len(ranks) == 0:
        return {f"R@{k}": 0.0 for k in RECALL_KS} | {"MedR": 0, "n": 0}
    out: dict[str, Any] = {"n": int(len(ranks))}
    for k in RECALL_KS:
        out[f"R@{k}"] = round(100.0 * float(np.sum(ranks < k)) / len(ranks), 4)
    out["MedR"] = int(np.median(ranks))
    return out


def load_mapping(mappings_dir: pathlib.Path, sport: str, split: str) -> list[dict]:
    """Per-text-idx (tier, category, composition) list pre-computed from source CSVs."""
    path = pathlib.Path(mappings_dir) / f"{sport}_{split}.json"
    with open(path) as f:
        return json.load(f)


def load_hn_bins(data_dir: pathlib.Path, sport: str, split: str, n_expected: int) -> list:
    """Per-positive hn_bin string (None for tier 1) read from the data JSON."""
    path = pathlib.Path(data_dir) / split / f"{sport}_{split}.json"
    with open(path) as f:
        data = json.load(f)
    positives = [d for d in data if d.get("is_positive")]
    if len(positives) != n_expected:
        raise ValueError(f"{path}: expected {n_expected} positives, got {len(positives)}")
    return [d.get("hn_bin") for d in positives]


def aggregate_results(ranks, mapping: list[dict], hn_bins: list | None = None) -> dict:
    """Roll up per-tier / per-category / per-composition / per-hn-bin metrics."""
    import numpy as np  # lazy

    ranks = np.array(ranks)
    n = len(ranks)
    results: dict[str, Any] = {"overall": compute_metrics(ranks)}

    # Per tier.
    tier_indices = defaultdict(list)
    for i, m in enumerate(mapping):
        tier_indices[m["tier"]].append(i)
    results["per_tier"] = {
        t: compute_metrics(ranks[idx]) for t, idx in sorted(tier_indices.items())
    }

    # Per category within tier.
    cat_indices = defaultdict(lambda: defaultdict(list))
    for i, m in enumerate(mapping):
        cat_indices[m["tier"]][m["category"]].append(i)
    results["per_category"] = {
        t: {c: compute_metrics(ranks[idx]) for c, idx in sorted(cats.items())}
        for t, cats in sorted(cat_indices.items())
    }

    # Per composition within tier > category.
    comp_indices = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, m in enumerate(mapping):
        comp_key = " + ".join(m["composition"])
        comp_indices[m["tier"]][m["category"]][comp_key].append(i)
    results["per_composition"] = {
        t: {
            c: {k: compute_metrics(ranks[idx]) for k, idx in sorted(comps.items())}
            for c, comps in sorted(cats.items())
        }
        for t, cats in sorted(comp_indices.items())
    }

    # Per hard-negative bucket (tier 2/3 only).
    if hn_bins is not None:
        if len(hn_bins) != n:
            raise ValueError(f"hn_bins length {len(hn_bins)} != ranks length {n}")
        hn_bin_indices = defaultdict(list)
        for i, hb in enumerate(hn_bins):
            if hb is not None:
                hn_bin_indices[hb].append(i)
        results["per_hn_bin"] = {
            hb: compute_metrics(ranks[hn_bin_indices[hb]])
            for hb in HN_BIN_ORDER if hb in hn_bin_indices
        }
        hn_tier_indices = defaultdict(lambda: defaultdict(list))
        for i, hb in enumerate(hn_bins):
            if hb is not None:
                hn_tier_indices[mapping[i]["tier"]][hb].append(i)
        results["per_hn_bin_per_tier"] = {
            t: {
                hb: compute_metrics(ranks[idx_dict[hb]])
                for hb in HN_BIN_ORDER if hb in (idx_dict := hn_tier_indices[t])
            }
            for t in sorted(hn_tier_indices.keys())
        }

    return results


def evaluate_one(
    sport: str,
    split: str,
    caption: str,
    *,
    data_dir: pathlib.Path,
    embed_dir: pathlib.Path,
    mappings_dir: pathlib.Path,
    output_dir: pathlib.Path | None = None,
) -> dict[str, Any] | None:
    """Run one (sport, split, caption) eval and optionally save results.

    Returns None if the embedding file is missing.
    """
    import torch  # lazy

    embed_path = pathlib.Path(embed_dir) / f"embeds_{split}_{sport}_{caption}.pt"
    if not embed_path.exists():
        return None

    data = torch.load(embed_path, map_location="cpu", weights_only=False)
    text_emb = data["text_emb"].float()
    video_emb = data["video_emb"].float()
    txt2neg = data["txt2neg"]
    if video_emb.dim() == 3:
        video_emb = video_emb.squeeze(1)
    text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)
    video_emb = video_emb / video_emb.norm(dim=1, keepdim=True)

    mapping = load_mapping(mappings_dir, sport, split)
    if len(mapping) != text_emb.shape[0]:
        raise ValueError(
            f"Mapping length {len(mapping)} != text_emb rows {text_emb.shape[0]}"
        )
    hn_bins = load_hn_bins(data_dir, sport, split, n_expected=text_emb.shape[0])

    ranks = compute_ranks(text_emb, video_emb, txt2neg)
    results = aggregate_results(ranks, mapping, hn_bins=hn_bins)
    results["config"] = {
        "sport": sport,
        "split": split,
        "caption_type": f"{caption}_caption",
        "embed_file": embed_path.name,
        "num_texts": int(text_emb.shape[0]),
        "num_videos": int(video_emb.shape[0]),
    }

    if output_dir is not None:
        out_dir = pathlib.Path(output_dir) / sport
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{caption}_caption_{split}.json", "w") as f:
            json.dump(results, f, indent=2)

    return results
