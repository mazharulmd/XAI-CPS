from __future__ import annotations

import time
import numpy as np


def channel_groups(names: list[str]) -> list[str]:
    groups = []
    for name in names:
        if "." in name:
            groups.append(name.split(".", 1)[0])
        elif "_" in name:
            groups.append(name.split("_", 1)[0])
        else:
            groups.append("OTHER")
    return groups


def component_group_ablation(detector, x: np.ndarray, dx: np.ndarray,
                             feature_names: list[str], previous_x: np.ndarray | None = None):
    """Delete one semantic component at a time and recompute temporal differences."""
    x = np.asarray(x, dtype=float)
    dx = np.asarray(dx, dtype=float)
    if x.ndim != 1 or dx.shape != x.shape or len(feature_names) != len(x):
        raise ValueError("x, dx, and feature_names must describe the same one-dimensional sample")
    previous = x - dx if previous_x is None else np.asarray(previous_x, dtype=float)
    if previous.shape != x.shape or not np.isfinite(previous).all():
        raise ValueError("previous_x must be finite and have the same shape as x")
    groups = channel_groups(feature_names)
    unique_groups = list(dict.fromkeys(groups))
    base_score = float(detector.score(x[None, :], dx[None, :])[0])
    values: dict[str, float] = {}
    for group in unique_groups:
        mask = np.asarray([name == group for name in groups], dtype=bool)
        ablated = x.copy()
        ablated[mask] = detector.x_ref[mask]
        ablated_dx = ablated - previous
        score = float(detector.score(ablated[None, :], ablated_dx[None, :])[0])
        values[group] = base_score - score
    return values


def grouped_ablation(detector, x: np.ndarray, dx: np.ndarray, feature_names: list[str], allowed_groups=None,
                     previous_x: np.ndarray | None = None, state_only: bool = False,
                     preserve_temporal: bool = True):
    """Black-box channel ablation using training medians/modes.

    For temporal detectors, an ablated state is paired with the corresponding
    first difference from the fixed preceding state. This keeps each query on
    the representation identity ``dx[t] = x[t] - x[t-1]``. ``preserve_temporal``
    is false only for reproducing the legacy, representation-inconsistent audit.
    """
    groups = channel_groups(feature_names)
    x = np.asarray(x, dtype=float)
    dx = np.asarray(dx, dtype=float)
    if x.ndim != 1 or dx.shape != x.shape or len(feature_names) != len(x):
        raise ValueError("x, dx, and feature_names must describe the same one-dimensional sample")
    previous = x - dx if previous_x is None else np.asarray(previous_x, dtype=float)
    if previous.shape != x.shape or not np.isfinite(previous).all():
        raise ValueError("previous_x must be finite and have the same shape as x")
    if state_only and not np.allclose(dx, 0.0):
        raise ValueError("state_only ablation requires the detector input dx to be zero")
    base_score = float(detector.score(x[None, :], dx[None, :])[0])
    started = time.perf_counter()
    dimension = len(feature_names)
    indices = np.arange(dimension)
    xe = np.broadcast_to(x, (dimension, dimension)).copy()
    de = np.broadcast_to(dx, (dimension, dimension)).copy()
    xe[indices, indices] = detector.x_ref
    if preserve_temporal:
        de = np.zeros_like(dx)[None, :].repeat(dimension, axis=0) if state_only else xe - previous
    else:
        de[indices, indices] = detector.dx_ref
    values = base_score - np.asarray(detector.score(xe, de), dtype=float)
    if allowed_groups is not None:
        values[np.asarray([group not in allowed_groups for group in groups], dtype=bool)] = 0.0
    return values, time.perf_counter() - started


def greedy_counterfactual(detector, x: np.ndarray, dx: np.ndarray, importance: np.ndarray,
                          threshold: float, feature_names: list[str], allowed_groups=None,
                          previous_x: np.ndarray | None = None, state_only: bool = False):
    """Greedy sparse score recourse with identity-preserving temporal edits.

    This is recourse for the fitted score, not a safe plant action. For temporal
    detectors, every state replacement recomputes its first difference against
    the same observed preceding state. A state-only detector retains zero delta.
    """
    groups = channel_groups(feature_names)
    x = np.asarray(x, dtype=float)
    dx = np.asarray(dx, dtype=float)
    if x.ndim != 1 or dx.shape != x.shape or len(feature_names) != len(x):
        raise ValueError("x, dx, and feature_names must describe the same one-dimensional sample")
    previous = x - dx if previous_x is None else np.asarray(previous_x, dtype=float)
    if previous.shape != x.shape or not np.isfinite(previous).all():
        raise ValueError("previous_x must be finite and have the same shape as x")
    if state_only and not np.allclose(dx, 0.0):
        raise ValueError("state_only counterfactuals require the detector input dx to be zero")
    editable = [j for j in np.argsort(-importance) if importance[j] > 0 and
                (allowed_groups is None or groups[j] in allowed_groups)]
    original = float(detector.score(x[None, :], dx[None, :])[0])
    xe = x.copy(); de = dx.copy(); edits=[]
    for j in editable:
        xe[j] = detector.x_ref[j]
        de = np.zeros_like(dx) if state_only else xe - previous
        edits.append(int(j))
        if float(detector.score(xe[None, :], de[None, :])[0]) <= threshold:
            break
    final = float(detector.score(xe[None, :], de[None, :])[0])
    return {"success": bool(final <= threshold), "original_score": original, "counterfactual_score": final,
            "edit_indices": edits, "edit_names": [feature_names[j] for j in edits],
            "edit_count": len(edits), "normalized_l1": float(np.abs((xe-x)/detector.x_scale).sum()+np.abs((de-dx)/detector.dx_scale).sum()),
            "edited_x": xe, "edited_dx": de}


def capability_support_counterfactual(detector, x: np.ndarray, dx: np.ndarray, importance: np.ndarray,
                                      threshold: float, feature_names: list[str], refs_x: np.ndarray,
                                      refs_dx: np.ndarray, support_threshold: float,
                                      capability_groups: set[str], max_groups: int = 4,
                                      donor_count: int = 8, max_tags: int = 16,
                                      previous_x: np.ndarray | None = None,
                                      state_only: bool = False,
                                      transition_score_fn=None,
                                      transition_threshold: float | None = None) -> dict:
    """Search sparse tag-prefix edits from normal exemplars under group/tag budgets.

    This is empirical-support-constrained recourse, not a validated plant-physics model.
    The caller supplies a capability set fixed without test labels. A candidate changes
    only the selected groups and borrows their level values from one training-normal
    donor. For temporal detectors, first differences are recomputed from candidate
    levels and the observed preceding state, enforcing dx[t] = x[t] - x[t-1]. A state-only
    detector receives its zero-difference representation. The full candidate must remain
    close to the normal reference bank and flip the frozen detector score. The search is deliberately
    approximate: it tests ranked group prefixes and ranked tag prefixes, not every subset.
    """
    groups = channel_groups(feature_names)
    x = np.asarray(x, dtype=float)
    dx = np.asarray(dx, dtype=float)
    if x.ndim != 1 or dx.shape != x.shape or len(feature_names) != len(x):
        raise ValueError("x, dx, and feature_names must describe the same one-dimensional sample")
    if previous_x is None:
        previous = x - dx
    else:
        previous = np.asarray(previous_x, dtype=float)
        if previous.shape != x.shape or not np.isfinite(previous).all():
            raise ValueError("previous_x must be finite and have the same shape as x")
    if state_only and not np.allclose(dx, 0.0):
        raise ValueError("state_only counterfactuals require the detector input dx to be zero")
    if (transition_score_fn is None) != (transition_threshold is None):
        raise ValueError("transition_score_fn and transition_threshold must be supplied together")
    original_transition_error = (None if state_only else float(np.max(np.abs(dx - (x - previous)))))
    original_score = float(detector.score(x[None, :], dx[None, :])[0])
    eligible = sorted(set(groups) & set(capability_groups))
    group_importance = {g: 0.0 for g in eligible}
    for group, value in zip(groups, importance):
        if group in group_importance:
            group_importance[group] += max(0.0, float(value))
    ranked = sorted(eligible, key=lambda g: (-group_importance[g], g))
    budget = max(0, min(int(max_groups), len(ranked)))
    if original_score <= threshold:
        return {"success": True, "minimum_required_groups": 0, "minimum_required_tags": 0, "reason": "already_below_threshold",
                "original_score": original_score, "counterfactual_score": original_score,
                "edited_groups": [], "edit_names": [], "edit_count": 0, "normalized_l1": 0.0,
                "support_distance": None, "edited_x": x.copy(), "edited_dx": dx.copy(),
                "transition_consistency_max_abs_error": original_transition_error,
                "donor_index": None, "candidates_checked": 0,
                "success_by_group_budget":{str(k):True for k in sorted(set([1,2,int(max_groups)]))},
                "success_by_tag_budget":{str(k):True for k in sorted(set([1,2,4,8,int(max_tags)]))}}
    if budget == 0 or not len(refs_x):
        return {"success": False, "minimum_required_groups": None, "minimum_required_tags": None, "reason": "empty_capability_or_reference",
                "original_score": original_score, "counterfactual_score": original_score,
                "edited_groups": [], "edit_names": [], "edit_count": 0, "normalized_l1": 0.0,
                "support_distance": None, "edited_x": x.copy(), "edited_dx": dx.copy(),
                "transition_consistency_max_abs_error": original_transition_error,
                "donor_index": None, "candidates_checked": 0,
                "success_by_group_budget":{str(k):False for k in sorted(set([1,2,int(max_groups)]))},
                "success_by_tag_budget":{str(k):False for k in sorted(set([1,2,4,8,int(max_tags)]))}}

    zx, zdx = detector.standardized(x[None, :], dx[None, :])
    rz, rdz = detector.standardized(refs_x, refs_dx)
    reference = np.concatenate([rz, rdz], axis=1)
    query = np.concatenate([zx[0], zdx[0]])
    donor_distances = np.sqrt(np.sum((reference - query) ** 2, axis=1))
    donors = np.argsort(donor_distances)[:min(int(donor_count), len(refs_x))]
    scale_x = np.maximum(detector.x_scale, 1e-12)
    scale_dx = np.maximum(detector.dx_scale, 1e-12)
    importance = np.asarray(importance, dtype=float)
    checked = 0
    transition_rejections = 0
    feasible_candidates = []

    for count in range(1, budget + 1):
        allowed_prefix = set(ranked[:count])
        eligible_indices = [j for j, group in enumerate(groups) if group in allowed_prefix]
        eligible_indices.sort(key=lambda j: (-max(0.0, float(importance[j])), j))
        tag_budget = min(max(int(max_tags), count), len(eligible_indices))
        for tag_count in range(count, tag_budget + 1):
            edited_indices = eligible_indices[:tag_count]
            edited_groups = sorted({groups[j] for j in edited_indices})
            if len(edited_groups) != count:
                continue
            mask = np.zeros(len(feature_names), dtype=bool)
            mask[edited_indices] = True
            feasible = []
            for donor in donors:
                cx = x.copy(); cdx = dx.copy()
                cx[mask] = refs_x[donor, mask]
                cdx = np.zeros_like(dx) if state_only else cx - previous
                checked += 1
                process_transition_score = None
                if transition_score_fn is not None:
                    process_transition_score = float(transition_score_fn(previous, cx))
                    if not np.isfinite(process_transition_score) or process_transition_score > float(transition_threshold):
                        transition_rejections += 1
                        continue
                score = float(detector.score(cx[None, :], cdx[None, :])[0])
                if score > threshold:
                    continue
                cz, cdz = detector.standardized(cx[None, :], cdx[None, :])
                candidate = np.concatenate([cz[0], cdz[0]])
                support_distance = float(np.sqrt(np.min(np.sum((reference - candidate) ** 2, axis=1))))
                if support_distance > support_threshold:
                    continue
                normalized_l1 = float(np.abs((cx - x) / scale_x).sum() + np.abs((cdx - dx) / scale_dx).sum())
                feasible.append(((normalized_l1, support_distance, score), cx, cdx, score,
                                 support_distance, normalized_l1, int(donor), edited_indices.copy(),
                                 process_transition_score))
            if feasible:
                for item in feasible:
                    _, cx, cdx, score, support_distance, normalized_l1, donor, edited_indices, process_transition_score = item
                    actual_groups=sorted({groups[j] for j in edited_indices})
                    feasible_candidates.append({"x":cx,"dx":cdx,"score":score,"support_distance":support_distance,
                        "normalized_l1":normalized_l1,"donor_index":donor,"indices":edited_indices,
                        "group_count":len(actual_groups),"tag_count":len(edited_indices),"groups":actual_groups,
                        "process_transition_score":process_transition_score})
    if feasible_candidates:
        best=min(feasible_candidates,key=lambda c:(c["tag_count"],c["group_count"],c["normalized_l1"],c["support_distance"],c["score"]))
        group_success={str(k):any(c["group_count"]<=k for c in feasible_candidates) for k in sorted(set([1,2,budget]))}
        tag_success={str(k):any(c["tag_count"]<=k for c in feasible_candidates) for k in sorted(set([1,2,4,8,int(max_tags)]))}
        return {"success": True, "minimum_required_groups":min(c["group_count"] for c in feasible_candidates),
                "minimum_required_tags":min(c["tag_count"] for c in feasible_candidates), "reason": "score_flip_and_support_pass",
                "original_score": original_score, "counterfactual_score": best["score"],
                "edited_groups": best["groups"], "edit_names": [feature_names[j] for j in best["indices"]],
                "edit_count": best["tag_count"], "normalized_l1": best["normalized_l1"],
                "support_distance": best["support_distance"], "edited_x": best["x"], "edited_dx": best["dx"],
                "transition_consistency_max_abs_error": (None if state_only else float(np.max(np.abs(best["dx"] - (best["x"] - previous))))),
                "process_transition_score": best["process_transition_score"],
                "process_transition_threshold": transition_threshold,
                "process_transition_rejections": transition_rejections,
                "donor_index": best["donor_index"], "candidates_checked": checked,
                "success_by_group_budget":group_success,"success_by_tag_budget":tag_success}
    return {"success": False, "minimum_required_groups": None, "minimum_required_tags": None, "reason": "no_supported_score_flip_within_budget",
            "original_score": original_score, "counterfactual_score": original_score,
            "edited_groups": [], "edit_names": [], "edit_count": 0, "normalized_l1": 0.0,
            "support_distance": None, "edited_x": x.copy(), "edited_dx": dx.copy(),
            "transition_consistency_max_abs_error": original_transition_error,
            "process_transition_score": None, "process_transition_threshold": transition_threshold,
            "process_transition_rejections": transition_rejections,
            "donor_index": None, "candidates_checked": checked,
            "success_by_group_budget":{str(k):False for k in sorted(set([1,2,budget]))},
            "success_by_tag_budget":{str(k):False for k in sorted(set([1,2,4,8,int(max_tags)]))}}


def process_localization(importance: np.ndarray, feature_names: list[str], active_processes: set[str]) -> dict:
    groups = channel_groups(feature_names)
    totals: dict[str,float] = {}
    for g,v in zip(groups,importance): totals[g]=totals.get(g,0.0)+max(0.0,float(v))
    ranked=sorted(totals,key=totals.get,reverse=True)
    return {"predicted_process": ranked[0] if ranked else None,
            "hit_at_1": bool(ranked and ranked[0] in active_processes),
            "hit_at_3": bool(set(ranked[:3]) & active_processes),
            "active_processes": sorted(active_processes),"ranked_processes": ranked,
            "process_scores": totals}


def stability_topk(detector, x: np.ndarray, dx: np.ndarray, feature_names: list[str], k: int,
                   perturbations: int = 5, sigma: float = 0.02, seed: int = 42,
                   previous_x: np.ndarray | None = None, state_only: bool = False) -> list[float]:
    """Measure top-k ablation stability under small state noise.

    Temporal first differences are recomputed from the perturbed current state and
    fixed preceding state, so every perturbed sample obeys dx[t] = x[t] - x[t-1].
    A state-only detector continues to receive its zero-difference representation.
    """
    rng=np.random.default_rng(seed)
    x=np.asarray(x,dtype=float)
    dx=np.asarray(dx,dtype=float)
    previous=x-dx if previous_x is None else np.asarray(previous_x,dtype=float)
    if x.ndim!=1 or dx.shape!=x.shape or previous.shape!=x.shape:
        raise ValueError("x, dx, and previous_x must be matching one-dimensional samples")
    if state_only and not np.allclose(dx,0.0):
        raise ValueError("state_only stability requires the detector input dx to be zero")
    base,_=grouped_ablation(detector,x,dx,feature_names,previous_x=previous,state_only=state_only)
    top=set(np.argsort(-base)[:k]); out=[]
    for _ in range(perturbations):
        xp=x+rng.normal(0,sigma,size=x.shape)*detector.x_scale
        dp=np.zeros_like(dx) if state_only else xp-previous
        vals,_=grouped_ablation(detector,xp,dp,feature_names,previous_x=previous,state_only=state_only)
        other=set(np.argsort(-vals)[:k])
        out.append(len(top&other)/max(1,len(top|other)))
    return out
