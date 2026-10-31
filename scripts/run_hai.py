from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import json
import platform
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from xai_cps.data import load_hai, contiguous_events
from xai_cps.detectors import RobustZDetector, PCAResidualDetector
from xai_cps.dynamics import NormalTransitionModel
from xai_cps.explainers import (capability_support_counterfactual, grouped_ablation,
                                channel_groups, greedy_counterfactual,
                                process_localization, stability_topk)
from xai_cps.metrics import binary_metrics, event_metrics, bootstrap_ci, paired_bootstrap_difference


def peak_working_set_mb():
    class PMC(ctypes.Structure):
        _fields_=[("cb",ctypes.c_ulong),("PageFaultCount",ctypes.c_ulong),("PeakWorkingSetSize",ctypes.c_size_t),
                  ("WorkingSetSize",ctypes.c_size_t),("QuotaPeakPagedPoolUsage",ctypes.c_size_t),
                  ("QuotaPagedPoolUsage",ctypes.c_size_t),("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),
                  ("QuotaNonPagedPoolUsage",ctypes.c_size_t),("PagefileUsage",ctypes.c_size_t),
                  ("PeakPagefileUsage",ctypes.c_size_t)]
    try:
        counters=PMC(); counters.cb=ctypes.sizeof(PMC)
        psapi=ctypes.WinDLL("psapi")
        handle=ctypes.WinDLL("kernel32").GetCurrentProcess()
        ok=psapi.GetProcessMemoryInfo(handle,ctypes.byref(counters),counters.cb)
        return float(counters.PeakWorkingSetSize/(1024**2)) if ok else None
    except Exception:
        return None


def safe_float(x):
    if x is None: return None
    x=float(x)
    return x if np.isfinite(x) else None


def nearest_distances(detector, x, dx, refs_x, refs_dx):
    zx,zd=detector.standardized(x[None,:],dx[None,:])
    rz,rd=detector.standardized(refs_x,refs_dx)
    query=np.concatenate([zx[0],zd[0]])
    refs=np.concatenate([rz,rd],axis=1)
    return float(np.sqrt(np.min(np.sum((refs-query)**2,axis=1))))


def detector_reference_delta(dx, state_only):
    """Return reference deltas in the same representation used by a detector."""
    return np.zeros_like(dx) if state_only else dx


def model_specs(config, dataset):
    zeros_train=np.zeros_like(dataset.train_dx)
    zeros_val=np.zeros_like(dataset.validation.dx)
    return [
        ("RobustZ-TEMPORAL",RobustZDetector(),dataset.train_dx,dataset.validation.dx,False),
        ("PCA-STATE",PCAResidualDetector(config["variance_explained"],config["max_components"]),zeros_train,zeros_val,True),
        ("PCA-TEMPORAL",PCAResidualDetector(config["variance_explained"],config["max_components"]),dataset.train_dx,dataset.validation.dx,False),
    ]


def explain_false_alarm_episodes(model_name, detector, state_only, segment, segment_dx, pred,
                                 feature_names, threshold, seed, sample_limit, refs_x, refs_dx,
                                 support_threshold, capability_groups, max_groups, donor_count, max_tags,
                                 transition_model, transition_threshold):
    """Explain a deterministic sample of pure false-alarm episodes at first alarm.

    Attack labels are used only to identify pure false alarms for this post-hoc evaluation
    sample. They are not passed to the detector, attribution, counterfactual, or donor search.
    """
    predicted_episodes = contiguous_events(pred)
    false_episodes = [(i, start, end) for i, (start, end) in enumerate(predicted_episodes)
                      if not np.asarray(segment.attack[start:end], dtype=bool).any()]
    sample_n = min(max(0, int(sample_limit)), len(false_episodes))
    if sample_n == len(false_episodes):
        selected = np.arange(len(false_episodes), dtype=int)
    elif sample_n:
        salt = zlib.crc32(f"{segment.name}|{model_name}".encode("utf-8"))
        rng = np.random.default_rng(int(seed) + int(salt))
        selected = np.sort(rng.choice(len(false_episodes), size=sample_n, replace=False))
    else:
        selected = np.array([], dtype=int)

    rows = []
    for sample_index in selected:
        episode_index, start, end = false_episodes[int(sample_index)]
        alarm_row = int(start)
        x = segment.x[alarm_row]
        dx = segment_dx[alarm_row]
        previous_x = segment.x[alarm_row-1] if alarm_row > 0 else x
        started = time.perf_counter()
        native = np.asarray(detector.contributions(x[None, :], dx[None, :])[0], dtype=float)
        ablation, ablation_seconds = grouped_ablation(
            detector, x, dx, feature_names, previous_x=previous_x, state_only=state_only)
        greedy = greedy_counterfactual(
            detector, x, dx, ablation, threshold, feature_names,
            previous_x=previous_x, state_only=state_only)
        capability = capability_support_counterfactual(
            detector, x, dx, ablation, threshold, feature_names, refs_x, refs_dx, support_threshold,
            capability_groups=capability_groups, max_groups=max_groups, donor_count=donor_count,
            max_tags=max_tags, previous_x=previous_x, state_only=state_only)
        dynamics_started = time.perf_counter()
        dynamics = capability_support_counterfactual(
            detector, x, dx, ablation, threshold, feature_names, refs_x, refs_dx, support_threshold,
            capability_groups=capability_groups, max_groups=max_groups, donor_count=donor_count,
            max_tags=max_tags, previous_x=previous_x, state_only=state_only,
            transition_score_fn=transition_model.score, transition_threshold=transition_threshold)
        dynamics_seconds = float(time.perf_counter() - dynamics_started)
        explanation_seconds = float(time.perf_counter() - started)
        original_distance = nearest_distances(detector, x, dx, refs_x, refs_dx)
        rows.append({
            "model": model_name,
            "file": segment.name,
            "predicted_episode_index": int(episode_index),
            "false_alarm_sample_index": int(sample_index),
            "start_row": int(start),
            "end_row_exclusive": int(end),
            "explanation_row": alarm_row,
            "score": float(detector.score(x[None, :], dx[None, :])[0]),
            "threshold": float(threshold),
            "native_top10_tags": "|".join(feature_names[j] for j in np.argsort(-native)[:10]),
            "ablation_top10_tags": "|".join(feature_names[j] for j in np.argsort(-ablation)[:10]),
            "greedy_score_flip": bool(greedy["success"]),
            "greedy_edit_count": int(greedy["edit_count"]),
            "capability_support_success": bool(capability["success"]),
            "capability_support_reason": capability["reason"],
            "capability_edit_count": int(capability["edit_count"]),
            "capability_edit_names": ";".join(capability["edit_names"]),
            "capability_edited_groups": ";".join(capability["edited_groups"]),
            "capability_score": float(capability["counterfactual_score"]),
            "capability_support_distance": safe_float(capability["support_distance"]),
            "dynamics_constrained_success": bool(dynamics["success"]),
            "dynamics_constrained_edit_count": int(dynamics["edit_count"]),
            "dynamics_constrained_edit_names": ";".join(dynamics["edit_names"]),
            "dynamics_transition_score": safe_float(dynamics.get("process_transition_score")),
            "dynamics_transition_threshold": float(transition_threshold),
            "dynamics_transition_rejections": int(dynamics.get("process_transition_rejections", 0)),
            "dynamics_explanation_seconds": dynamics_seconds,
            "normal_reference_distance_before": original_distance,
            "normal_reference_q95_validation": float(support_threshold),
            "transition_consistency_max_abs_error": capability.get("transition_consistency_max_abs_error"),
            "explanation_seconds": explanation_seconds,
            "ablation_seconds": float(ablation_seconds),
        })
    return {"total_pure_false_alarm_episodes": len(false_episodes), "rows": rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=None)
    args=parser.parse_args()
    config_path=ROOT/"configs"/"hai_21_03.json"
    config=json.loads(config_path.read_text(encoding="utf-8-sig"))
    seed=config["random_seed"] if args.seed is None else args.seed
    data_load_start=time.perf_counter()
    dataset=load_hai(config_path,max_train_rows=config["training_rows_max"],seed=seed)
    data_load_seconds=time.perf_counter()-data_load_start
    unknown_capability_groups=set(config["capability_groups"])-set(dataset.process_groups)
    if unknown_capability_groups:
        raise ValueError(f"Capability groups not present in this dataset schema: {sorted(unknown_capability_groups)}")
    run_stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir=ROOT/"results"/"runs"/f"hai21_seed{seed}_{run_stamp}"
    run_dir.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    ref_n=min(int(config.get("normal_reference_count",256)),len(dataset.train_x))
    ref_ix=np.sort(rng.choice(len(dataset.train_x),size=ref_n,replace=False))
    refs_x=dataset.train_x[ref_ix]; refs_dx=dataset.train_dx[ref_ix]
    transition_model=NormalTransitionModel(
        ridge=float(config.get("transition_ridge",1.0)),
        top_k=int(config.get("transition_top_k_residuals",5))).fit(dataset.train_x,dataset.train_dx)
    val_previous=dataset.validation.x-dataset.validation.dx
    transition_threshold=transition_model.calibrate(
        val_previous,dataset.validation.x,
        false_positive_rate=float(config.get("transition_validation_false_positive_rate",0.05)))
    val_n=min(3000,len(dataset.validation.x))
    val_ix=np.linspace(0,len(dataset.validation.x)-1,val_n,dtype=int)

    prediction_path=run_dir/"predictions.csv.gz"
    prediction_stream=gzip.open(prediction_path,"wt",encoding="utf-8",newline="")
    prediction_header=True; summaries={}; all_false_alarm_explanation_rows=[]
    scores_for_comparison={}
    for model_name,detector,train_dx,val_dx,state_only in model_specs(config,dataset):
        fit_start=time.perf_counter(); detector.fit(dataset.train_x,train_dx); fit_s=time.perf_counter()-fit_start
        model_refs_dx=detector_reference_delta(refs_dx,state_only)
        val_x=dataset.validation.x
        val_score=detector.score(val_x,val_dx)
        q=1-float(config["validation_false_positive_rate"])
        threshold=float(np.quantile(val_score,q))
        val_fpr=float(np.mean(val_score>threshold))
        # Empirical nearest-normal distance tolerance is calibrated on a held-out normal subset.
        dist_val_ix=np.linspace(0,len(dataset.validation.x)-1,min(512,len(dataset.validation.x)),dtype=int)
        zv,zdv=detector.standardized(dataset.validation.x[dist_val_ix],val_dx[dist_val_ix])
        zref,dref=detector.standardized(refs_x,model_refs_dx)
        ref_join=np.concatenate([zref,dref],axis=1)
        dist_val=[]
        for qx,qd in zip(zv,zdv):
            query=np.concatenate([qx,qd])
            dist_val.append(float(np.sqrt(np.min(np.sum((ref_join-query)**2,axis=1)))))
        plausibility_q95=float(np.quantile(dist_val,.95))
        file_summaries=[]; point_y=[]; point_scores=[]; model_event_rows=[]
        explain_times=[]; stability=[]; cf_all=[]; cf_cap=[]; cf_cap_groups=[]; cf_cap_cost=[]; cf_cap_support=[]
        cf_cap_tag_counts=[]; cf_cap_edit_counts=[]; cf_cap_group_counts=[]; cf_cap_hit=[]
        cf_dyn=[]; cf_dyn_edit_counts=[]; cf_dyn_hit=[]; cf_dyn_times=[]; cf_dyn_transition_scores=[]
        hit_ab=[]; hit_native=[]; event_hit=[]; event_detect=[]; delays=[]
        cf_group_budget={str(k):[] for k in sorted(set([1,2,int(config["capability_max_groups"])]))}
        cf_tag_budget={str(k):[] for k in sorted(set([1,2,4,8,int(config["capability_max_tags"])]))}
        normal_dist_before=[]; normal_dist_after=[]; plausibility_pass=[]; point_latency=[]
        false_alarm_explanation_rows=[]; pure_false_alarm_count=0
        model_rows=[]
        for segment in dataset.tests:
            segment_dx=np.zeros_like(segment.dx) if state_only else segment.dx
            t0=time.perf_counter(); score=detector.score(segment.x,segment_dx); score_s=time.perf_counter()-t0
            pred=score>threshold
            m=binary_metrics(segment.attack,score,threshold)
            ev=event_metrics(segment.attack,pred,segment.timestamps)
            file_summaries.append({"file":segment.name,"rows":len(segment.attack),"score_seconds":score_s,
                                   "point_metrics":m,"event_metrics":ev})
            point_y.append(segment.attack); point_scores.append(score)
            for i,(start,end) in enumerate(contiguous_events(segment.attack)):
                peak=start+int(np.argmax(score[start:end]))
                active={f"P{j+1}" for j,v in enumerate(segment.process_labels[start:end].max(axis=0)) if v}
                pred_detected=bool(pred[start:end].any())
                event_detect.append(float(pred_detected))
                if not pred_detected:
                    # Explanations are evaluated at actual alarm points. Missed events have no alarm to explain.
                    continue
                first=int(np.flatnonzero(pred[start:end])[0]); alarm_row=start+first
                delays.append(float(first))
                event_hit.append(float(bool(active)))
                x=segment.x[alarm_row]; dx=segment_dx[alarm_row]
                previous_x=segment.x[alarm_row-1] if alarm_row>0 else x
                t1=time.perf_counter(); native=np.asarray(detector.contributions(x[None,:],dx[None,:])[0],dtype=float)
                abl,abl_s=grouped_ablation(
                    detector,x,dx,dataset.feature_names,previous_x=previous_x,state_only=state_only)
                rec_all=greedy_counterfactual(
                    detector,x,dx,abl,threshold,dataset.feature_names,
                    previous_x=previous_x,state_only=state_only)
                capability=set(config["capability_groups"])
                rec_cap=capability_support_counterfactual(
                    detector,x,dx,abl,threshold,dataset.feature_names,refs_x,model_refs_dx,plausibility_q95,
                    capability_groups=capability,max_groups=int(config["capability_max_groups"]),
                    donor_count=int(config["capability_donor_count"]),max_tags=int(config["capability_max_tags"]),
                    previous_x=previous_x,state_only=state_only)
                dyn_start=time.perf_counter()
                rec_dyn=capability_support_counterfactual(
                    detector,x,dx,abl,threshold,dataset.feature_names,refs_x,model_refs_dx,plausibility_q95,
                    capability_groups=capability,max_groups=int(config["capability_max_groups"]),
                    donor_count=int(config["capability_donor_count"]),max_tags=int(config["capability_max_tags"]),
                    previous_x=previous_x,state_only=state_only,
                    transition_score_fn=transition_model.score,transition_threshold=transition_threshold)
                dyn_s=time.perf_counter()-dyn_start; cf_dyn_times.append(dyn_s)
                expl_s=time.perf_counter()-t1; explain_times.append(expl_s)
                loc_native=process_localization(native,dataset.feature_names,active)
                loc_ab=process_localization(abl,dataset.feature_names,active)
                loc_cf=process_localization(np.isin(np.arange(len(abl)),rec_all["edit_indices"]).astype(float),dataset.feature_names,active)
                cap_indices=[j for j,name in enumerate(dataset.feature_names) if channel_groups([name])[0] in rec_cap["edited_groups"]]
                loc_cap=process_localization(np.isin(np.arange(len(abl)),cap_indices).astype(float),dataset.feature_names,active)
                dyn_groups=set(rec_dyn["edited_groups"])
                dyn_indices=[j for j,name in enumerate(dataset.feature_names) if channel_groups([name])[0] in dyn_groups]
                loc_dyn=process_localization(np.isin(np.arange(len(abl)),dyn_indices).astype(float),dataset.feature_names,active)
                if active:
                    hit_native.append(float(loc_native["hit_at_1"]))
                    hit_ab.append(float(loc_ab["hit_at_1"]))
                    if rec_cap["success"]: cf_cap_hit.append(float(loc_cap["hit_at_1"]))
                    if rec_dyn["success"]: cf_dyn_hit.append(float(loc_dyn["hit_at_1"]))
                cf_all.append(float(rec_all["success"]))
                cf_cap.append(float(rec_cap["success"]))
                cf_dyn.append(float(rec_dyn["success"]))
                cf_dyn_edit_counts.append(rec_dyn["edit_count"] if rec_dyn["success"] else None)
                if rec_dyn["success"]:
                    cf_dyn_transition_scores.append(float(rec_dyn["process_transition_score"]))
                cf_cap_groups.append(rec_cap["minimum_required_groups"])
                cf_cap_tag_counts.append(rec_cap["minimum_required_tags"])
                for key,value in rec_cap["success_by_group_budget"].items(): cf_group_budget[key].append(float(value))
                for key,value in rec_cap["success_by_tag_budget"].items(): cf_tag_budget[key].append(float(value))
                if rec_cap["success"]:
                    cf_cap_cost.append(float(rec_cap["normalized_l1"]))
                    cf_cap_support.append(float(rec_cap["support_distance"]))
                    cf_cap_edit_counts.append(int(rec_cap["edit_count"]))
                    cf_cap_group_counts.append(len(rec_cap["edited_groups"]))
                stab=stability_topk(detector,x,dx,dataset.feature_names,k=5,
                                    perturbations=int(config["stability_perturbations"]),seed=seed+peak+i,
                                    previous_x=(segment.x[alarm_row-1] if alarm_row>0 else x),
                                    state_only=state_only) if i<10 else []
                stability.extend(stab)
                before=nearest_distances(detector,x,dx,refs_x,model_refs_dx)
                after=nearest_distances(detector,rec_all["edited_x"],rec_all["edited_dx"],refs_x,model_refs_dx)
                normal_dist_before.append(before); normal_dist_after.append(after)
                plausibility_pass.append(float(after<=plausibility_q95))
                row={"model":model_name,"file":segment.name,"event_index":i,"start_row":start,"end_row_exclusive":end,
                     "peak_row":peak,"peak_score":float(score[peak]),"explanation_row":alarm_row,
                     "explanation_score":float(score[alarm_row]),"threshold":threshold,
                     "detected_in_event":pred_detected,"delay_seconds_from_label_start":(float(np.flatnonzero(pred[start:end])[0]) if pred_detected else None),
                     "active_process_labels":sorted(active),"native_top_process":loc_native["predicted_process"],
                     "native_top10_tags":"|".join(dataset.feature_names[j] for j in np.argsort(-native)[:10]),
                     "ablation_top10_tags":"|".join(dataset.feature_names[j] for j in np.argsort(-abl)[:10]),
                     "native_hit_at_1":loc_native["hit_at_1"],"ablation_top_process":loc_ab["predicted_process"],
                     "ablation_hit_at_1":loc_ab["hit_at_1"],"ablation_hit_at_3":loc_ab["hit_at_3"],
                     "counterfactual_all_success":rec_all["success"],"counterfactual_all_edit_count":rec_all["edit_count"],
                     "counterfactual_all_normalized_l1":rec_all["normalized_l1"],"counterfactual_all_names":";".join(rec_all["edit_names"]),
                     "counterfactual_all_score":rec_all["counterfactual_score"],"normal_reference_distance_before":before,
                     "normal_reference_distance_after":after,"normal_reference_q95_validation":plausibility_q95,
                     "counterfactual_within_validation_reference_q95":bool(after<=plausibility_q95),
                     "counterfactual_score_margin":float(threshold-rec_all["counterfactual_score"]),
                     "capability_support_success":rec_cap["success"],"capability_support_reason":rec_cap["reason"],
                     "capability_groups_declared":";".join(sorted(capability)),
                     "capability_max_groups":int(config["capability_max_groups"]),
                     "capability_max_tags":int(config["capability_max_tags"]),
                     "capability_minimum_required_groups":rec_cap["minimum_required_groups"],
                     "capability_minimum_required_tags":rec_cap["minimum_required_tags"],
                     "capability_edited_groups":";".join(rec_cap["edited_groups"]),
                     "capability_edit_count":rec_cap["edit_count"],"capability_edit_names":";".join(rec_cap["edit_names"]),
                     "capability_normalized_l1":rec_cap["normalized_l1"],
                     "capability_counterfactual_score":rec_cap["counterfactual_score"],
                     "capability_transition_consistency_max_abs_error":rec_cap.get("transition_consistency_max_abs_error"),
                     "capability_support_distance":rec_cap["support_distance"],
                     "capability_validation_support_q95":plausibility_q95,
                     "capability_stage_hit_at_1_after_generation":loc_cap["hit_at_1"] if active and rec_cap["success"] else None,
                     "capability_candidate_count":rec_cap["candidates_checked"],
                     "dynamics_constrained_success":rec_dyn["success"],
                     "dynamics_constrained_reason":rec_dyn["reason"],
                     "dynamics_constrained_edit_count":rec_dyn["edit_count"],
                     "dynamics_constrained_edit_names":";".join(rec_dyn["edit_names"]),
                     "dynamics_constrained_edited_groups":";".join(rec_dyn["edited_groups"]),
                     "dynamics_constrained_score":float(rec_dyn["counterfactual_score"]),
                     "dynamics_transition_score":safe_float(rec_dyn.get("process_transition_score")),
                     "dynamics_transition_threshold":float(transition_threshold),
                     "dynamics_transition_rejections":int(rec_dyn.get("process_transition_rejections",0)),
                     "dynamics_transition_consistency_max_abs_error":rec_dyn.get("transition_consistency_max_abs_error"),
                     "dynamics_explanation_seconds":float(dyn_s),
                     "explanation_seconds":expl_s,"ablation_seconds":abl_s,"top5_stability_mean":float(np.mean(stab)) if stab else None}
                model_rows.append(row); model_event_rows.append(row)
            fa_result = explain_false_alarm_episodes(
                model_name, detector, state_only, segment, segment_dx, pred, dataset.feature_names,
                threshold, seed, int(config.get("false_alarm_explanations_per_file", 20)),
                refs_x, model_refs_dx, plausibility_q95, set(config["capability_groups"]),
                int(config["capability_max_groups"]), int(config["capability_donor_count"]),
                int(config["capability_max_tags"]), transition_model, transition_threshold)
            pure_false_alarm_count += int(fa_result["total_pure_false_alarm_episodes"])
            false_alarm_explanation_rows.extend(fa_result["rows"])
            all_false_alarm_explanation_rows.extend(fa_result["rows"])
            prediction_frame=pd.DataFrame({"file":segment.name,"row":np.arange(len(score)),"timestamp":segment.timestamps,
                "attack":segment.attack,"attack_P1":segment.process_labels[:,0],"attack_P2":segment.process_labels[:,1],
                "attack_P3":segment.process_labels[:,2],"score":score,"prediction":pred.astype(np.int8),"model":model_name})
            prediction_frame.to_csv(prediction_stream,index=False,header=prediction_header)
            prediction_header=False
        false_alarm_frame = pd.DataFrame(false_alarm_explanation_rows)
        if len(false_alarm_frame):
            fa_success = false_alarm_frame[false_alarm_frame.capability_support_success.astype(bool)]
            fa_dynamics_success = false_alarm_frame[false_alarm_frame.dynamics_constrained_success.astype(bool)]
            false_alarm_explanation_summary = {
                "total_pure_false_alarm_episodes": pure_false_alarm_count,
                "sampled_episodes": int(len(false_alarm_frame)),
                "sample_fraction": float(len(false_alarm_frame) / max(1, pure_false_alarm_count)),
                "sampled_greedy_score_flip_rate": float(false_alarm_frame.greedy_score_flip.mean()),
                "sampled_capability_support_success_rate": float(false_alarm_frame.capability_support_success.mean()),
                "sampled_dynamics_constrained_success_rate": float(false_alarm_frame.dynamics_constrained_success.mean()),
                "median_edited_tags_on_success": float(fa_success.capability_edit_count.median()) if len(fa_success) else None,
                "median_edited_tags_dynamics_constrained_success": float(fa_dynamics_success.dynamics_constrained_edit_count.median()) if len(fa_dynamics_success) else None,
                "explanation_latency_p50_ms": float(false_alarm_frame.explanation_seconds.quantile(.50) * 1000),
                "explanation_latency_p95_ms": float(false_alarm_frame.explanation_seconds.quantile(.95) * 1000),
                "dynamics_explanation_latency_p95_ms": float(false_alarm_frame.dynamics_explanation_seconds.quantile(.95) * 1000),
                "interpretation": "Post-hoc sampled pure false-alarm episodes; no target truth is available, so recourse success is behavior only, not explanation correctness."
            }
        else:
            false_alarm_explanation_summary = {
                "total_pure_false_alarm_episodes": pure_false_alarm_count, "sampled_episodes": 0,
                "sample_fraction": 0.0, "sampled_greedy_score_flip_rate": None,
                "sampled_capability_support_success_rate": None, "median_edited_tags_on_success": None,
                "sampled_dynamics_constrained_success_rate": None,
                "median_edited_tags_dynamics_constrained_success": None,
                "explanation_latency_p50_ms": None, "explanation_latency_p95_ms": None,
                "dynamics_explanation_latency_p95_ms": None,
                "interpretation": "No sampled pure false-alarm episodes."
            }
        y=np.concatenate(point_y); sc=np.concatenate(point_scores)
        overall=binary_metrics(y,sc,threshold)
        # Single-window latency approximates online detector scoring; batch throughput is reported separately.
        sample_ix=np.linspace(0,len(dataset.validation.x)-1,min(500,len(dataset.validation.x)),dtype=int)
        for ix in sample_ix:
            xx=dataset.validation.x[ix:ix+1]; dd=val_dx[ix:ix+1]
            t1=time.perf_counter(); detector.score(xx,dd); point_latency.append((time.perf_counter()-t1)*1000)
        total_events=sum(x["event_metrics"]["attack_events"] for x in file_summaries)
        detected_events=sum(x["event_metrics"]["detected_attack_events"] for x in file_summaries)
        false_alarm_events=sum(x["event_metrics"]["false_alarm_events"] for x in file_summaries)
        if pure_false_alarm_count != false_alarm_events:
            raise RuntimeError("False-alarm explanation sampler and event metric disagree on episode count")
        normal_hours=sum((int((~np.asarray(seg.attack,dtype=bool)).sum())/3600.0) for seg in dataset.tests)
        summary={"threshold":threshold,"validation_false_positive_rate_observed":val_fpr,"fit_seconds":fit_s,
                 "validation_score_seconds":None,"overall_point_metrics":overall,
                 "overall_event_metrics":{"attack_events":total_events,"detected_attack_events":detected_events,
                    "event_recall":detected_events/max(1,total_events),"event_recall_ci95":bootstrap_ci(event_detect,seed),
                    "mean_detection_delay_seconds":float(np.mean(delays)) if delays else None,
                    "mean_detection_delay_ci95":bootstrap_ci(delays,seed) if delays else None,
                    "median_detection_delay_seconds":float(np.median(delays)) if delays else None,
                    "false_alarm_events":false_alarm_events,"false_alarm_events_per_normal_hour":false_alarm_events/max(normal_hours,1e-9)},
                 "process_localization":{"events_with_process_labels":len(hit_ab),"native_hit_at_1":float(np.mean(hit_native)) if hit_native else None,
                    "native_hit_at_1_ci95":bootstrap_ci(hit_native,seed),
                    "grouped_ablation_hit_at_1":float(np.mean(hit_ab)) if hit_ab else None,
                    "grouped_ablation_hit_at_1_ci95":bootstrap_ci(hit_ab,seed),
                    "grouped_ablation_hit_at_3":float(np.mean([r["ablation_hit_at_3"] for r in model_rows if r["active_process_labels"]])) if hit_ab else None},
                 "counterfactual":{"unconstrained_greedy_score_flip_rate":float(np.mean(cf_all)) if cf_all else None,
                    "unconstrained_greedy_score_flip_ci95":bootstrap_ci(cf_all,seed),
                    "capability_support_score_flip_rate":float(np.mean(cf_cap)) if cf_cap else None,
                    "capability_support_score_flip_ci95":bootstrap_ci(cf_cap,seed),
                    "dynamics_constrained_score_flip_support_rate":float(np.mean(cf_dyn)) if cf_dyn else None,
                    "dynamics_constrained_score_flip_support_ci95":bootstrap_ci(cf_dyn,seed),
                    "dynamics_constrained_successes":int(np.sum(cf_dyn)) if cf_dyn else 0,
                    "dynamics_constrained_n_events":len(cf_dyn),
                    "dynamics_constrained_stage_hit_at_1_given_success":float(np.mean(cf_dyn_hit)) if cf_dyn_hit else None,
                    "dynamics_constrained_stage_hit_at_1_ci95_given_success":bootstrap_ci(cf_dyn_hit,seed),
                    "dynamics_constrained_median_edit_count_on_success":float(np.median([v for v in cf_dyn_edit_counts if v is not None])) if any(v is not None for v in cf_dyn_edit_counts) else None,
                    "dynamics_constrained_median_transition_score_on_success":float(np.median(cf_dyn_transition_scores)) if cf_dyn_transition_scores else None,
                    "dynamics_validation_q95_threshold":float(transition_threshold),
                    "dynamics_model":transition_model.fit_metadata,
                    "capability_support_successes":int(np.sum(cf_cap)) if cf_cap else 0,
                    "capability_support_n_events":len(cf_cap),
                    "capability_max_groups":int(config["capability_max_groups"]),
                    "capability_max_tags":int(config["capability_max_tags"]),
                    "capability_donor_count":int(config["capability_donor_count"]),
                    "normal_reference_count":int(ref_n),
                    "capability_success_rate_by_group_budget":{
                        k:float(np.mean(values)) for k,values in cf_group_budget.items() if values},
                    "capability_success_rate_by_tag_budget":{
                        k:float(np.mean(values)) for k,values in cf_tag_budget.items() if values},
                    "capability_declared_groups":sorted(set(config["capability_groups"])),
                    "capability_minimum_required_groups_distribution":{str(k):int(sum(v==k for v in cf_cap_groups)) for k in sorted(set(v for v in cf_cap_groups if v is not None))},
                    "capability_minimum_required_tags_distribution":{str(k):int(sum(v==k for v in cf_cap_tag_counts)) for k in sorted(set(v for v in cf_cap_tag_counts if v is not None))},
                    "capability_selected_group_count_distribution":{str(k):int(sum(v==k for v in cf_cap_group_counts)) for k in sorted(set(cf_cap_group_counts))},
                    "capability_median_edit_count_on_success":float(np.median(cf_cap_edit_counts)) if cf_cap_edit_counts else None,
                    "capability_median_group_count_on_success":float(np.median(cf_cap_group_counts)) if cf_cap_group_counts else None,
                    "capability_median_normalized_edit_cost_on_success":float(np.median(cf_cap_cost)) if cf_cap_cost else None,
                    "capability_median_support_distance_on_success":float(np.median(cf_cap_support)) if cf_cap_support else None,
                    "capability_stage_hit_at_1_given_success":float(np.mean(cf_cap_hit)) if cf_cap_hit else None,
                    "capability_stage_hit_at_1_ci95_given_success":bootstrap_ci(cf_cap_hit,seed),
                    "fraction_within_validation_reference_q95":float(np.mean(plausibility_pass)) if plausibility_pass else None,
                    "median_nearest_normal_distance_before":float(np.median(normal_dist_before)) if normal_dist_before else None,
                    "median_nearest_normal_distance_after":float(np.median(normal_dist_after)) if normal_dist_after else None,
                    "capability_model":"Assumes the attacker can edit up to the configured number of observed process-prefix groups; groups and budget are fixed without test labels.",
                    "support_constraint_scope":"Empirical distance to sampled training-normal records calibrated by a validation-normal q95 threshold; not a plant-physics or safety certificate.",
                    "search_scope":"Greedy prefixes of grouped-ablation ranks and a bounded set of nearest training-normal donors; approximate search, no optimality guarantee."},
                 "stability":{"top5_jaccard_mean":float(np.mean(stability)) if stability else None,
                    "top5_jaccard_ci95":bootstrap_ci(stability,seed)},
                 "latency":{"per_event_explanation_p50_ms":float(np.quantile(explain_times,.50)*1000) if explain_times else None,
                    "per_event_explanation_p95_ms":float(np.quantile(explain_times,.95)*1000) if explain_times else None,
                    "dynamics_constrained_search_p50_ms":float(np.quantile(cf_dyn_times,.50)*1000) if cf_dyn_times else None,
                    "dynamics_constrained_search_p95_ms":float(np.quantile(cf_dyn_times,.95)*1000) if cf_dyn_times else None,
                    "single_sample_score_p50_ms":float(np.quantile(point_latency,.50)) if point_latency else None,
                    "single_sample_score_p95_ms":float(np.quantile(point_latency,.95)) if point_latency else None,
                    "batch_throughput_samples_per_second":float(sum(len(seg.x) for seg in dataset.tests)/max(1e-9,sum(f["score_seconds"] for f in file_summaries)))},
                 "false_alarm_explanation_sample":false_alarm_explanation_summary,
                 "per_file":file_summaries,"events":model_rows}
        summaries[model_name]=summary
        scores_for_comparison[model_name]=event_detect
    # Paired event bootstrap comparison of temporal vs state-only PCA detection recall.
    comparison=paired_bootstrap_difference(scores_for_comparison.get("PCA-TEMPORAL",[]),scores_for_comparison.get("PCA-STATE",[]),seed)
    summary={"run_id":run_dir.name,"created_utc":datetime.now(timezone.utc).isoformat(),"dataset":"HAI 21.03","data_load_seconds":data_load_seconds,
             "data_manifest":dataset.manifest,"data_release_url":"https://github.com/icsdataset/hai/tree/master/hai-21.03",
             "schema":{"rows_train":int(len(dataset.train_x)),"features":len(dataset.feature_names),"train_files":["train1.csv.gz","train2.csv.gz"],
                       "validation_file":"train3.csv.gz","test_files":[s.name for s in dataset.tests],"test_rows":int(sum(len(s.x) for s in dataset.tests)),
                       "test_attack_events":int(sum(len(contiguous_events(s.attack)) for s in dataset.tests)),
                       "features_by_process":{g:sum(1 for x in dataset.process_groups if x==g) for g in sorted(set(dataset.process_groups))},
                       "labels":["attack","attack_P1","attack_P2","attack_P3"]},
             "protocol":{"threshold_quantile":1-config["validation_false_positive_rate"],"threshold_fit":"normal-only train3 validation file",
                         "scaler_fit":"normal-only train1 and train2, sampled with configured seed",
                         "windowing":"one-step process state and first difference; files are never concatenated across boundaries",
                         "test_labels_used_for_fit_or_threshold":False,"confidence_intervals":"event-level percentile bootstrap, 2000 draws",
                         "explanation_label_use":"attack labels are used only for post-hoc evaluation and pure-false-alarm episode sampling; detector, attribution, and counterfactual generation do not receive test labels",
                         "false_alarm_explanation_sample_per_file":int(config.get("false_alarm_explanations_per_file",20)),
                         "false_alarm_episode_definition":"predicted-positive contiguous episodes with no attack-label overlap, matching event_metrics",
                         "normal_transition_model":transition_model.fit_metadata,
                         "normal_transition_validation_threshold":float(transition_threshold),
                         "normal_transition_validation_quantile":1-float(config.get("transition_validation_false_positive_rate",0.05))},
             "software":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"platform":platform.platform(),
                         "cpu_count":__import__("os").cpu_count(),"processor":platform.processor(),"peak_working_set_mb":peak_working_set_mb()},
             "models":summaries,"paired_event_detection_difference_PCA_temporal_minus_state":comparison,
             "seed":seed,"config":config}
    prediction_stream.close()
    (run_dir/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    pd.DataFrame([row for model in summaries.values() for row in model["events"]]).to_csv(run_dir/"event_explanations.csv",index=False)
    pd.DataFrame(all_false_alarm_explanation_rows).to_csv(run_dir/"false_alarm_explanations.csv",index=False)
    print(json.dumps({"run_dir":str(run_dir),"models":{k:{"f1":v["overall_point_metrics"]["f1"],"event_recall":v["overall_event_metrics"]["event_recall"],
                "process_hit1":v["process_localization"]["grouped_ablation_hit_at_1"],"cf_success":v["counterfactual"]["capability_support_score_flip_rate"],
                "explain_p95_ms":v["latency"]["per_event_explanation_p95_ms"]} for k,v in summaries.items()},
                "paired_detection_test":comparison},indent=2))

if __name__=="__main__":
    main()
