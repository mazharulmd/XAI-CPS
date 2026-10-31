from __future__ import annotations

import argparse
import gzip
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from xai_cps.data import load_batadal, contiguous_events
from xai_cps.detectors import RobustZDetector,PCAResidualDetector
from xai_cps.explainers import capability_support_counterfactual,grouped_ablation,greedy_counterfactual,stability_topk
from xai_cps.metrics import bootstrap_ci


def nearest_distances(detector,x,dx,refs_x,refs_dx):
    zx,zdx=detector.standardized(x[None,:],dx[None,:])
    rz,rdz=detector.standardized(refs_x,refs_dx)
    refs=np.concatenate([rz,rdz],axis=1);q=np.concatenate([zx[0],zdx[0]])
    return float(np.sqrt(np.min(np.sum((refs-q)**2,axis=1))))


def model_specs(dataset,config):
    zero_train=np.zeros_like(dataset.train_dx);zero_val=np.zeros_like(dataset.validation.dx)
    return [("RobustZ-TEMPORAL",RobustZDetector(),dataset.train_dx,dataset.validation.dx,False),
            ("PCA-STATE",PCAResidualDetector(config["variance_explained"],config["max_components"]),zero_train,zero_val,True),
            ("PCA-TEMPORAL",PCAResidualDetector(config["variance_explained"],config["max_components"]),dataset.train_dx,dataset.validation.dx,False)]


def main():
    parser=argparse.ArgumentParser(description="Evaluate on BATADAL's partially labeled development attacks.")
    parser.add_argument("--seed",type=int,default=None)
    args=parser.parse_args()
    config_path=ROOT/"configs"/"batadal.json"
    config=json.loads(config_path.read_text(encoding="utf-8"))
    seed=config["random_seed"] if args.seed is None else args.seed
    start=time.perf_counter();data=load_batadal(config_path);load_seconds=time.perf_counter()-start
    unknown=set(config["capability_groups"])-set(data.process_groups)
    if unknown: raise ValueError(f"Capability groups absent from BATADAL schema: {sorted(unknown)}")
    ts=data.tests[0]
    known=np.asarray(ts.label_known,dtype=bool)
    attack=np.asarray(ts.attack,dtype=bool)
    events=contiguous_events(attack & known)
    specs=model_specs(data,config)
    rng=np.random.default_rng(seed)
    ref_n=min(int(config["normal_reference_count"]),len(data.train_x))
    ref_idx=np.sort(rng.choice(len(data.train_x),size=ref_n,replace=False))
    refs_x,refs_dx=data.train_x[ref_idx],data.train_dx[ref_idx]
    val_distance_idx=np.linspace(0,len(data.validation.x)-1,min(512,len(data.validation.x)),dtype=int)
    model_results={};all_events=[]
    prediction_path=None
    run_stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir=ROOT/"results"/"runs"/f"batadal_seed{seed}_{run_stamp}"
    run_dir.mkdir(parents=True,exist_ok=True)
    prediction_path=run_dir/"predictions.csv.gz"
    pred_stream=gzip.open(prediction_path,"wt",encoding="utf-8",newline="")
    write_header=True

    for model_name,detector,train_dx,val_dx,state_only in specs:
        fit_start=time.perf_counter();detector.fit(data.train_x,train_dx);fit_seconds=time.perf_counter()-fit_start
        model_refs_dx=np.zeros_like(refs_dx) if state_only else refs_dx
        val_scores=detector.score(data.validation.x,val_dx)
        q=1-float(config["validation_false_positive_rate"])
        threshold=float(np.quantile(val_scores,q))
        val_fpr=float(np.mean(val_scores>threshold))
        vz,vdz=detector.standardized(data.validation.x[val_distance_idx],val_dx[val_distance_idx])
        rz,rdz=detector.standardized(refs_x,model_refs_dx);ref_join=np.concatenate([rz,rdz],axis=1)
        val_dist=[]
        for a,b in zip(vz,vdz):
            query=np.concatenate([a,b]);val_dist.append(float(np.sqrt(np.min(np.sum((ref_join-query)**2,axis=1)))))
        support_q95=float(np.quantile(val_dist,.95))
        dx=ts.dx if model_name!="PCA-STATE" else np.zeros_like(ts.dx)
        score_start=time.perf_counter();scores=detector.score(ts.x,dx);score_seconds=time.perf_counter()-score_start
        pred=scores>threshold
        # Unknown (-999) rows are never treated as normal or negative test labels.
        labeled_positive=known & attack
        point_recall=float(np.mean(pred[labeled_positive])) if labeled_positive.any() else None
        event_detected=[];delays=[];false_alarm_rows=[];event_rows=[];explain_times=[];cap_results=[]
        for event_idx,(lo,hi) in enumerate(events):
            alarm_offsets=np.flatnonzero(pred[lo:hi])
            detected=bool(alarm_offsets.size)
            event_detected.append(float(detected))
            if not detected: continue
            alarm=lo+int(alarm_offsets[0])
            delay=float((ts.timestamps[alarm]-ts.timestamps[lo])/np.timedelta64(1,"s"))
            delays.append(delay)
            tick=time.perf_counter()
            previous_x=ts.x[alarm-1] if alarm>0 else ts.x[alarm]
            abl,abl_seconds=grouped_ablation(
                detector,ts.x[alarm],dx[alarm],data.feature_names,
                previous_x=previous_x,state_only=state_only)
            greedy=greedy_counterfactual(
                detector,ts.x[alarm],dx[alarm],abl,threshold,data.feature_names,
                previous_x=previous_x,state_only=state_only)
            cap=capability_support_counterfactual(
                detector,ts.x[alarm],dx[alarm],abl,threshold,data.feature_names,refs_x,model_refs_dx,support_q95,
                capability_groups=set(config["capability_groups"]),max_groups=int(config["capability_max_groups"]),
                donor_count=int(config["capability_donor_count"]),max_tags=int(config["capability_max_tags"]),
                previous_x=previous_x,state_only=state_only)
            stable=stability_topk(detector,ts.x[alarm],dx[alarm],data.feature_names,5,
                                  int(config["stability_perturbations"]),seed=seed+event_idx,
                                  previous_x=previous_x,state_only=state_only)
            native_scores=np.asarray(detector.contributions(ts.x[alarm:alarm+1],dx[alarm:alarm+1])[0],dtype=float)
            elapsed=time.perf_counter()-tick;explain_times.append(elapsed);cap_results.append(cap)
            before=nearest_distances(detector,ts.x[alarm],dx[alarm],refs_x,model_refs_dx)
            after=nearest_distances(detector,greedy["edited_x"],greedy["edited_dx"],refs_x,model_refs_dx)
            event_rows.append({"model":model_name,"event_index":event_idx,"start_row":lo,"end_row_exclusive":hi,
                "alarm_row":alarm,"approx_label_delay_seconds":delay,"alarm_score":float(scores[alarm]),
                "threshold":threshold,"detected":detected,"known_attack_positive_label":True,
                "greedy_scoreflip":greedy["success"],"greedy_tag_edits":greedy["edit_count"],
                "greedy_normalized_l1":greedy["normalized_l1"],"greedy_support_distance":after,
                "validation_support_q95":support_q95,"greedy_passes_support":after<=support_q95,
                "capability_support_scoreflip":cap["success"],"capability_reason":cap["reason"],
                "capability_min_groups":cap["minimum_required_groups"],"capability_min_tags":cap["minimum_required_tags"],
                "capability_edited_groups":";".join(cap["edited_groups"]),"capability_edits":cap["edit_count"],
                "capability_edit_names":";".join(cap["edit_names"]),"capability_normalized_l1":cap["normalized_l1"],
                "capability_support_distance":cap["support_distance"],
                "capability_group_budget_success":json.dumps(cap["success_by_group_budget"],sort_keys=True),
                "capability_tag_budget_success":json.dumps(cap["success_by_tag_budget"],sort_keys=True),
                "support_distance_before":before,"explanation_seconds":elapsed,"ablation_seconds":abl_seconds,
                "top5_stability_mean":float(np.mean(stable)),
                "native_tag_scores":{name:float(value) for name,value in zip(data.feature_names,native_scores)},
                "grouped_ablation_tag_scores":{name:float(value) for name,value in zip(data.feature_names,abl)}})
        frame=pd.DataFrame({"timestamp":ts.timestamps,"attack_flag_raw":np.where(known,attack.astype(np.int8),-999),
            "label_known":known.astype(np.int8),"score":scores,"prediction":pred.astype(np.int8),"model":model_name})
        frame.to_csv(pred_stream,index=False,header=write_header);write_header=False
        detected=sum(event_detected)
        model_results[model_name]={"threshold":threshold,"validation_point_fpr":val_fpr,"fit_seconds":fit_seconds,
            "validation_support_distance_q95":support_q95,"test_partial_label_rows":int(known.sum()),
            "test_unknown_label_rows":int((~known).sum()),"known_attack_positive_rows":int(labeled_positive.sum()),
            "known_attack_positive_point_recall":point_recall,"approx_labeled_attack_events":len(events),
            "detected_approx_labeled_attack_events":int(detected),"approx_labeled_event_recall":detected/max(1,len(events)),
            "approx_label_mean_detection_delay_seconds":float(np.mean(delays)) if delays else None,
            "event_recall_bootstrap_ci95":bootstrap_ci(event_detected,seed),
            "event_detection_by_index":[bool(value) for value in event_detected],
            "explanation_events":len(event_rows),
            "capability_support_scoreflip_rate":float(np.mean([x["success"] for x in cap_results])) if cap_results else None,
            "capability_success_by_group_budget":{k:float(np.mean([x["success_by_group_budget"][k] for x in cap_results])) for k in sorted(cap_results[0]["success_by_group_budget"],key=int)} if cap_results else {},
            "capability_success_by_tag_budget":{k:float(np.mean([x["success_by_tag_budget"][k] for x in cap_results])) for k in sorted(cap_results[0]["success_by_tag_budget"],key=int)} if cap_results else {},
            "capability_median_edit_count":float(np.median([x["edit_count"] for x in cap_results if x["success"]])) if any(x["success"] for x in cap_results) else None,
            "capability_median_normalized_l1":float(np.median([x["normalized_l1"] for x in cap_results if x["success"]])) if any(x["success"] for x in cap_results) else None,
            "capability_target_metadata":"not passed to the explainer; the official BATADAL FAQ provides post-hoc pump-component targets for the five approximate positive intervals, but not exact manipulated channels",
            "explanation_p50_ms":float(np.quantile(explain_times,.5)*1000) if explain_times else None,
            "explanation_p95_ms":float(np.quantile(explain_times,.95)*1000) if explain_times else None,
            "test_false_positive_rate":None,"test_precision":None,"test_f1":None,
            "test_metric_note":"BATADAL dataset04 has only ATT_FLAG=1 annotated positives and ATT_FLAG=-999 unknown rows; false positives, precision, and F1 cannot be estimated from this file."}
        all_events.extend(event_rows)
    pred_stream.close()
    summary={"run_id":run_dir.name,"created_utc":datetime.now(timezone.utc).isoformat(),"dataset":"BATADAL development dataset",
        "source":"https://www.batadal.net/data.html","data_manifest":data.manifest,"data_load_seconds":load_seconds,
        "chronological_split":{"normal_file":config["normal_file"],"train_rows":len(data.train_x),"validation_rows":len(data.validation.x),
            "train_fraction":config["train_fraction"],"partial_attack_file":config["partial_label_file"],"attack_rows":len(ts.x),
            "features":len(data.feature_names),"known_attack_events":len(events),"unknown_label_value":-999},
        "protocol":{"normalization_fit":"chronological first 75% of normal-only Dataset03",
            "threshold":"normal-only final 25% of Dataset03 at configured point FPR",
            "positive_evaluation":"only ATT_FLAG=1 rows/events in Dataset04; attack labels are approximate",
            "unknown_handling":"ATT_FLAG=-999 is masked from test-negative, precision, FPR, and F1 metrics",
            "capability":"analyst-declared channel-prefix group budgets; no labels passed to explanation generation",
            "support":"validation-normal q95 distance to 256 sampled normal training references; empirical proxy only"},
        "software":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"platform":platform.platform()},
        "models":model_results,"explanations":all_events,"seed":seed,"config":config}
    (run_dir/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    pd.DataFrame(all_events).to_csv(run_dir/"event_explanations.csv",index=False)
    print(json.dumps({"run_dir":str(run_dir),"labeled_event_recall":{m:v["approx_labeled_event_recall"] for m,v in model_results.items()},
                      "capability_support":{m:v["capability_support_scoreflip_rate"] for m,v in model_results.items()},
                      "note":"No precision/F1/test-FPR reported because other BATADAL labels are unknown."},indent=2))


if __name__=="__main__": main()
