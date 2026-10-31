import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from xai_cps.data import Segment, contiguous_events,load_batadal
from xai_cps.detectors import RobustZDetector, PCAResidualDetector
from xai_cps.dynamics import NormalTransitionModel
from xai_cps.explainers import (capability_support_counterfactual, grouped_ablation,
                                greedy_counterfactual, stability_topk)
from xai_cps.metrics import binary_metrics, event_metrics
from scripts.run_hai import detector_reference_delta, explain_false_alarm_episodes
from scripts.evaluate_hai_target_localization import evaluate_run


class EventTests(unittest.TestCase):
    def test_contiguous_events_half_open(self):
        self.assertEqual(contiguous_events(np.array([0,1,1,0,1,0])),[(1,3),(4,5)])

    def test_point_metrics_and_event_delay(self):
        y=np.array([0,1,1,0,0,1],dtype=bool)
        s=np.array([0.0,0.9,0.8,0.1,0.2,0.7])
        m=binary_metrics(y,s,0.5)
        self.assertEqual((m['tp'],m['fp'],m['fn']),(3,0,0))
        ts=np.arange('2024-01-01T00:00:00','2024-01-01T00:00:06',dtype='datetime64[s]')
        ev=event_metrics(y,s>0.5,ts)
        self.assertEqual(ev['attack_events'],2)
        self.assertEqual(ev['detected_attack_events'],2)
        self.assertEqual(ev['mean_detection_delay_seconds'],0.0)

    def test_unconditional_counterfactual_localization_counts_missed_events_as_misses(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / 'run_seed7'
            run_dir.mkdir()
            mapping = pd.DataFrame([
                {'attack_id':'A1','dataset_file':'test.csv.gz','event_index':0,
                 'target_annotation_complete':True,'target_points_present_in_csv':'P1_TAG'},
                {'attack_id':'A2','dataset_file':'test.csv.gz','event_index':1,
                 'target_annotation_complete':True,'target_points_present_in_csv':'P1_TAG'},
            ])
            pd.DataFrame([{
                'model':'M','file':'test.csv.gz','event_index':0,
                'native_top10_tags':'P1_TAG','ablation_top10_tags':'P1_TAG',
                'capability_support_success':True,'capability_edit_names':'P1_TAG',
                'dynamics_constrained_success':True,'dynamics_constrained_edit_names':'P1_TAG',
            }]).to_csv(run_dir/'event_explanations.csv', index=False)
            result = evaluate_run(run_dir, mapping)
            model = result['models']['M']
            self.assertEqual(model['capability_cf_target_hits_unconditional'], 1)
            self.assertEqual(model['dynamics_cf_target_hits_unconditional'], 1)
            self.assertEqual(model['capability_cf_target_hit_rate_unconditional'], 0.5)
            self.assertEqual(model['dynamics_cf_target_hit_rate_unconditional'], 0.5)
            outcomes = result['event_outcomes']['M']
            self.assertEqual(outcomes[1]['capability_cf_target_hit_unconditional'], 0.0)
            self.assertEqual(outcomes[1]['dynamics_cf_target_hit_unconditional'], 0.0)

    def test_false_alarm_sampler_excludes_predicted_episodes_overlapping_attacks(self):
        x=np.zeros((6,2)); dx=np.zeros_like(x)
        segment=Segment('test.csv',x,dx,np.array([0,0,0,0,1,1]),np.zeros((6,3),dtype=np.int8),
                        np.arange('2024-01-01T00:00:00','2024-01-01T00:00:06',dtype='datetime64[s]'))
        result=explain_false_alarm_episodes(
            'test-model',None,False,segment,dx,np.array([0,1,1,0,1,1],dtype=bool),['P1_A','P2_A'],
            1.0,42,0,np.zeros((1,2)),np.zeros((1,2)),1.0,set(),1,1,1,None,None)
        self.assertEqual(result['total_pure_false_alarm_episodes'],1)
        self.assertEqual(result['rows'],[])

    def test_batadal_loader_masks_unknown_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            normal_dates=pd.date_range('2014-01-01',periods=8,freq='h').strftime('%d/%m/%y %H')
            partial_dates=pd.date_range('2016-01-01',periods=4,freq='h').strftime('%d/%m/%y %H')
            pd.DataFrame({'DATETIME':normal_dates,'L_T1':np.arange(8,dtype=float),'ATT_FLAG':0}).to_csv(root/'normal.csv',index=False)
            pd.DataFrame({'DATETIME':partial_dates,'L_T1':[0.,1.,2.,3.],'ATT_FLAG':[1,-999,1,-999]}).to_csv(root/'partial.csv',index=False)
            config=root/'config.json'
            config.write_text(json.dumps({'data_dir':str(root),'normal_file':'normal.csv','partial_label_file':'partial.csv','train_fraction':.75}),encoding='utf-8')
            dataset=load_batadal(config)
            test=dataset.tests[0]
            self.assertEqual(len(dataset.train_x),6)
            self.assertEqual(len(dataset.validation.x),2)
            np.testing.assert_array_equal(test.label_known,np.array([True,False,True,False]))
            np.testing.assert_array_equal(test.attack,np.array([True,False,True,False]))


class DetectorTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(5)
        self.x=rng.normal(0,1,size=(2000,4))
        self.dx=np.diff(self.x,axis=0,prepend=self.x[[0]])
        self.names=['P1_A','P1_B','P2_A','P2_B']

    def test_robust_detector_ranks_shifted_sample(self):
        model=RobustZDetector().fit(self.x,self.dx)
        normal=float(model.score(self.x[100:101],self.dx[100:101])[0])
        attack=self.x[100:101].copy();attack[0,0]+=8
        abnormal=float(model.score(attack,self.dx[100:101])[0])
        self.assertGreater(abnormal,normal)

    def test_normal_transition_model_flags_unseen_one_step_jump(self):
        rng=np.random.default_rng(19)
        x=np.zeros((3000,3))
        for t in range(1,len(x)):
            x[t]=0.8*x[t-1]+rng.normal(0,0.03,size=3)
        dx=np.diff(x,axis=0,prepend=x[[0]])
        model=NormalTransitionModel(ridge=0.1,top_k=1).fit(x,dx)
        normal=float(model.score(x[100],x[101])[0])
        jumped=x[101].copy(); jumped[0]+=2.0
        abnormal=float(model.score(x[100],jumped)[0])
        self.assertGreater(abnormal,normal*10+1e-6)
        threshold=model.calibrate(x[:-1],x[1:],false_positive_rate=.05)
        self.assertGreaterEqual(threshold,0.0)

    def test_pca_and_grouped_counterfactual_are_finite(self):
        model=PCAResidualDetector(variance=.9,max_components=3).fit(self.x,self.dx)
        x=self.x[100].copy();dx=self.dx[100].copy();x[0]+=8
        score=float(model.score(x[None,:],dx[None,:])[0])
        importance,_=grouped_ablation(model,x,dx,self.names)
        self.assertTrue(np.isfinite(score))
        self.assertEqual(len(importance),len(self.names))
        cf=greedy_counterfactual(model,x,dx,importance,score,self.names,allowed_groups={'P1'})
        self.assertTrue(all(name.startswith('P1_') for name in cf['edit_names']))
        self.assertTrue(cf['counterfactual_score']<=score+1e-9)

    def test_temporal_ablation_and_greedy_recourse_preserve_state_difference_identity(self):
        model=PCAResidualDetector(variance=.9,max_components=3).fit(self.x,self.dx)
        x=self.x[25].copy(); dx=self.dx[25].copy(); previous=x-dx
        importance,_=grouped_ablation(model,x,dx,self.names,previous_x=previous)
        replaced=x.copy(); replaced[0]=model.x_ref[0]
        replaced_dx=dx.copy(); replaced_dx[0]=replaced[0]-previous[0]
        expected=float(model.score(x[None,:],dx[None,:])[0]-
                       model.score(replaced[None,:],replaced_dx[None,:])[0])
        self.assertAlmostEqual(float(importance[0]),expected,places=10)
        result=greedy_counterfactual(
            model,x,dx,importance,-1.0,self.names,previous_x=previous)
        np.testing.assert_allclose(result['edited_dx'],result['edited_x']-previous,rtol=0,atol=1e-12)

    def test_state_only_greedy_recourse_keeps_zero_difference(self):
        zeros=np.zeros_like(self.x)
        model=PCAResidualDetector(variance=.9,max_components=3).fit(self.x,zeros)
        x=self.x[31].copy(); dx=np.zeros_like(x)
        importance,_=grouped_ablation(model,x,dx,self.names,state_only=True)
        result=greedy_counterfactual(
            model,x,dx,importance,-1.0,self.names,state_only=True)
        np.testing.assert_array_equal(result['edited_dx'],np.zeros_like(x))

    def test_capability_support_counterfactual_obeys_declared_group_budget(self):
        class SquaredScore:
            x_scale=np.ones(4); dx_scale=np.ones(4)
            def score(self, x, dx): return np.sum(x*x+dx*dx,axis=1)
            def standardized(self, x, dx): return x,dx
        model=SquaredScore()
        refs_x=np.zeros((16,4)); refs_dx=np.zeros((16,4))
        x=np.array([3.0,3.0,0.0,0.0]); dx=np.array([3.0,3.0,0.0,0.0])
        result=capability_support_counterfactual(
            model,x,dx,np.array([2.0,1.0,0.0,0.0]),1.0,
            ['P1_A','P1_B','P2_A','P2_B'],refs_x,refs_dx,1.0,
            capability_groups={'P1'},max_groups=1,donor_count=4)
        self.assertTrue(result['success'])
        self.assertEqual(result['edited_groups'],['P1'])
        self.assertLessEqual(result['minimum_required_groups'],1)
        self.assertEqual(result['minimum_required_tags'],2)
        self.assertTrue(all(name.startswith('P1_') for name in result['edit_names']))
        self.assertLessEqual(result['support_distance'],1.0)
        self.assertTrue(result['success_by_group_budget']['1'])
        self.assertFalse(result['success_by_tag_budget']['1'])
        self.assertTrue(result['success_by_tag_budget']['2'])
        np.testing.assert_allclose(result['edited_dx'], result['edited_x'] - (x-dx))
        self.assertEqual(result['transition_consistency_max_abs_error'], 0.0)
        constrained=capability_support_counterfactual(
            model,x,dx,np.array([2.0,1.0,0.0,0.0]),1.0,
            ['P1_A','P1_B','P2_A','P2_B'],refs_x,refs_dx,1.0,
            capability_groups={'P1'},max_groups=1,donor_count=4,transition_score_fn=lambda _p,c: float(np.max(np.abs(c))),
            transition_threshold=0.5)
        self.assertTrue(constrained['success'])
        self.assertLessEqual(constrained['process_transition_score'],0.5)
        rejected=capability_support_counterfactual(
            model,x,dx,np.array([2.0,1.0,0.0,0.0]),1.0,
            ['P1_A','P1_B','P2_A','P2_B'],refs_x,refs_dx,1.0,
            capability_groups={'P1'},max_groups=1,donor_count=4,transition_score_fn=lambda _p,_c: 10.0,
            transition_threshold=0.5)
        self.assertFalse(rejected['success'])
        self.assertGreater(rejected['process_transition_rejections'],0)

    def test_state_only_counterfactual_keeps_zero_delta_representation(self):
        class SquaredScore:
            x_scale=np.ones(2); dx_scale=np.ones(2)
            def score(self, x, dx): return np.sum(x*x,axis=1)
            def standardized(self, x, dx): return x,dx
        model=SquaredScore()
        x=np.array([3.0,0.0]); dx=np.zeros(2)
        refs_x=np.zeros((4,2)); refs_dx=np.zeros((4,2))
        result=capability_support_counterfactual(
            model,x,dx,np.array([3.0,0.0]),1.0,['P1_A','P2_A'],refs_x,refs_dx,1.0,
            capability_groups={'P1'},max_groups=1,max_tags=1,state_only=True)
        self.assertTrue(result['success'])
        np.testing.assert_array_equal(result['edited_dx'],np.zeros_like(dx))

    def test_state_only_reference_deltas_match_detector_input(self):
        dx=np.array([[1.0,-2.0],[3.0,4.0]])
        state_delta=detector_reference_delta(dx,state_only=True)
        temporal_delta=detector_reference_delta(dx,state_only=False)
        np.testing.assert_array_equal(state_delta,np.zeros_like(dx))
        np.testing.assert_array_equal(temporal_delta,dx)

    def test_stability_perturbations_preserve_detector_representation(self):
        class SpyDetector:
            x_scale=np.ones(2); dx_scale=np.ones(2)
            x_ref=np.zeros(2); dx_ref=np.zeros(2)
            def __init__(self): self.seen=[]
            def score(self,x,dx):
                self.seen.append((np.asarray(x).copy(),np.asarray(dx).copy()))
                return np.sum(np.asarray(x)**2+np.asarray(dx)**2,axis=1)
        names=['P1_A','P2_B']
        x=np.array([1.0,2.0]); zero=np.zeros(2)
        state=SpyDetector()
        stability_topk(state,x,zero,names,2,perturbations=3,seed=11,state_only=True)
        self.assertTrue(all(np.all(dx_seen==0.0) for _,dx_seen in state.seen))

        temporal=SpyDetector(); previous=np.array([0.5,1.0]); dx=x-previous
        stability_topk(temporal,x,dx,names,2,perturbations=3,seed=11,previous_x=previous)
        # grouped_ablation edits paired channels after its baseline; every third call
        # is the unedited sample used for ranking.
        for x_seen,dx_seen in temporal.seen[::3]:
            np.testing.assert_allclose(dx_seen,x_seen-previous,rtol=0.0,atol=1e-12)


if __name__=='__main__':
    unittest.main()
