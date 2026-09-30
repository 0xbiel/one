"""Tests for scoring integrity, not detector accuracy."""
import json,subprocess,tempfile,unittest
from pathlib import Path
P=Path(__file__).parent
class ScoringTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.d=Path(self.tmp.name);self.manifest=json.loads((P/'assets/manifest.json').read_text());self.rows=[]
  for c in self.manifest['clips']:
   for i in range(80):self.rows.append(dict(clip_id=c['id'],frame_index=i,timestamp_s=i/10,person_detections=0,stable_person_tracks=0,events=[]))
  self.meta=dict(code_commit='scorer-unit-test',model_name='none-unit-test',model_sha256='none-unit-test',input_kind='rendered_rgb_frames',inference_fps=10)
 def tearDown(self):self.tmp.cleanup()
 def run_score(self):
  (self.d/'input.jsonl').write_text('\n'.join(json.dumps(x) for x in self.rows));(self.d/'meta.json').write_text(json.dumps(self.meta))
  return subprocess.run(['python',str(P/'score_results.py'),str(self.d/'input.jsonl'),'--metadata',str(self.d/'meta.json'),'--output',str(self.d/'output.json')],capture_output=True)
 def test_no_recognition_is_missed_fall(self):
  self.assertEqual(self.run_score().returncode,0);o=json.loads((self.d/'output.json').read_text());self.assertEqual(o['counts'],dict(TP=0,FN=3,FP=0,TN=4));self.assertIsNone(o['median_delay_s'])
 def test_ground_truth_injection_rejected(self):
  self.meta['input_kind']='oracle_ground_truth_boxes';self.assertNotEqual(self.run_score().returncode,0)
 def test_missing_frame_rejected(self):
  self.rows.pop();self.assertNotEqual(self.run_score().returncode,0)
 def test_bad_timestamp_rejected(self):
  self.rows[5]['timestamp_s']=2.5;self.assertNotEqual(self.run_score().returncode,0)
 def test_signal_counts_and_delay(self):
  self.rows[30]['events']=[dict(event_type='fall_suspected')];self.rows[3*80+40]['events']=[dict(event_type='fall_suspected')]
  self.assertEqual(self.run_score().returncode,0);o=json.loads((self.d/'output.json').read_text());self.assertEqual(o['counts'],dict(TP=1,FN=2,FP=1,TN=3));self.assertEqual(o['median_delay_s'],1)
if __name__=='__main__':unittest.main()
