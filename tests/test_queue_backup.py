"""Headless snapshot helper tests; no production queue modifications."""
import importlib.util
import pathlib
import unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
class SnapshotTests(unittest.TestCase):
    def test_running_first_sorted_pending_metadata_and_no_credentials(self):
        path=ROOT/'queue_backup.py'
        self.assertTrue(path.exists(), 'Headless queue backup helper missing')
        spec=importlib.util.spec_from_file_location('queue_backup',path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        def row(n,i): return [n,i,{'1':{'class_type':'SaveImage','inputs':{'seed':42}}},{'client_id':'old','extra_pnginfo':{'workflow':{}},'api_key_comfy_org':'SECRET'},['1'],{'auth_token_comfy_org':'SECRET'}]
        result=module.make_snapshot({'queue_running':[row(0,'run')],'queue_pending':[row(4,'last'),row(1,'first')]})
        self.assertEqual([j['source_prompt_id'] for j in result['jobs']],['run','first','last'])
        self.assertTrue(result['jobs'][0]['was_running'])
        self.assertEqual(result['jobs'][0]['prompt']['1']['inputs']['seed'],42)
        self.assertEqual(result['jobs'][0]['outputs_to_execute'],['1'])
        self.assertNotIn('SECRET',str(result))
        self.assertNotIn('client_id',result['jobs'][0]['extra_data'])
if __name__=='__main__': unittest.main()
