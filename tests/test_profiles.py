"""Data-contract roundtrips exercise the same estimator/search/PD entry points."""
import copy,json,tempfile,unittest
from pathlib import Path
from dataclasses import asdict,replace
from inferfit import *
from inferfit.cli import evaluate_document

class ProfileTests(unittest.TestCase):
    def test_real_model_roundtrips_both_engines(self):
        for model in [load_qwen3_235b(),load_glm51()]:
            clone=load_model_profile(model_profile_to_dict(model))
            for runtime in [RuntimeProfile(),RuntimeProfile.sglang()]:
                plan=Parallelism(8,1,8,2,16)
                a=check(model,parallelism=plan,runtime=runtime)
                b=check(clone,parallelism=plan,runtime=runtime)
                self.assertEqual(a.status,b.status)
                self.assertEqual(a.ranks,b.ranks)
                self.assertEqual(logical_weight_bytes(model),logical_weight_bytes(clone))

    def test_arbitrary_names_keep_semantic_placement(self):
        doc=model_profile_to_dict(load_qwen3_235b())
        a=load_model_profile(doc)
        for i,tensor in enumerate(doc['tensors']):tensor['name']=f'custom_tensor_{i}'
        b=load_model_profile(doc)
        runtime=RuntimeProfile.sglang()
        plan=Parallelism(8,1,8,2,16)
        x=optimize_pipeline(a,parallelism=plan,runtime=runtime,weight_quantization='bf16')
        y=optimize_pipeline(b,parallelism=plan,runtime=runtime,weight_quantization='bf16')
        self.assertEqual(x.summary['stage_layers'],y.summary['stage_layers'])
        self.assertEqual(x.ranks,y.ranks)

    def test_search_and_pd_reuse_external_profile(self):
        m=load_model_profile(model_profile_to_dict(load_qwen3_235b()))
        r=search(m,tp_values=[8],dp_values=[1],pp_values=[1,2],max_gpus=16,weight_quantization='bf16')
        self.assertTrue(r.candidates)
        plan=Parallelism(8,1,8,1,8)
        p=PDPool(tuple(f'p{i}' for i in range(8)),parallelism=plan)
        d=replace(p,gpu_ids=tuple(f'd{i}' for i in range(8)))
        self.assertTrue(check_pd(m,prefill=p,decode=d).memory_pass)

    def test_hardware_new_name_needs_no_catalog_edit(self):
        d={'schema_version':1,'kind':'hardware','hardware':{
            'name':'new-cuda-board','memory_bytes':100_000_000_000,'platform':'cuda',
            'compute_capability':90,'gpus_per_node':4}}
        h=load_hardware_profile(d)
        self.assertEqual(h.name,'new-cuda-board')
        self.assertTrue(check(load_qwen3_235b(),hardware=h,parallelism=Parallelism(4,1,4,2,8),weight_quantization='bf16').memory_pass)
        d['hardware'].update(platform='ppu',compute_capability=None)
        self.assertEqual(check(load_qwen3_235b(),hardware=load_hardware_profile(d)).status,'unsupported')

    def test_explicit_profile_version_and_unknown_keys(self):
        doc=model_profile_to_dict(load_qwen3_235b())
        for modified in [{**doc,'schema_version':2},{**doc,'typo':1},{**doc,'memory_family':'new-cache'},
                         {**doc,'tensors':[]}]:
            with self.assertRaises(ValueError):load_model_profile(modified)
        for key,val in [('shape',[-1,2]),('role','typo'),('placement','typo'),('layer',999),('multiplicity',0)]:
            modified=copy.deepcopy(doc);modified['tensors'][3][key]=val
            with self.assertRaises(ValueError):load_model_profile(modified)

    def test_roles_owners_and_inventory_reference(self):
        doc=model_profile_to_dict(load_qwen3_235b())
        doc['reference_weight_bytes']={'bf16':470187269120}
        load_model_profile(doc)
        bad=copy.deepcopy(doc);bad['tensors'].pop()
        with self.assertRaisesRegex(ValueError,'reference'):load_model_profile(bad)
        bad=copy.deepcopy(doc);bad['tensors'][0]['pp_owner']=None
        with self.assertRaisesRegex(ValueError,'pp_owner'):load_model_profile(bad)
        bad=copy.deepcopy(doc);bad['tensors'][0]['role']='auto'
        with self.assertRaises(ValueError):load_model_profile(bad)

    def test_model_converter_registration(self):
        original=load_qwen3_235b()
        def convert(c,**kwargs):
            self.assertEqual(c['model_type'],'test_custom_moe')
            return replace(original,model_id=kwargs.get('model_id','custom'))
        register_model_adapter('test_custom_moe',convert)
        self.assertEqual(from_config({'model_type':'test_custom_moe'},model_id='new-id').model_id,'new-id')
        with self.assertRaises(ValueError):register_model_adapter('test_custom_moe',convert)
        with self.assertRaises(ValueError):register_model_adapter('qwen3_moe',convert)

    def test_cli_resolves_all_profiles_relative_to_scenario(self):
        base=Path(__file__).resolve().parents[1]/'examples/profiles'
        doc=json.loads((base/'scenario.json').read_text())
        self.assertTrue(evaluate_document(doc,base=base)['memory_pass'])
        bad=copy.deepcopy(doc);bad['hardware']['memory_bytes']=1
        with self.assertRaises(ValueError):evaluate_document(bad,base=base)
        doc.pop('parallelism');doc['search']={'tp_values':[8],'dp_values':[1],'pp_values':[1]}
        self.assertTrue(evaluate_document(doc,base=base,mode='search')['candidates'])

    def test_runtime_profile_cannot_bypass_capabilities(self):
        r=load_runtime_profile({'schema_version':1,'kind':'runtime','runtime':{'engine':'sglang','version':'0.5.10'}})
        self.assertEqual(r.name,RuntimeProfile.sglang().name)
        r=replace(r,version='latest')
        self.assertEqual(check(load_glm51(),runtime=r).status,'unsupported')
        c=copy.deepcopy(load_glm51().config);c['kv_lora_rank']=256
        from inferfit.models import glm_profile
        self.assertEqual(check(glm_profile(c)).status,'unsupported')
