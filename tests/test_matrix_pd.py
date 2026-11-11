"""Real Qwen metadata, accelerator boundaries, and PD liveness cases."""
import json, subprocess, sys, unittest, tempfile
from pathlib import Path
from dataclasses import replace
from importlib.resources import files
from inferfit import *

class MatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.model=load_qwen3_235b()

    def test_official_qwen_weight_inventory(self):
        source=self.model.provenance['source']
        self.assertEqual(logical_weight_bytes(self.model),source['safetensors']['parameters']['BF16']*2)
        self.assertEqual(logical_weight_bytes(self.model),470187269120)
        # 22B active parameters must not replace the 235B resident inventory.
        self.assertGreater(logical_weight_bytes(self.model),235_000_000_000*1.9)

    def test_qwen_gqa_cache_oracle(self):
        self.assertEqual(kv_per_token(self.model)['total'],192512)
        self.assertEqual(kv_per_token(self.model,tp=4)['total'],48128)
        self.assertEqual(kv_per_token(self.model,tp=8)['total'],48128)
        self.assertEqual(kv_per_token(self.model,tp=4,kv_quantization='fp8')['total'],24064)

    def test_real_hardware_capacity_matrix_both_engines(self):
        # This is an analytical memory test, not measured device performance.
        for runtime in [RuntimeProfile(),RuntimeProfile.sglang()]:
            for name,small,large in [('h100-sxm-80gb',4,8),('h20-96gb',4,8),('a100-80gb',4,8),('a100-40gb',8,24)]:
                with self.subTest(engine=runtime.engine,hardware=name):
                    h=load_hardware(name)
                    tp=4 if small==4 else 8
                    a=check(self.model,hardware=h,runtime=runtime,weight_quantization='bf16',parallelism=Parallelism(tp,1,tp,small//tp,small))
                    b=check(self.model,hardware=h,runtime=runtime,weight_quantization='bf16',parallelism=Parallelism(tp,1,tp,large//tp,large))
                    self.assertEqual(a.status,'fail')
                    self.assertEqual(b.status,'pass')

    def test_a100_precision_capability(self):
        for runtime in [RuntimeProfile(),RuntimeProfile.sglang()]:
            for weight,kv in [('fp8','bf16'),('bf16','fp8')]:
                result=check(self.model,hardware=load_hardware('a100-80gb'),runtime=runtime,weight_quantization=weight,kv_quantization=kv)
                self.assertEqual(result.status,'unsupported')
                self.assertIsNone(result.memory_pass)

    def test_qwen_fp8_hopper_capacity_and_ep(self):
        for runtime in [RuntimeProfile(),RuntimeProfile.sglang()]:
            r=check(self.model,hardware=load_hardware('h100-sxm-80gb'),runtime=runtime,
                    parallelism=Parallelism(4,1,4,1,4),weight_quantization='fp8')
            self.assertTrue(r.memory_pass)
        r=check(self.model,parallelism=Parallelism(8,1,3,1,8),weight_quantization='bf16')
        self.assertEqual(r.status,'invalid')

    def test_search_hardware_matrix(self):
        for name in ['h20-96gb','h100-sxm-80gb','a100-80gb','a100-40gb']:
            r=search(self.model,hardware=load_hardware(name),weight_quantization='bf16',
                     tp_values=[4,8],dp_values=[1],pp_values=[1,2,4],max_gpus=32)
            self.assertTrue(r.candidates)
            self.assertTrue(all(x.memory_pass for x in r.candidates))

    def test_unknown_platform_has_no_execution_backend(self):
        h=Hardware(name='custom-device', platform='custom', compute_capability=None, memory_bytes=96*GiB)
        for runtime in [RuntimeProfile(),RuntimeProfile.sglang()]:
            r=check(self.model,hardware=h,runtime=runtime,weight_quantization='bf16')
            self.assertEqual(r.status,'unsupported')
            self.assertIsNone(r.memory_pass)
        with self.assertRaises(ValueError):load_hardware('a100')

    def test_deepseek_v41_official_config_fails_closed(self):
        config=json.loads(files('inferfit').joinpath('data/deepseek-v41-flash-config.json').read_text())
        self.assertEqual(config['model_type'],'deepseek_v41')
        with self.assertRaisesRegex(UnsupportedModelError,'compressed/shared cache'):
            from_config(config)

class PDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model=load_qwen3_235b()
    def pools(self,engine='vllm'):
        r=RuntimeProfile.sglang() if engine=='sglang' else RuntimeProfile()
        p=Parallelism(8,1,8,1,8)
        return (PDPool(tuple(f'p-node/gpu{i}' for i in range(8)),load_hardware('h100-sxm-80gb'),p,r),
                PDPool(tuple(f'd-node/gpu{i}' for i in range(8)),load_hardware('h20-96gb'),p,r))

    def test_heterogeneous_pd_two_independent_weight_copies(self):
        for engine in ['vllm','sglang']:
            p,d=self.pools(engine)
            r=check_pd(self.model,prefill=p,decode=d)
            self.assertTrue(r.memory_pass)
            self.assertEqual(r.summary['num_gpus_total'],16)
            self.assertGreater(r.prefill.summary['placed_weight_bytes_total'],460_000_000_000)
            self.assertGreater(r.decode.summary['placed_weight_bytes_total'],460_000_000_000)
            self.assertFalse(r.summary['connector_verified'])

    def test_source_retention_and_decode_reservation(self):
        p,d=self.pools()
        zero=check_pd(self.model,prefill=p,decode=d,demand=PDDemand(inflight_handoffs_per_dp=0))
        two=check_pd(self.model,prefill=p,decode=d,demand=PDDemand(inflight_handoffs_per_dp=2))
        self.assertGreater(two.prefill.ranks[0].kv_required_bytes,zero.prefill.ranks[0].kv_required_bytes)
        self.assertGreater(two.decode.ranks[0].kv_required_bytes,zero.decode.ranks[0].kv_required_bytes)
        self.assertEqual(two.summary['handoff_logical_payload_bytes_per_request'],8192*192512)
        self.assertEqual(two.summary['source_release'],'after_ack')

    def test_output_budget_only_expands_decode(self):
        p,d=self.pools()
        a=check_pd(self.model,prefill=p,decode=d,demand=PDDemand(output_tokens=128))
        b=check_pd(self.model,prefill=p,decode=d,demand=PDDemand(output_tokens=8192))
        self.assertEqual(a.prefill.ranks[0].kv_required_bytes,b.prefill.ranks[0].kv_required_bytes)
        self.assertGreater(b.decode.ranks[0].kv_required_bytes,a.decode.ranks[0].kv_required_bytes)

    def test_explicit_transfer_buffers_are_per_gpu(self):
        p,d=self.pools()
        a=check_pd(self.model,prefill=p,decode=d)
        b=check_pd(self.model,prefill=replace(p,transfer_buffer_bytes_per_gpu=GiB),decode=d)
        self.assertEqual(a.prefill.ranks[0].headroom_bytes-b.prefill.ranks[0].headroom_bytes,GiB)
        self.assertEqual(a.decode.ranks[0].headroom_bytes,b.decode.ranks[0].headroom_bytes)

    def test_one_pool_failure_is_pd_failure(self):
        p,d=self.pools()
        d=replace(d,hardware=load_hardware('a100-40gb'))
        r=check_pd(self.model,prefill=p,decode=d)
        self.assertTrue(r.prefill.memory_pass)
        self.assertFalse(r.decode.memory_pass)
        self.assertFalse(r.memory_pass)

    def test_overlapping_and_missing_device_ids_rejected(self):
        p,d=self.pools()
        for bad in [replace(d,gpu_ids=p.gpu_ids),replace(d,gpu_ids=('only-one',)),replace(d,gpu_ids=('same',)*8)]:
            self.assertEqual(check_pd(self.model,prefill=p,decode=bad).status,'invalid')

    def test_transfer_layout_precision_and_engine_rejected(self):
        p,d=self.pools()
        for bad in [replace(d,kv_quantization='fp8'),replace(d,parallelism=Parallelism(4,1,4,2,8)),
                    replace(d,runtime=RuntimeProfile.sglang()),replace(d,weight_quantization='fp8')]:
            self.assertEqual(check_pd(self.model,prefill=p,decode=bad).status,'unsupported')

    def test_glm_pd_with_fp8_weights(self):
        p,d=self.pools()
        plan=Parallelism(8,1,8,2,16)
        p=replace(p,gpu_ids=tuple(f'p/gpu{i}' for i in range(16)),parallelism=plan,weight_quantization='fp8')
        d=replace(d,gpu_ids=tuple(f'd/gpu{i}' for i in range(16)),parallelism=plan,weight_quantization='fp8')
        r=check_pd(load_glm51(),prefill=p,decode=d)
        self.assertTrue(r.memory_pass)
        self.assertEqual(r.summary['num_gpus_total'],32)
        self.assertEqual(r.summary['handoff_logical_payload_bytes_per_request'],8192*100152)

    def test_cli_pd_and_unsupported_model_exit_codes(self):
        root=Path(__file__).resolve().parents[1]
        done=subprocess.run([sys.executable,'-m','inferfit','pd-check',str(root/'examples/h100-h20-qwen-pd.json')],capture_output=True,text=True)
        self.assertEqual(done.returncode,0,done.stderr)
        self.assertTrue(json.loads(done.stdout)['memory_pass'])
        config=json.loads(files('inferfit').joinpath('data/deepseek-v41-flash-config.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            scenario=Path(temp)/'deepseek.json'
            scenario.write_text(json.dumps({'model':{'config':config}}))
            done=subprocess.run([sys.executable,'-m','inferfit','check',str(scenario)],capture_output=True,text=True)
            self.assertEqual(done.returncode,2)
            self.assertEqual(json.loads(done.stdout)['status'],'unsupported')

    def test_invalid_demands(self):
        p,d=self.pools()
        for demand in [PDDemand(prompt_tokens=0),PDDemand(output_tokens=-1),PDDemand(inflight_handoffs_per_dp=-1),PDDemand(output_tokens=40960)]:
            self.assertEqual(check_pd(self.model,prefill=p,decode=d,demand=demand).status,'invalid')
