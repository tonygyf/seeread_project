import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from tools.v5_research_queue import (container_record_path, owned_host_pid, run,
                                     stop_own_job, verify_container, development_gate, gpu_processes)
from tools import v5_job_runner as runner


class QueueSafetyTests(unittest.TestCase):
    def test_detection_branch_waits_for_pretraining_and_stops_on_parent_failure(self):
        from tools.v5_train_detection_branch import wait_for_predecessor
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, upstream = root / 'status.json', root / 'queue.json'
            upstream.write_text(json.dumps(dict(status='running')))

            def complete_parent(_seconds):
                report.write_text(json.dumps(dict(state=dict(status='stage_complete'))))

            with patch('tools.v5_train_detection_branch.time.sleep', side_effect=complete_parent) as sleep:
                wait_for_predecessor(report, upstream)
                sleep.assert_called_once_with(10)
            for status in ('error_abort', 'resource_abort', 'budget_stop_validation_pending'):
                report.write_text(json.dumps(dict(state=dict(status=status))))
                with self.assertRaisesRegex(RuntimeError, 'Pretraining stopped'):
                    wait_for_predecessor(report, upstream)
            report.unlink()
            upstream.write_text(json.dumps(dict(status='needs_attention')))
            with self.assertRaisesRegex(RuntimeError, 'Upstream queue ended'):
                wait_for_predecessor(report, upstream)

    def test_detection_branch_executes_only_the_target_with_the_full_recipe(self):
        from tools.v5_train_detection_branch import run as run_branch
        pipeline = Path('configs/v5/seeing_pipeline_ctw1500_pm_fp32.yaml')
        from seeread.config import load_config
        from tools.v5_detection_pilot import plain
        target = plain(load_config(pipeline.parent / 'seeing_ctw1500_pm_fp32.yaml'))
        expected = plain(load_config(pipeline.parent / 'seeing_ctw1500.yaml'))
        expected['stage']['init_checkpoint'] = 'results/v5/detection/MLT17_pm_fp32/last.pt'
        expected['train']['out_dir'] = 'results/v5/detection/CTW1500_pm_fp32'
        self.assertEqual(target, expected)
        with patch('tools.v5_train_detection_branch.wait_for_predecessor') as waiting, \
             patch('tools.v5_train_detection_branch.execute', return_value=dict(status='stage_complete')) as execute:
            run_branch(pipeline, Path('upstream.json'))
        waiting.assert_called_once_with(Path('results/v5/detection/MLT17_pm_fp32/status.json'), Path('upstream.json'))
        execute.assert_called_once_with(target, reduced=False)

    def test_long_training_gate_uses_complete_dev_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / 'candidate.json', root / 'reference.json'
            manifest = {'dev': {'sha256': 'same'}}
            a.write_text(json.dumps(dict(manifests=manifest, validations=[
                dict(complete=False, raw={'cer': .1}), dict(complete=True, raw={'cer': .6})])))
            b.write_text(json.dumps(dict(manifests=manifest, result=dict(complete=True, raw={'cer': .7}))))
            gate = dict(candidate_report=str(a), reference_report=str(b), minimum_cer_improvement=.2)
            self.assertFalse(development_gate(gate)['passed'])
            gate['minimum_cer_improvement'] = .01
            self.assertTrue(development_gate(gate)['passed'])

    def container(self):
        return [{'HostConfig': dict(DeviceRequests=[dict(DeviceIDs=['2'])], Memory=24*1024**3,
                                    MemorySwap=24*1024**3, NanoCpus=8*10**9, PidsLimit=512),
                 'State': {'Running': True}}]

    def test_other_gpu_or_missing_memory_cap_is_rejected(self):
        good = self.container()
        with patch('tools.v5_research_queue.command', return_value=json.dumps(good)):
            verify_container('ours')
        for wrong in ('gpu', 'memory'):
            value = self.container()
            if wrong == 'gpu':
                value[0]['HostConfig']['DeviceRequests'][0]['DeviceIDs'] = ['2', '3']
            else:
                value[0]['HostConfig']['Memory'] = 0
            with patch('tools.v5_research_queue.command', return_value=json.dumps(value)):
                with self.assertRaises(ValueError):
                    verify_container('ours')

    def test_second_authorized_gpu_is_explicit_and_container_must_match(self):
        gpu3 = self.container()
        gpu3[0]['HostConfig']['DeviceRequests'][0]['DeviceIDs'] = ['3']
        with patch('tools.v5_research_queue.command', return_value=json.dumps(gpu3)):
            verify_container('ours', gpu=3)
            with self.assertRaises(ValueError):
                verify_container('ours', gpu=2)
        with self.assertRaises(ValueError):
            verify_container('ours', gpu=1)
        with patch('tools.v5_research_queue.command', return_value='123\n456') as command:
            self.assertEqual(gpu_processes(3), {123, 456})
            self.assertEqual(command.call_args.args[0][2], '3')

    def test_interrupt_uses_unique_record_and_token_not_command_line_matching(self):
        with patch('tools.v5_research_queue.command', return_value='{"signaled":true,"container_pid":17}') as command:
            self.assertEqual(stop_own_job('ours', '/workspace/owner.json', 'unique')['container_pid'], 17)
            command.assert_called_once_with(['docker', 'exec', 'ours', 'python', 'tools/v5_job_runner.py',
                                             'stop', '--record', '/workspace/owner.json', '--token', 'unique'])

    def record(self):
        return dict(format=runner.FORMAT, token='our-unique-token', container_pid=17,
                    start_ticks=12345, argv=['python', 'tools/v5_reading.py', '--config', 'ours.yaml'])

    def identity(self):
        record = self.record()
        return dict(pid=17, namespace_pids=[1017, 17], start_ticks=record['start_ticks'],
                    argv=record['argv'], token=record['token'])

    def save_record(self, root):
        path = root / 'owner.json'
        path.write_text(json.dumps(self.record()))
        path.chmod(0o600)
        return path

    def test_host_mapping_rejects_same_command_with_other_token_namespace_or_start(self):
        expected = self.record()
        good = self.identity()
        identities = {1017: dict(good, pid=1017),
                      1018: dict(good, pid=1018, namespace_pids=[1018, 18], token='other'),
                      1019: dict(good, pid=1019, start_ticks=999),
                      1020: dict(good, pid=1020, namespace_pids=[])}
        with patch('tools.v5_research_queue.read_record', return_value=expected), \
             patch('tools.v5_research_queue.host_container_pids', return_value=set(identities)), \
             patch('tools.v5_research_queue.process_identity', side_effect=lambda pid: identities[pid]):
            self.assertEqual(owned_host_pid('ours', 'owner.json', expected['token']), 1017)

    def test_pidfd_signal_pins_only_the_verified_owned_process(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.save_record(Path(directory))
            with patch.object(runner.os, 'pidfd_open', return_value=99) as opened, \
                 patch.object(runner.signal, 'pidfd_send_signal') as sent, \
                 patch.object(runner, 'process_identity', return_value=self.identity()), \
                 patch.object(runner.os, 'close') as closed, patch.object(runner.os, 'kill') as unsafe:
                self.assertTrue(runner.stop_job(path, self.record()['token'])['signaled'])
                opened.assert_called_once_with(17, 0)
                sent.assert_called_once_with(99, signal.SIGINT, None, 0)
                closed.assert_called_once_with(99)
                unsafe.assert_not_called()

    def test_pid_reuse_other_command_and_other_environment_never_receive_a_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.save_record(Path(directory))
            for changed in (dict(start_ticks=999), dict(argv=['python', 'unrelated.py']), dict(token='other')):
                with patch.object(runner.os, 'pidfd_open', return_value=99), \
                     patch.object(runner.signal, 'pidfd_send_signal') as sent, \
                     patch.object(runner, 'process_identity', return_value=dict(self.identity(), **changed)), \
                     patch.object(runner.os, 'close') as closed, patch.object(runner.os, 'kill') as unsafe:
                    with self.assertRaisesRegex(ValueError, 'identity changed'):
                        runner.stop_job(path, self.record()['token'])
                    sent.assert_not_called(); unsafe.assert_not_called()
                    closed.assert_called_once_with(99)

    def test_explicit_host_adapter_verifies_namespace_before_pidfd_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.save_record(Path(directory))
            identity = dict(self.identity(), pid=1017)
            with patch.object(runner.os, 'pidfd_open', return_value=99) as opened, \
                 patch.object(runner.signal, 'pidfd_send_signal') as sent, \
                 patch.object(runner, 'process_identity', return_value=identity), \
                 patch.object(runner.os, 'close'), patch.object(runner.os, 'kill') as unsafe:
                self.assertTrue(runner.stop_job(path, self.record()['token'], host_pid=1017)['signaled'])
                opened.assert_called_once_with(1017, 0)
                sent.assert_called_once_with(99, signal.SIGINT, None, 0)
                unsafe.assert_not_called()
                identity['namespace_pids'] = [1017, 18]
                sent.reset_mock()
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    runner.stop_job(path, self.record()['token'], host_pid=1017)
                sent.assert_not_called()

    def test_absent_pidfd_support_wrong_record_token_and_insecure_record_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.save_record(Path(directory))
            with patch.object(runner.os, 'pidfd_open', None), \
                 patch.object(runner.signal, 'pidfd_send_signal') as sent, patch.object(runner.os, 'kill') as unsafe:
                with self.assertRaisesRegex(RuntimeError, 'pidfd'):
                    runner.stop_job(path, self.record()['token'])
                sent.assert_not_called(); unsafe.assert_not_called()
            with patch.object(runner.os, 'pidfd_open') as opened:
                with self.assertRaisesRegex(ValueError, 'token'):
                    runner.stop_job(path, 'other-token')
                opened.assert_not_called()
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, '0600'):
                runner.read_record(path, self.record()['token'])

    def test_runner_start_publishes_exclusive_record_and_execs_same_pid_with_unique_token(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'owner.json'
            expected = self.record()
            with patch.object(runner.os, 'getpid', return_value=17), \
                 patch.object(runner, 'process_identity', return_value=self.identity()), \
                 patch.object(runner.os, 'execvpe', side_effect=RuntimeError('intercepted exec')) as execute:
                with self.assertRaisesRegex(RuntimeError, 'intercepted exec'):
                    runner.start_job(path, expected['token'], expected['argv'])
                record = runner.read_record(path, expected['token'])
                self.assertEqual(record, expected)
                self.assertEqual(execute.call_args.args[2][runner.TOKEN_ENV], expected['token'])
                self.assertEqual(execute.call_args.args[:2], ('python', expected['argv']))
                with self.assertRaises(FileExistsError):
                    runner.start_job(path, 'another', expected['argv'])
                self.assertEqual(execute.call_count, 1)

    def test_runner_cli_preserves_the_explicit_child_argv_after_separator(self):
        argv = ['runner', 'start', '--record', '/workspace/owner.json', '--token', 'unique',
                '--', 'python', 'tools/v5_reading.py', '--config', 'ours.yaml', '--mode', 'train']
        with patch('sys.argv', argv), patch.object(runner, 'start_job') as launch:
            runner.main()
        launch.assert_called_once_with(Path('/workspace/owner.json'), 'unique',
                                       ['python', 'tools/v5_reading.py', '--config', 'ours.yaml', '--mode', 'train'])

    def test_proc_parser_handles_parentheses_and_reads_namespace_argv_and_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); process = root / '1017'; process.mkdir()
            fields = ['S'] + ['0'] * 18 + ['12345']
            (process / 'stat').write_text('1017 (python ) tricky) ' + ' '.join(fields))
            (process / 'cmdline').write_bytes(b'python\0tools/v5_reading.py\0')
            (process / 'environ').write_bytes(b'OTHER=value\0SEEREAD_V5_JOB_TOKEN=unique\0')
            (process / 'status').write_text('Name: python\nNSpid:\t1017\t17\n')
            actual = runner.process_identity(1017, root)
            self.assertEqual(actual['start_ticks'], 12345)
            self.assertEqual(actual['namespace_pids'], [1017, 17])
            self.assertEqual(actual['argv'], ['python', 'tools/v5_reading.py'])
            self.assertEqual(actual['token'], 'unique')

    def test_gpu3_queue_treats_other_container_process_as_external(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = self.container()[0]
            info['HostConfig']['DeviceRequests'][0]['DeviceIDs'] = ['3']
            info['Mounts'] = [dict(Source=str(root), Destination='/workspace', RW=True)]
            plan = root / 'plan.json'; state = root / 'state.json'
            plan.write_text(json.dumps(dict(container='ours', gpu=3, jobs=[dict(
                id='one', argv=['python', 'tools/v5_reading.py'], log=str(root / 'job.log'))])))
            class Process:
                pid = 555
                def poll(self): return None
                def wait(self, timeout=None): return 0
            with patch('tools.v5_research_queue.verify_container', return_value=info), \
                 patch('tools.v5_research_queue.gpu_processes', side_effect=[set(), {1017, 1018}]) as gpu_query, \
                 patch('tools.v5_research_queue.owned_host_pid', return_value=1017), \
                 patch('tools.v5_research_queue.subprocess.Popen', return_value=Process()) as spawn, \
                 patch('tools.v5_research_queue.time.sleep'), \
                 patch('tools.v5_research_queue.stop_own_job', return_value=dict(signaled=True, container_pid=17)) as stop:
                self.assertEqual(run(plan, state), 1)
                result = json.loads(state.read_text())
                self.assertEqual(result['jobs'][0]['external_pids'], [1018])
                self.assertEqual(result['jobs'][0]['owned_host_pid'], 1017)
                self.assertEqual(result['gpu'], 3)
                self.assertEqual([c.args for c in gpu_query.call_args_list], [(3,), (3,)])
                self.assertIn('tools/v5_job_runner.py', spawn.call_args.args[0])
                stop.assert_called_once()
                self.assertEqual(stop.call_args.args[2], result['jobs'][0]['owner_token'])
            with self.assertRaisesRegex(ValueError, 'shared mount'):
                container_record_path('/outside/project', info)

    def test_stale_gpu_pid_after_clean_exit_does_not_turn_completion_into_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = self.container()[0]
            info['Mounts'] = [dict(Source=str(root), Destination='/workspace', RW=True)]
            plan, state, report = root / 'plan.json', root / 'state.json', root / 'report.json'
            report.write_text(json.dumps(dict(status='complete')))
            plan.write_text(json.dumps(dict(container='ours', gpu=2, jobs=[dict(
                id='one', argv=['python', 'tools/v5_reading.py'], log=str(root / 'job.log'),
                report=str(report))])))

            class ExitingProcess:
                pid = 555
                def poll(self): return None  # Docker exec has not yet reaped its cleanly exited child.
                def wait(self, timeout=None): return 0

            with patch('tools.v5_research_queue.verify_container', return_value=info), \
                 patch('tools.v5_research_queue.gpu_processes', side_effect=[set(), {1017}]), \
                 patch('tools.v5_research_queue.owned_host_pid', return_value=None), \
                 patch('tools.v5_research_queue.subprocess.Popen', return_value=ExitingProcess()), \
                 patch('tools.v5_research_queue.time.sleep'), \
                 patch('tools.v5_research_queue.stop_own_job', return_value=dict(signaled=False, reason='already_exited')):
                self.assertEqual(run(plan, state), 0)
            result = json.loads(state.read_text())
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['jobs'][0]['report_status'], 'complete')
            self.assertTrue(result['jobs'][0]['exit_race_reconciled'])


if __name__ == '__main__':
    unittest.main()
