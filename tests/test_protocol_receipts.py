#!/usr/bin/env python3
"""P5 and P3's post-lag re-audit leave evidence, and the run stops without it.

Three of the five protocols are enforced by the base class: they sit inside
concrete methods the setup skeleton calls, so a paradigm cannot reach the
models without passing through them. The other two cannot be, because they need
the materialised fold -- the one thing the three paradigms exist to build
differently. They run inside each paradigm's model code, and what the core
guaranteed about them was that their author had remembered to call them.

The README said the anti-leakage checks were inherited. For these two it was
not true, and the substitute was a test that parsed the plugin's source looking
for a local variable named X_train: a contract satisfied by naming, not by
behaviour. A fourth paradigm that skipped either call would have produced a
complete set of results under a protocol it never ran.

These tests pin the gate that closes it from the other side -- not by moving
the calls into the core, which would mean the core materialising the fold and
erasing the difference the experiment measures, but by requiring the receipt
each call leaves behind.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'src'))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.paradigm_registry import discover_paradigms  # noqa: E402
from core.validation import AntiLeakageViolation  # noqa: E402

PARADIGMS = sorted(discover_paradigms())


RUN_ID = 'a1b2c3d4e5f60718293a4b5c6d7e8f90'


def _imputation(created: datetime) -> str:
    return json.dumps({
        'architecture': 'x',
        'creation_timestamp': created.isoformat(),
        'run_id': RUN_ID,
        'folds': {'0': {'filled_cells': {'train': {'rows': 10, 'total': 1}}}},
        'across_folds': {'train': {'rows': 10, 'total': 1, 'fraction': 0.1}},
    })


def _audit(created: datetime) -> str:
    return json.dumps({
        'architecture': 'x',
        'creation_timestamp': created.isoformat(),
        'run_id': RUN_ID,
        'features_audited': ['gdp_per_capita', 'dropout_rate_lag_2'],
        'joint_reconstruction_r2': 0.31,
    })


class TestTheReceiptGate:

    @pytest.fixture
    def gate(self, tmp_path, monkeypatch):
        import pipeline

        outputs = tmp_path / 'outputs'
        monkeypatch.setattr(
            pipeline, 'get_absolute_output_path',
            lambda relative: str(outputs / relative.replace('outputs/', '')))
        return pipeline, outputs, RUN_ID, datetime.now()

    @staticmethod
    def _write(outputs, created, *, skip=(), payloads=None):
        """The receipts a completed run leaves, minus whatever `skip` names.

        Existing receipts are removed first: a caller that writes twice into
        one tmp_path would otherwise be handed the previous call's files, and a
        test that skips a receipt would silently find it still there.
        """
        payloads = payloads or {}
        for paradigm in PARADIGMS:
            prep = (outputs / 'ml_pipeline' / 'architectures' / paradigm
                    / 'prep')
            prep.mkdir(parents=True, exist_ok=True)
            for stem, default in (('fold_imputation', _imputation),
                                  ('feature_audit', _audit)):
                stale = prep / f'{stem}_{paradigm}.json'
                if stale.exists():
                    stale.unlink()
                if (paradigm, stem) in skip:
                    continue
                body = payloads.get((paradigm, stem))
                path = prep / f'{stem}_{paradigm}.json'
                path.write_text(json.dumps(body) if body is not None
                                else default(created))

    def test_complete_receipts_pass(self, gate):
        pipeline, outputs, run_id, created = gate
        self._write(outputs, created)
        pipeline._validate_protocol_receipts(run_id)

    def test_a_missing_p5_receipt_halts(self, gate):
        pipeline, outputs, run_id, created = gate
        self._write(outputs, created,
                    skip={(PARADIGMS[0], 'fold_imputation')})
        with pytest.raises(AntiLeakageViolation, match='P5'):
            pipeline._validate_protocol_receipts(run_id)

    def test_a_missing_p3_receipt_halts(self, gate):
        """The audit writer is conditional, so its absence is the real signal.

        A paradigm that never calls the re-audit leaves feature_audit unset,
        and the writer is skipped -- no file, and previously no complaint.
        """
        pipeline, outputs, run_id, created = gate
        self._write(outputs, created, skip={(PARADIGMS[0], 'feature_audit')})
        with pytest.raises(AntiLeakageViolation, match='P3'):
            pipeline._validate_protocol_receipts(run_id)

    def test_an_empty_imputation_receipt_halts(self, gate):
        """The P5 writer runs unconditionally, so the file proves nothing.

        A paradigm that skips impute_from_training_window still reaches the
        writer with an empty list of reports, and a gate that only checked for
        the file would have accepted the result.
        """
        pipeline, outputs, run_id, created = gate
        self._write(outputs, created, payloads={
            (PARADIGMS[-1], 'fold_imputation'): {
                'creation_timestamp': created.isoformat(),
                'folds': {}, 'across_folds': {}}})
        with pytest.raises(AntiLeakageViolation, match='P5'):
            pipeline._validate_protocol_receipts(run_id)

    def test_an_empty_audit_receipt_halts(self, gate):
        pipeline, outputs, run_id, created = gate
        self._write(outputs, created, payloads={
            (PARADIGMS[0], 'feature_audit'): {
                'creation_timestamp': created.isoformat(),
                'features_audited': []}})
        with pytest.raises(AntiLeakageViolation, match='P3'):
            pipeline._validate_protocol_receipts(run_id)

    def test_an_unstamped_receipt_halts(self, gate):
        pipeline, outputs, run_id, created = gate
        self._write(outputs, created, payloads={
            (PARADIGMS[0], 'fold_imputation'): {'folds': {'0': {}}}})
        with pytest.raises(AntiLeakageViolation, match='run_id'):
            pipeline._validate_protocol_receipts(run_id)

    def test_a_receipt_from_another_run_halts(self, gate):
        """Otherwise the gate attests to a protocol some other run followed.

        Carried by a nonce rather than by a clock comparison. The receipt here
        is *newer* than the run -- which a timestamp test would have accepted,
        and which is exactly the shape a concurrent second run leaves behind.
        """
        pipeline, outputs, run_id, created = gate
        self._write(outputs, created, payloads={
            (PARADIGMS[0], 'feature_audit'): {
                'creation_timestamp': (created + timedelta(hours=3)).isoformat(),
                'run_id': 'ffffffffffffffffffffffffffffffff',
                'features_audited': ['gdp_per_capita']}})
        with pytest.raises(AntiLeakageViolation, match='de outra corrida'):
            pipeline._validate_protocol_receipts(run_id)

    def test_a_backward_clock_step_does_not_abort(self, gate):
        """The reason the gate stopped comparing wall clocks.

        Its window is the whole of stages 1-5 -- hours on the INEP panel -- and
        a backward NTP correction inside it would make a fresh receipt look
        stale and kill the run.
        """
        pipeline, outputs, run_id, created = gate
        self._write(outputs, created, payloads={
            (PARADIGMS[0], 'feature_audit'): {
                'creation_timestamp': (created - timedelta(hours=4)).isoformat(),
                'run_id': run_id,
                'features_audited': ['gdp_per_capita']}})
        pipeline._validate_protocol_receipts(run_id)

    def test_every_paradigm_is_checked(self, gate):
        """Not only the first: the gate exists for the paradigm added later."""
        pipeline, outputs, run_id, created = gate
        last = PARADIGMS[-1]
        self._write(outputs, created, skip={(last, 'fold_imputation')})
        with pytest.raises(AntiLeakageViolation, match=last):
            pipeline._validate_protocol_receipts(run_id)

    def test_it_covers_both_protocols_for_every_paradigm(self, gate):
        """Each paradigm accounts for both, so the count is 2N, not N + 2."""
        pipeline, outputs, run_id, created = gate
        for paradigm in PARADIGMS:
            for stem, protocol in (('fold_imputation', 'P5'),
                                   ('feature_audit', 'P3')):
                self._write(outputs, created, skip={(paradigm, stem)})
                with pytest.raises(AntiLeakageViolation) as caught:
                    pipeline._validate_protocol_receipts(run_id)
                assert paradigm in str(caught.value)
                assert protocol in str(caught.value)


class TestTheWritersStampTheirReceipts:
    """Without a stamp of its own the gate could only read the file's mtime.

    Which is a property of the filesystem rather than of the artifact, and
    survives neither a copy nor a restore from an archive.
    """

    @pytest.fixture
    def prep(self, tmp_path, monkeypatch):
        from core import config
        monkeypatch.setattr(
            config, 'get_absolute_output_path',
            lambda relative: str(tmp_path / relative))
        # pipeline.main exports this before any model runs; the writers read it
        # from the environment because they are three subprocesses below it.
        monkeypatch.setenv('RAMPART_RUN_ID', RUN_ID)
        return tmp_path

    def test_the_imputation_report_is_stamped(self, prep):
        from core.models.hierarchical import write_imputation_report

        path = write_imputation_report(
            [(0, {'filled_cells': {'train': {'rows': 8, 'total': 2}}})],
            architecture='sql_engine')
        payload = json.loads(Path(path).read_text())
        assert datetime.fromisoformat(payload['creation_timestamp'])
        assert payload['run_id'] == RUN_ID, 'the run nonce did not reach disk'
        assert payload['folds'], 'the evidence field the gate reads is empty'

    def test_the_feature_audit_is_stamped(self, prep):
        from core.models.hierarchical import write_feature_audit

        path = write_feature_audit({'features_audited': ['gdp_per_capita']},
                                   architecture='sql_engine')
        payload = json.loads(Path(path).read_text())
        assert datetime.fromisoformat(payload['creation_timestamp'])
        assert payload['run_id'] == RUN_ID, 'the run nonce did not reach disk'
        assert payload['features_audited']

    def test_the_stamp_does_not_displace_the_report(self, prep):
        """The report is splatted in, so a key collision would be silent."""
        from core.models.hierarchical import write_feature_audit

        report = {'features_audited': ['a'], 'joint_reconstruction_r2': 0.4,
                  'design_rank': 3}
        path = write_feature_audit(report, architecture='task_graph')
        payload = json.loads(Path(path).read_text())
        for key, value in report.items():
            assert payload[key] == value


class TestTheGateRunsWhereTheReceiptsExist:
    """Placed after the models, because they are what writes the receipts.

    Installed next to the temporal gate -- which runs right after setup -- it
    would have checked for files that no stage had yet produced, and failed
    every run.
    """

    SOURCE = (_ROOT / 'pipeline.py').read_text()

    def _at(self, needle: str) -> int:
        index = self.SOURCE.find(needle)
        assert index > 0, f'{needle!r} is gone from pipeline.py'
        return index

    def test_it_runs_after_the_hierarchical_stage(self):
        assert (self._at('_validate_protocol_receipts(run_id)')
                > self._at('info["hierarchical_script"]'))

    def test_it_runs_before_the_equivalence_gate(self):
        assert (self._at('_validate_protocol_receipts(run_id)')
                < self._at('prediction_equivalence.py'))

    def test_it_runs_before_the_benchmark(self):
        """A latency comparison of paradigms under different protocols."""
        assert (self._at('_validate_protocol_receipts(run_id)')
                < self._at('architectural_benchmark.py'))

    def test_the_nonce_is_exported_before_any_stage_runs(self):
        """The writers are three subprocesses down; they read it from the env.

        Caught by mutation: removing the export left every test green, and the
        cost surfaced only as a run that aborts hours in, blaming the paradigm
        for a receipt the orchestrator never let it stamp.
        """
        assert (self._at("os.environ['RAMPART_RUN_ID'] = run_id")
                < self._at('run([py, os.path.join(root, "src/collection'))

    def test_the_gate_is_given_the_exported_nonce(self):
        """Not a fresh uuid4, which would never match any receipt."""
        assert self._at('run_id = uuid4().hex') > 0
        assert '_validate_protocol_receipts(run_id)' in self.SOURCE
        assert '_validate_protocol_receipts(uuid4' not in self.SOURCE

    def test_it_halts_rather_than_warns(self):
        """AntiLeakageViolation, so run_setup's handler cannot demote it."""
        body = self.SOURCE[self._at('def _validate_protocol_receipts'):
                           self._at('def _prediction_digests')]
        assert 'AntiLeakageViolation' in body
        assert body.count('raise ') >= 4
        assert 'warn' not in body.lower()
