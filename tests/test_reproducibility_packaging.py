#!/usr/bin/env python3
"""The reproduction path is a committed artifact, not tacit knowledge.

Reproducing required knowing a sequence written down nowhere: install from the
lock, check the core budget fits, run the pipeline for the right dataset, then the
analysis stages. The script and the image make that sequence checkable.

Data snapshots exist for a separate reason. Collection reads an external API whose
values are revised, so a re-collected run does not reproduce an earlier one, and
the difference is indistinguishable from a code change. Hashing the input separates
the two, and allows running without network.
"""

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'scripts'))

REPRODUCE = _ROOT / 'scripts' / 'reproduce.sh'
VERIFIER = _ROOT / 'scripts' / 'verify_data_snapshot.py'
DOCKERFILE = _ROOT / 'Dockerfile'


class TestReproduceScript:

    def test_exists_and_is_executable(self):
        assert REPRODUCE.exists()
        assert REPRODUCE.stat().st_mode & 0o111, 'not executable'

    def test_fails_on_error_and_on_unset_variables(self):
        body = REPRODUCE.read_text()
        assert re.search(r'set -euo pipefail', body), (
            'without this a failing stage is followed by the next one'
        )

    def test_installs_from_the_lock_not_the_range(self):
        body = REPRODUCE.read_text()
        assert 'requirements-lock.txt' in body
        assert 'requirements.txt' not in body.replace('requirements-lock.txt', '')

    def test_checks_the_core_budget_before_running(self):
        body = REPRODUCE.read_text()
        assert body.index('engine_threads') < body.index('pipeline.py'), (
            'the budget must fail before hours of compute, not after'
        )

    def test_runs_the_pipeline_for_the_chosen_dataset(self):
        body = REPRODUCE.read_text()
        assert 'pipeline.py --dataset "$DATASET"' in body

    def test_both_datasets_are_accepted(self):
        body = REPRODUCE.read_text()
        assert 'inep_censo' in body and 'worldbank' in body

    def test_the_cost_asymmetry_is_stated(self):
        """Twenty-nine hours against ninety minutes is not a detail."""
        body = REPRODUCE.read_text()
        assert '29' in body or 'vinte e nove' in body

    def test_it_is_valid_bash(self):
        result = subprocess.run(['bash', '-n', str(REPRODUCE)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    def test_help_does_not_execute_anything(self):
        result = subprocess.run(['bash', str(REPRODUCE), '--help'],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0
        assert 'pip' not in result.stdout.lower() or 'Uso' in result.stdout


class TestSnapshotVerifier:

    @pytest.fixture
    def snapshot(self, tmp_path):
        root = tmp_path / 'snap'
        (root / 'raw_data').mkdir(parents=True)
        (root / 'raw_data' / 'x.csv').write_text('a,b\n1,2\n')
        (root / 'raw_data' / 'meta.json').write_text('{"k": 1}')
        return root

    def _run(self, *args):
        import verify_data_snapshot as verifier
        return verifier.main([str(a) for a in args])

    def test_record_then_verify_passes(self, snapshot):
        assert self._run('--snapshot', snapshot, '--dataset', 'worldbank',
                         '--record') == 0
        assert self._run('--snapshot', snapshot, '--dataset', 'worldbank') == 0

    def test_a_changed_byte_is_detected(self, snapshot):
        self._run('--snapshot', snapshot, '--dataset', 'worldbank', '--record')
        (snapshot / 'raw_data' / 'x.csv').write_text('a,b\n1,3\n')
        with pytest.raises(ValueError, match='changed='):
            self._run('--snapshot', snapshot, '--dataset', 'worldbank')

    def test_an_extra_file_is_detected(self, snapshot):
        self._run('--snapshot', snapshot, '--dataset', 'worldbank', '--record')
        (snapshot / 'raw_data' / 'y.csv').write_text('c\n9\n')
        with pytest.raises(ValueError, match='extra='):
            self._run('--snapshot', snapshot, '--dataset', 'worldbank')

    def test_a_removed_file_is_detected(self, snapshot):
        self._run('--snapshot', snapshot, '--dataset', 'worldbank', '--record')
        (snapshot / 'raw_data' / 'meta.json').unlink()
        with pytest.raises(ValueError, match='missing='):
            self._run('--snapshot', snapshot, '--dataset', 'worldbank')

    def test_a_renamed_file_is_detected(self):
        """Hashing content alone would miss it; the inventory digest does not."""
        import verify_data_snapshot as verifier
        before = {'a.csv': 'deadbeef'}
        after = {'b.csv': 'deadbeef'}
        assert hashlib.sha256(json.dumps(before, sort_keys=True).encode()) \
            .hexdigest() != hashlib.sha256(
                json.dumps(after, sort_keys=True).encode()).hexdigest()

    def test_the_wrong_dataset_is_refused(self, snapshot):
        self._run('--snapshot', snapshot, '--dataset', 'worldbank', '--record')
        with pytest.raises(ValueError, match='the panels are not interchangeable'):
            self._run('--snapshot', snapshot, '--dataset', 'inep_censo')

    def test_a_snapshot_without_a_manifest_is_refused(self, snapshot):
        with pytest.raises(FileNotFoundError, match='Snapshot without a manifest'):
            self._run('--snapshot', snapshot, '--dataset', 'worldbank')

    def test_the_manifest_is_excluded_from_its_own_inventory(self, snapshot):
        import verify_data_snapshot as verifier
        self._run('--snapshot', snapshot, '--dataset', 'worldbank', '--record')
        manifest = json.loads(
            (snapshot / verifier.MANIFEST_NAME).read_text())
        assert verifier.MANIFEST_NAME not in manifest['files']
        assert manifest['file_count'] == 2


class TestSnapshotSurvivesItsOwnAge:
    """A verified snapshot is authoritative, whatever its age.

    copytree preserves mtime, and the collector invalidated the cache by age, so
    a thirty-day-old snapshot triggered a call to the API -- exactly what it
    exists to avoid. Being old is its defining characteristic.
    """

    def test_the_installer_leaves_the_manifest(self):
        source = VERIFIER.read_text()
        assert 'ignore_patterns(MANIFEST_NAME)' not in source, (
            'without the manifest at the destination the collector does not '
            'know the data comes from a verified snapshot'
        )

    def test_the_collector_gates_on_the_manifest(self):
        source = (_ROOT / 'src' / 'collection'
                  / 'raw_data_collector.py').read_text()
        block = source[source.index('def _cache_is_valid'):]
        block = block[:block.index('\n    def ', 1)]
        assert 'snapshot_manifest.json' in block
        assert block.index('snapshot_manifest.json') < block.index('age_hours')

    def test_an_old_snapshot_is_still_accepted(self, tmp_path, monkeypatch):
        """Reproduced: files thirty days old, manifest present.

        The manifest goes at the root of collection/, which is where the
        installer puts it. This test used to drop it beside the data files in
        output_dir and pass, which is how the two halves stayed out of step:
        the fixture agreed with the collector, and neither agreed with the
        installer.
        """
        import os
        import time

        monkeypatch.setenv('DATASET_NAME', 'worldbank')
        monkeypatch.setattr('core.config.get_outputs_root',
                            lambda: str(tmp_path / 'worldbank'))

        import collection.raw_data_collector as collector
        from core.config import get_absolute_output_path

        raw = Path(get_absolute_output_path('collection/raw_data'))
        raw.mkdir(parents=True)

        instance = object.__new__(
            next(getattr(collector, name) for name in dir(collector)
                 if isinstance(getattr(collector, name), type)
                 and hasattr(getattr(collector, name), '_cache_is_valid')))
        instance.output_dir = str(raw)
        old = time.time() - 30 * 24 * 3600
        for name in ('complete_data.parquet', 'raw_data_long.parquet',
                     'scientific_collection_metadata.json',
                     'scientific_imputation_log.json'):
            path = raw / name
            path.write_text('{}')
            os.utime(path, (old, old))

        assert not instance._cache_is_valid(), (
            'without a manifest it should expire'
        )
        (Path(get_absolute_output_path('collection'))
         / 'snapshot_manifest.json').write_text('{}')
        assert instance._cache_is_valid(), (
            'with a manifest the verified snapshot should be accepted'
        )


class TestDockerfile:

    def test_exists(self):
        assert DOCKERFILE.exists()

    def test_base_image_is_pinned_by_digest(self):
        """A tag resolves to different images over time."""
        body = DOCKERFILE.read_text()
        match = re.search(r'^FROM \S+@sha256:([0-9a-f]{64})$', body,
                          re.MULTILINE)
        assert match, 'the base image is not pinned by digest'

    def test_dependencies_come_from_the_lock(self):
        body = DOCKERFILE.read_text()
        assert 'requirements-lock.txt' in body

    def test_thread_budget_is_set_in_the_image(self):
        body = DOCKERFILE.read_text()
        for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'MKL_NUM_THREADS', 'PYTHONHASHSEED'):
            assert variable in body, variable

    def test_thread_budget_matches_the_configuration(self):
        from core.scientific_config import SCIENTIFIC_CONFIG
        body = DOCKERFILE.read_text()
        assert f"OMP_NUM_THREADS={SCIENTIFIC_CONFIG['blas_threads']}" in body
        assert f"PYTHONHASHSEED={SCIENTIFIC_CONFIG['random_seed']}" in body

    def test_the_suite_runs_during_the_build(self):
        body = DOCKERFILE.read_text()
        assert 'pytest' in body, (
            'an image that fails its tests should not come into existence'
        )

    def test_the_dockerfile_is_in_the_image(self):
        """The suite runs during the build and inspects this file.

        Without copying it, eight tests fail inside the build and no image is
        produced -- so the path documented in the README did not work.
        """
        body = DOCKERFILE.read_text()
        copied = [line for line in body.splitlines()
                  if line.startswith('COPY') and 'Dockerfile' in line]
        assert copied, 'the image does not contain the Dockerfile itself'

    def test_it_is_copied_before_the_suite_runs(self):
        body = DOCKERFILE.read_text()
        copy_line = next(i for i, line in enumerate(body.splitlines())
                         if line.startswith('COPY') and 'Dockerfile' in line)
        test_line = next(i for i, line in enumerate(body.splitlines())
                         if 'pytest' in line)
        assert copy_line < test_line

    def test_the_lock_is_copied_before_the_source(self):
        """Otherwise every code change reinstalls every dependency."""
        body = DOCKERFILE.read_text()
        assert body.index('requirements-lock.txt') < body.index('COPY src/')

    def test_no_dataset_is_baked_into_the_entrypoint(self):
        """The two differ by roughly twenty times in cost."""
        body = DOCKERFILE.read_text()
        assert '--dataset inep_censo' not in body


class TestTheSnapshotSeamHolds:
    """The installer's directory and the collector's directory are the same one.

    Both halves were tested and both passed. `install` was checked for leaving
    the manifest behind; `_cache_is_valid` was checked by pointing output_dir at
    a tmp_path and writing a manifest into it. Neither asked whether the path
    one writes is the path the other reads, and it was not: the installer put
    the manifest at the root of `collection/`, the collector looked for it in
    `collection/raw_data/`.

    The consequence is the failure the collector's own comment describes and
    claims to have fixed -- a snapshot older than a day looking like an expired
    cache and reaching for the network. An offline, hash-verified run silently
    became an online one.
    """

    def test_the_installer_writes_where_the_collector_reads(self, tmp_path,
                                                             monkeypatch):
        import importlib
        import json as _json

        monkeypatch.setenv('DATASET_NAME', 'worldbank')
        outputs = tmp_path / 'outputs'
        monkeypatch.setattr('core.config.get_outputs_root',
                            lambda: str(outputs / 'worldbank'))

        # A snapshot shaped as the installer expects: the collection tree, with
        # the manifest at its root.
        snapshot = tmp_path / 'snap'
        (snapshot / 'raw_data').mkdir(parents=True)
        for name in ('complete_data.parquet', 'raw_data_long.parquet',
                     'scientific_collection_metadata.json',
                     'scientific_imputation_log.json'):
            (snapshot / 'raw_data' / name).write_text('{}')
        (snapshot / 'snapshot_manifest.json').write_text(_json.dumps({}))

        verifier = importlib.import_module('verify_data_snapshot')
        destination = verifier.install(snapshot, 'worldbank')

        from core.config import get_absolute_output_path
        collector_looks_at = Path(
            get_absolute_output_path('collection')) / 'snapshot_manifest.json'
        assert collector_looks_at.exists(), (
            f'installed at {destination}, collector reads '
            f'{collector_looks_at}')

    def test_an_old_snapshot_is_still_authoritative(self, tmp_path,
                                                     monkeypatch):
        """Being old is a snapshot's characteristic, not a defect."""
        import importlib
        import os
        import time

        monkeypatch.setenv('DATASET_NAME', 'worldbank')
        outputs = tmp_path / 'outputs'
        monkeypatch.setattr('core.config.get_outputs_root',
                            lambda: str(outputs / 'worldbank'))

        from core.config import get_absolute_output_path
        raw = Path(get_absolute_output_path('collection/raw_data'))
        raw.mkdir(parents=True)
        for name in ('complete_data.parquet', 'raw_data_long.parquet',
                     'scientific_collection_metadata.json',
                     'scientific_imputation_log.json'):
            path = raw / name
            path.write_text('{}')
            ancient = time.time() - 40 * 24 * 3600
            os.utime(path, (ancient, ancient))
        (Path(get_absolute_output_path('collection'))
         / 'snapshot_manifest.json').write_text('{}')

        module = importlib.import_module('collection.raw_data_collector')
        cls = next(getattr(module, n) for n in dir(module)
                   if isinstance(getattr(module, n), type)
                   and hasattr(getattr(module, n), '_cache_is_valid'))
        instance = cls.__new__(cls)
        instance.output_dir = str(raw)
        assert instance._cache_is_valid(), (
            'a forty-day-old verified snapshot triggered a re-download')


class TestTheImageCarriesWhatTheSuiteReads:
    """The build runs the suite, so a file the suite reads is a build input.

    `USAGE_GUIDE.md` was not among the copied files while three tests read it at
    collection time. The result was not an image missing its documentation: the
    build's `RUN pytest` failed at collection and no image was produced at all,
    so the containerised reproduction the README documents could not run.

    Pinned as a rule rather than as one filename, because the next file added to
    the repository root and read by a test would land in the same hole.
    """

    DOCKERFILE = (_ROOT / 'Dockerfile').read_text()

    @staticmethod
    def _root_files_the_suite_reads() -> set:
        """Root-level files referenced as `_ROOT / 'name'` anywhere in tests/."""
        pattern = re.compile(
            r"(?:_ROOT|_SRC\.parent|parents\[1\])\s*/\s*'([^'/]+\.[a-z]+)'")
        found = set()
        for path in (_ROOT / 'tests').glob('*.py'):
            found.update(pattern.findall(path.read_text()))
        # Two exclusions, and neither may use git: this test has to hold
        # inside the build context, which carries no .git -- a guard that
        # cannot run where the failure happens is not a guard.
        #
        # Dotfiles are the probes several tests write into the root and read
        # back; a file that does not exist is not a file the image is missing.
        return {name for name in found
                if not name.startswith('.') and (_ROOT / name).exists()}

    def _copied_root_files(self) -> set:
        copied = set()
        for line in self.DOCKERFILE.splitlines():
            if not line.startswith('COPY '):
                continue
            parts = line.split()[1:]
            # The last token is the destination; a trailing / marks a directory.
            for token in parts[:-1]:
                if not token.endswith('/'):
                    copied.add(token)
        return copied

    def test_the_build_runs_the_suite(self):
        """If it stops doing that, this whole class is checking nothing."""
        assert 'pytest' in self.DOCKERFILE, (
            'the build no longer runs the suite; the guarantee below is void')

    def test_every_root_file_the_suite_reads_is_copied(self):
        read = self._root_files_the_suite_reads()
        assert read, 'the detector found nothing; it has stopped working'
        missing = sorted(read - self._copied_root_files())
        assert not missing, (
            f'the suite reads {missing} from the repository root and the '
            f'Dockerfile does not copy them, so `docker build` fails at test '
            f'collection')
