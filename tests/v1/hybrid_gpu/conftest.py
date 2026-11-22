# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixtures for the real-GPU tests of the CPU KV tiers.

Every test drives a real engine on a real GPU: no fake device, no mocks, no
patched internals. Run them in parallel with pytest-xdist; each xdist worker
takes one GPU (`-n` up to twice the GPU count fits the tiny models), and
tests that need tensor parallelism run without xdist so they see all GPUs:

    pytest tests/v1/hybrid_gpu -n 4                 # TP=1 cases
    pytest tests/v1/hybrid_gpu -k "tp2 or tp4 or tp8"   # needs 2+ GPUs
"""

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

# Before vLLM is imported: the engine core runs in this process, so tests can
# read the scheduler at any tensor-parallel size.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

_REPO_ROOT = Path(__file__).resolve().parents[3]
# Workers of a TP>1 engine are separate processes: they import the probes'
# worker functions from here.
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(_REPO_ROOT), os.environ.get("PYTHONPATH")) if p
)


def _gpu_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [d for d in visible.split(",") if d]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


# One GPU per xdist worker, set before CUDA initializes in this process.
_xdist_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _xdist_worker is not None and (_ids := _gpu_ids()):
    os.environ["CUDA_VISIBLE_DEVICES"] = _ids[int(_xdist_worker[2:]) % len(_ids)]

import pytest  # noqa: E402
import torch  # noqa: E402

from .engine import Engine, EngineSpec  # noqa: E402

_HERE = Path(__file__).parent


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU: these tests run the real engine")
    for item in items:
        if _HERE in Path(item.path).parents:
            item.add_marker(skip)


class Engines:
    """At most one live engine per test process (GPU memory): a test asking
    for a different spec replaces it, one asking for the same spec gets it
    back with both tiers emptied."""

    def __init__(self, model_root: Path):
        self.model_root = model_root
        self.current: Engine | None = None

    def get(self, spec: EngineSpec) -> Engine:
        num_gpus = torch.cuda.device_count()
        if spec.tp > num_gpus:
            pytest.skip(f"TP={spec.tp} needs {spec.tp} GPUs, {num_gpus} visible")
        if self.current is not None and self.current.spec == spec:
            self.current.reset()
            return self.current
        self.close()
        model_dir = spec.model.write(self.model_root)
        try:
            self.current = Engine(spec, model_dir)
        except Exception:
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

            cleanup_dist_env_and_memory()
            raise
        return self.current

    def close(self) -> None:
        if self.current is not None:
            engine, self.current = self.current, None
            engine.shutdown()


@pytest.fixture(scope="session")
def model_root(tmp_path_factory) -> Path:
    shared = os.environ.get("HYBRID_GPU_MODEL_DIR")
    return Path(shared) if shared else tmp_path_factory.mktemp("models")


@pytest.fixture(scope="session")
def engines(model_root) -> Iterator[Engines]:
    pool = Engines(model_root)
    yield pool
    pool.close()


@pytest.fixture
def engine(request, engines) -> Engine:
    """The engine for the test's `EngineSpec` (indirect parametrization)."""
    spec = request.param
    assert isinstance(spec, EngineSpec), "parametrize `engine` with an EngineSpec"
    return engines.get(spec)


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    """tests/conftest.py tears down the distributed state after each test;
    the engines here outlive a test (`engines`)."""
    return False
