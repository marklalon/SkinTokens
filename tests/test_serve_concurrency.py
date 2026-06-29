"""Concurrency / parallelism tests for the staged serve pipeline.

Covered guarantees:
  * bpy stages (parse + export) never run concurrently  -> no Blender scene
    pollution.
  * GPU inference is a single-instance critical section  -> one forward at a
    time.
  * bpy and GPU stages of *different* requests overlap   -> real pipelining.
  * the remote renamer runs in parallel, bounded by the semaphore.
  * results never cross-contaminate between concurrent requests.
  * cancellation unwinds cleanly (locks released, no leaked jobs).
"""

import asyncio
import time

import pytest

import serve


# --------------------------------------------------------------------------- #
# Fake stage workers
# --------------------------------------------------------------------------- #
def install_fake_stages(monkeypatch, tracker, *, stage_sleep=0.05, renamer_sleep=0.05):
    """Replace the heavy stage workers with light, instrumented stand-ins.

    Each fake threads the request_id through batch -> preds -> glbs -> renamed
    bytes so a test can prove every byte of a result belongs to its own request.
    """

    def fake_prepare(input_path, params, request_id, cancellation, progress_callback=None):
        tracker.enter("bpy", request_id)
        try:
            time.sleep(stage_sleep)
            cancellation.raise_if_cancelled()
            return {"req": request_id, "num_samples": params.num_samples}
        finally:
            tracker.exit("bpy", request_id)

    def fake_infer(batch, params, request_id, cancellation, progress_callback=None):
        tracker.enter("gpu", request_id)
        try:
            time.sleep(stage_sleep)
            cancellation.raise_if_cancelled()
            return [f"pred:{request_id}:{i}" for i in range(batch["num_samples"])]
        finally:
            tracker.exit("gpu", request_id)

    def fake_export(preds, params, request_id, cancellation, progress_callback, tmp_output_dir):
        tracker.enter("bpy", request_id)
        try:
            time.sleep(stage_sleep)
            cancellation.raise_if_cancelled()
            return [f"glb:{request_id}:{i}".encode() for i in range(len(preds))]
        finally:
            tracker.exit("bpy", request_id)

    async def fake_rename(glb_data, file_name, conf_thresh, request_id, cancellation):
        tracker.enter("renamer", request_id)
        try:
            await asyncio.sleep(renamer_sleep)
            cancellation.raise_if_cancelled()
            return glb_data + b":renamed", {"req": request_id}
        finally:
            tracker.exit("renamer", request_id)

    monkeypatch.setattr(serve, "_prepare_inputs", fake_prepare)
    monkeypatch.setattr(serve, "_run_inference", fake_infer)
    monkeypatch.setattr(serve, "_export_samples", fake_export)
    monkeypatch.setattr(serve, "_run_skeleton_rename_async", fake_rename)


def make_params(**kw):
    return serve.GenParams(**kw)


# --------------------------------------------------------------------------- #
# Pipeline: serialization, pipelining, no cross-contamination
# --------------------------------------------------------------------------- #
def test_bpy_and_gpu_serialized_and_pipelined(monkeypatch, tracker):
    install_fake_stages(monkeypatch, tracker)

    async def scenario():
        params = make_params(num_samples=1, skip_renamer=True)
        results = await asyncio.gather(*[
            serve._generate(b"data", "model.obj", params, f"req{i}")
            for i in range(4)
        ])
        return results

    results = asyncio.run(scenario())

    # Safety: each resource is a single-instance critical section.
    assert tracker.max_active["bpy"] == 1, "two bpy stages ran concurrently"
    assert tracker.max_active["gpu"] == 1, "two GPU inferences ran concurrently"
    # Throughput: bpy of one request overlapped GPU of another.
    assert tracker.overlap_bpy_gpu, "no bpy/GPU pipelining observed"

    # No cross-contamination: every glb belongs to its own request.
    assert len(results) == 4
    for i, samples in enumerate(results):
        assert len(samples) == 1
        glb, meta = samples[0]
        assert glb == f"glb:req{i}:0".encode()


def test_no_cross_contamination_multi_sample(monkeypatch, tracker):
    install_fake_stages(monkeypatch, tracker)

    async def scenario():
        results = {}
        params = [make_params(num_samples=n, skip_renamer=True) for n in (1, 3, 2)]

        async def one(i):
            results[i] = await serve._generate(
                b"x", "m.obj", params[i], f"req{i}"
            )

        await asyncio.gather(*(one(i) for i in range(3)))
        return results

    results = asyncio.run(scenario())

    expected_counts = {0: 1, 1: 3, 2: 2}
    for i, count in expected_counts.items():
        samples = results[i]
        assert len(samples) == count
        for sample_idx, (glb, meta) in enumerate(samples):
            assert glb == f"glb:req{i}:{sample_idx}".encode()


def test_gpu_lock_is_single_instance(monkeypatch, tracker):
    """Even under heavy fan-in, GPU never has more than one occupant."""
    install_fake_stages(monkeypatch, tracker, stage_sleep=0.02)

    async def scenario():
        params = make_params(num_samples=1, skip_renamer=True)
        await asyncio.gather(*[
            serve._generate(b"d", "m.obj", params, f"r{i}") for i in range(8)
        ])

    asyncio.run(scenario())
    assert tracker.max_active["gpu"] == 1


# --------------------------------------------------------------------------- #
# Renamer: bounded parallelism, ordering, passthrough
# --------------------------------------------------------------------------- #
def test_renamer_concurrency_bounded_by_semaphore(monkeypatch, tracker):
    install_fake_stages(monkeypatch, tracker, stage_sleep=0.0, renamer_sleep=0.08)

    async def scenario():
        serve.state.renamer_sem = asyncio.Semaphore(2)
        params = make_params(num_samples=5, skip_renamer=False)
        return await serve._generate(b"d", "m.obj", params, "req")

    samples = asyncio.run(scenario())

    assert len(samples) == 5
    # The semaphore must cap parallel renames, and they must actually parallelize.
    assert tracker.max_active["renamer"] <= 2
    assert tracker.max_active["renamer"] == 2
    for sample_idx, (glb, meta) in enumerate(samples):
        assert glb == f"glb:req:{sample_idx}".encode() + b":renamed"
        assert meta == {"req": "req"}


def test_renamer_preserves_sample_order_despite_out_of_order_completion(monkeypatch):
    """Later samples finishing first must not reorder the results."""

    async def fake_rename(glb_data, file_name, conf_thresh, request_id, cancellation):
        # idx encoded as the trailing number of the glb payload "glb:req:<idx>"
        idx = int(glb_data.decode().rsplit(":", 1)[1])
        # Earlier indices sleep longer, so completion order is reversed.
        await asyncio.sleep(0.02 * (5 - idx))
        return glb_data + b":renamed", {"idx": idx}

    monkeypatch.setattr(serve, "_run_skeleton_rename_async", fake_rename)

    async def scenario():
        serve.state.renamer_sem = asyncio.Semaphore(10)
        glbs = [f"glb:req:{i}".encode() for i in range(5)]
        reporter = serve.ProgressReporter("req", serve.CancellationToken(), None)
        return await serve._run_renamers(
            glbs, "m.obj", make_params(), "req", serve.CancellationToken(), reporter
        )

    results = asyncio.run(scenario())
    assert [meta["idx"] for _, meta in results] == [0, 1, 2, 3, 4]


def test_skip_renamer_passthrough(monkeypatch, tracker):
    install_fake_stages(monkeypatch, tracker)

    async def scenario():
        params = make_params(num_samples=2, skip_renamer=True)
        return await serve._generate(b"d", "m.obj", params, "req")

    samples = asyncio.run(scenario())
    assert tracker.max_active["renamer"] == 0  # renamer never invoked
    for sample_idx, (glb, meta) in enumerate(samples):
        assert glb == f"glb:req:{sample_idx}".encode()
        assert meta == {}


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #
def test_cancellation_unwinds_pipeline_and_releases_locks(monkeypatch, tracker):
    install_fake_stages(monkeypatch, tracker, stage_sleep=0.5)

    async def scenario():
        cancellation = serve.CancellationToken()
        params = make_params(num_samples=1, skip_renamer=True)
        task = asyncio.create_task(
            serve._generate(b"d", "m.obj", params, "req", None, cancellation)
        )
        await asyncio.sleep(0.05)
        cancellation.cancel("test cancel")
        with pytest.raises(serve.GenerationCancelled):
            await task

    asyncio.run(scenario())

    assert serve.state.active_jobs == 0
    assert not serve.state.bpy_lock.locked()
    assert not serve.state.gpu_lock.locked()


def test_renamer_cancellation_cancels_siblings(monkeypatch):
    started = []
    completed = []

    async def fake_rename(glb_data, file_name, conf_thresh, request_id, cancellation):
        started.append(glb_data)
        await asyncio.sleep(0.3)
        cancellation.raise_if_cancelled()
        completed.append(glb_data)
        return glb_data, {}

    monkeypatch.setattr(serve, "_run_skeleton_rename_async", fake_rename)

    async def scenario():
        serve.state.renamer_sem = asyncio.Semaphore(10)
        cancellation = serve.CancellationToken()
        glbs = [f"g{i}".encode() for i in range(4)]
        reporter = serve.ProgressReporter("req", cancellation, None)
        task = asyncio.create_task(
            serve._run_renamers(glbs, "m.obj", make_params(), "req", cancellation, reporter)
        )
        await asyncio.sleep(0.05)
        cancellation.cancel("cancel all")
        with pytest.raises(serve.GenerationCancelled):
            await task

    asyncio.run(scenario())
    assert len(started) == 4   # all dispatched in parallel
    assert completed == []     # none allowed to finish after cancel


def test_unsupported_extension_rejected():
    async def scenario():
        with pytest.raises(ValueError):
            await serve._generate(b"d", "model.txt", make_params(), "req")

    asyncio.run(scenario())
    assert serve.state.active_jobs == 0


# --------------------------------------------------------------------------- #
# RNG isolation (fix #3)
# --------------------------------------------------------------------------- #
def test_seeded_rng_is_deterministic():
    import torch

    with serve._seeded_torch_rng(123):
        a = torch.rand(8)
    with serve._seeded_torch_rng(123):
        b = torch.rand(8)
    assert torch.equal(a, b)


def test_seeded_rng_does_not_perturb_surrounding_torch_stream():
    """A seeded request must not shift another request's global RNG stream."""
    import torch

    torch.manual_seed(999)
    ref_first = torch.rand(4)
    ref_second = torch.rand(4)

    torch.manual_seed(999)
    again_first = torch.rand(4)
    with serve._seeded_torch_rng(123):
        torch.rand(4)  # consume seeded stream; must be rolled back on exit
    again_second = torch.rand(4)

    assert torch.equal(again_first, ref_first)
    assert torch.equal(again_second, ref_second)


def test_seeded_rng_leaves_numpy_untouched():
    """Seed is scoped to torch only; numpy belongs to the (serialized) bpy stage."""
    import numpy as np

    np.random.seed(7)
    before = np.random.get_state()[1].copy()
    with serve._seeded_torch_rng(123):
        pass
    after = np.random.get_state()[1]
    assert np.array_equal(before, after)


def test_seeded_rng_none_is_passthrough():
    import torch

    torch.manual_seed(42)
    ref = torch.rand(4)
    torch.manual_seed(42)
    with serve._seeded_torch_rng(None):
        out = torch.rand(4)
    assert torch.equal(ref, out)
