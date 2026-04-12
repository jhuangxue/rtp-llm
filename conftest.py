# ============================================================================
# xdist Per-Worker GPU Lock (MUST be first — before any torch import)
#
# Uses DeviceResource to dynamically acquire GPUs via file locks.
# CUDA_VISIBLE_DEVICES is set to the locked GPUs before torch can import.
# flock() is auto-released by OS when the worker process exits (even on crash).
# ============================================================================
import os as _os
import sys as _sys

_xdist_worker = _os.environ.get("PYTEST_XDIST_WORKER")
_worker_device_resource = None  # held for worker process lifetime

if _xdist_worker:
    _gpu_count = int(_os.environ.get("GPU_COUNT_PER_WORKER", "1"))
    # CRITICAL: import device_resource WITHOUT triggering rtp_llm/__init__.py,
    # which imports torch (via torch_patch.py). Torch must NOT be imported before
    # we set CUDA_VISIBLE_DEVICES, or it binds to all visible GPUs permanently.
    import importlib.util as _ilu
    _dr_rel = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "rtp_llm", "test", "utils", "device_resource.py",
    )
    _dr_path = None
    if _os.path.isfile(_dr_rel):
        _dr_path = _dr_rel
    else:
        for _sp in _sys.path:
            _candidate = _os.path.join(_sp, "rtp_llm", "test", "utils", "device_resource.py")
            if _os.path.isfile(_candidate):
                _dr_path = _candidate
                break
    if not _dr_path:
        raise RuntimeError("Cannot find rtp_llm/test/utils/device_resource.py on sys.path")
    _dr_spec = _ilu.spec_from_file_location("_device_resource_early", _dr_path)
    _dr_mod = _ilu.module_from_spec(_dr_spec)
    _dr_spec.loader.exec_module(_dr_mod)

    _device_info = _dr_mod.get_device_info()
    if not _device_info:
        raise RuntimeError(
            f"xdist worker {_xdist_worker}: get_device_info() returned None — "
            f"nvidia-smi / rocm-smi not found or not working. "
            f"Cannot isolate GPUs without device detection."
        )
    _worker_device_resource = _dr_mod.DeviceResource(required_gpu_count=_gpu_count)
    _worker_device_resource.__enter__()
    _env_name = _dr_mod._get_visible_devices_env(_device_info[0])
    _os.environ[_env_name] = ",".join(_worker_device_resource.gpu_ids)
    _diag_msg = (
        f"[conftest] worker={_xdist_worker} pid={_os.getpid()} "
        f"{_env_name}={_os.environ[_env_name]} "
        f"locked_gpus={_worker_device_resource.gpu_ids}"
    )
    _sys.stderr.write(_diag_msg + "\n")
    _sys.stderr.flush()
    _diag_dir = "/tmp/rtp_llm_gpu_diag"
    _os.makedirs(_diag_dir, exist_ok=True)
    with open(f"{_diag_dir}/{_xdist_worker}", "w") as _f:
        _f.write(_diag_msg + "\n")

# ============================================================================
import logging
import os
import pytest
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Session-scoped diagnostic — prints GPU assignment visible in xdist output
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def _log_gpu_assignment():
    """Log GPU assignment at session start (visible in xdist worker output)."""
    worker = os.environ.get("PYTEST_XDIST_WORKER", "controller")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    hvd = os.environ.get("HIP_VISIBLE_DEVICES", "unset")
    gpus = f"CUDA={cvd}" if hvd == "unset" else f"HIP={hvd}"
    locked = _worker_device_resource.gpu_ids if _worker_device_resource else "N/A"
    print(f"\n[GPU_ASSIGN] {worker} pid={os.getpid()} {gpus} locked={locked}")
    yield


# ============================================================================
# Per-test GPU memory monitoring + isolation check
# ============================================================================

def _get_gpu_mem_mb():
    """Return (allocated_MB, reserved_MB) for current default CUDA device, or None."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return (
            torch.cuda.memory_allocated() / (1024 * 1024),
            torch.cuda.memory_reserved() / (1024 * 1024),
        )
    except Exception:
        return None


@pytest.fixture(scope="function", autouse=True)
def _gpu_isolation_and_mem_monitor(request):
    """Per-test GPU guard + memory tracking.

    - Checks that the test's gpu(count=N) fits the worker's allocation.
    - Logs GPU memory before/after the test to detect leaks.
    """
    gpu_marker = request.node.get_closest_marker("gpu")
    if gpu_marker and _xdist_worker and _worker_device_resource:
        gpu_count = int(gpu_marker.kwargs.get("count", 1))
        if gpu_count > len(_worker_device_resource.gpu_ids):
            pytest.fail(
                f"Test needs {gpu_count} GPUs but worker locked "
                f"{len(_worker_device_resource.gpu_ids)} "
                f"(GPU_COUNT_PER_WORKER="
                f"{_os.environ.get('GPU_COUNT_PER_WORKER', '?')}). "
                f"Check phase configuration.",
                pytrace=False,
            )

    before = _get_gpu_mem_mb()
    yield

    # Aggressively reclaim GPU memory between tests to prevent OOM from
    # accumulated allocations in the same xdist worker process.
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    after = _get_gpu_mem_mb()

    if before is not None and after is not None:
        alloc_before, reserved_before = before
        alloc_after, reserved_after = after
        delta_alloc = alloc_after - alloc_before
        delta_reserved = reserved_after - reserved_before
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        if abs(delta_alloc) > 10 or abs(delta_reserved) > 100:
            logger.warning(
                "[GPU_MEM] %s %s: alloc %.0f→%.0f MB (Δ%+.0f), "
                "reserved %.0f→%.0f MB (Δ%+.0f)",
                worker, request.node.nodeid,
                alloc_before, alloc_after, delta_alloc,
                reserved_before, reserved_after, delta_reserved,
            )


# ============================================================================
# Marker Registration & Rewriting
# ============================================================================

def _register_synthetic_gpu_marker(config, count: int) -> str:
    synthetic_name = f"gpu_count_{count}"
    registered = getattr(config, "_synthetic_gpu_markers", set())
    if synthetic_name not in registered:
        config.addinivalue_line(
            "markers",
            f"{synthetic_name}: synthetic marker for gpu(count={count}) filtering",
        )
        registered.add(synthetic_name)
        config._synthetic_gpu_markers = registered
    return synthetic_name


def pytest_configure(config):
    """Rewrite gpu(count=N) in -m expressions and register needed synthetic markers."""
    config.addinivalue_line("markers", "manual: test requires manual execution (deselected by default)")
    config._synthetic_gpu_markers = set()

    marker_expr = config.option.markexpr
    if marker_expr:
        rewritten = re.sub(
            r'(?<!\w)gpu\(count\s*=\s*(\d+)\)',
            r'gpu_count_\1',
            marker_expr,
        )
        config.option.markexpr = rewritten
        for count in re.findall(r'(?<!\w)gpu_count_(\d+)\b', rewritten):
            _register_synthetic_gpu_marker(config, int(count))
        logger.debug(f"Modified marker expression: {marker_expr} -> {rewritten}")


# ============================================================================
# GPU Lock for smoke tests (unchanged — uses DeviceResource directly)
# ============================================================================

def _get_gpu_count_from_markers(node) -> int:
    """Get required GPU count from @pytest.mark.gpu(count=N), GPU_COUNT env, or default 1."""
    gpu_marker = node.get_closest_marker("gpu")
    if gpu_marker:
        if "count" in gpu_marker.kwargs:
            return int(gpu_marker.kwargs["count"])
        return 1

    gpu_count_env = os.environ.get("GPU_COUNT")
    if gpu_count_env:
        try:
            return int(gpu_count_env)
        except ValueError:
            logger.warning(f"Invalid GPU_COUNT env: {gpu_count_env}, using default 1")

    return 1


@pytest.fixture(scope="function")
def gpu_lock(request):
    """Function-scoped GPU lock for smoke tests.

    Acquires N GPUs via DeviceResource file locks and sets CUDA_VISIBLE_DEVICES.
    This only affects SUBPROCESSES spawned after the fixture (e.g., server
    processes in smoke tests).  For py-ut under xdist, the module-level
    DeviceResource handles GPU assignment instead.
    """
    if request.node.get_closest_marker("no_gpu_lock"):
        yield None
        return

    gpu_count = _get_gpu_count_from_markers(request.node)
    if gpu_count < 1:
        yield None
        return

    from rtp_llm.test.utils.device_resource import (
        DeviceResource,
        GpuLockError,
        GPU_LOCK_DEFAULT_TIMEOUT,
        GPU_LOCK_TIMEOUT_ENV,
        get_device_info,
        _get_visible_devices_env,
    )

    device_info = get_device_info()
    if not device_info:
        yield None
        return

    device_name, _ = device_info
    env_name = _get_visible_devices_env(device_name)

    lock_timeout = int(os.environ.get(GPU_LOCK_TIMEOUT_ENV, GPU_LOCK_DEFAULT_TIMEOUT))
    try:
        with DeviceResource(required_gpu_count=gpu_count, timeout=lock_timeout) as gpu_resource:
            os.environ[env_name] = ",".join(gpu_resource.gpu_ids)
            logger.info(f"gpu_lock: {env_name}={os.environ[env_name]} (count={gpu_count})")
            yield gpu_resource
    except GpuLockError as exc:
        pytest.fail(f"GPU lock failed: {exc}", pytrace=False)


# ============================================================================
# Collection hooks
# ============================================================================

@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """
    - Deselect tests marked @pytest.mark.manual (require manual execution).
    - Add synthetic gpu_count_N markers before pytest applies -m selection.
    """
    marker_expr = getattr(config.option, "markexpr", "") or ""
    if "manual" not in marker_expr:
        manual_items = []
        remaining = []
        for item in items:
            if item.get_closest_marker("manual"):
                manual_items.append(item)
            else:
                remaining.append(item)
        if manual_items:
            config.hook.pytest_deselected(items=manual_items)
            items[:] = remaining

    for item in items:
        gpu_marker = item.get_closest_marker("gpu")
        if not gpu_marker:
            continue

        gpu_type = gpu_marker.kwargs.get("type")
        if gpu_type:
            item.add_marker(getattr(pytest.mark, gpu_type))

        count = gpu_marker.kwargs.get("count", 1)
        try:
            count = int(count)
        except (TypeError, ValueError):
            count = 1

        synthetic_name = _register_synthetic_gpu_marker(config, count)
        item.add_marker(getattr(pytest.mark, synthetic_name))
