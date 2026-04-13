# ============================================================================
# GPU isolation is handled by:
#   - gpu_pin_early.py (-p gpu_pin_early): xdist workers (Path A)
#   - device_resource.py __main__: per-test remote / smoke (Path B)
# Both set CUDA_VISIBLE_DEVICES BEFORE entry-point plugins trigger cuInit().
# This conftest only handles markers, diagnostics, and memory cleanup.
# ============================================================================
import logging
import os
import re

import pytest

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
    print(f"\n[GPU_ASSIGN] {worker} pid={os.getpid()} {gpus}")
    try:
        import torch
        if torch.cuda.is_available():
            dc = torch.cuda.device_count()
            free, total = torch.cuda.mem_get_info(0)
            name = torch.cuda.get_device_name(0)
            print(
                f"[GPU_VERIFY] {worker} device_count={dc} name={name} "
                f"free={free/1e9:.1f}GB total={total/1e9:.1f}GB"
            )
    except Exception as e:
        print(f"[GPU_VERIFY] {worker} error: {e}")
    yield


# ============================================================================
# Per-test GPU memory monitoring + cleanup
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
def _gpu_mem_monitor(request):
    """Per-test GPU memory tracking and aggressive cleanup between tests."""
    before = _get_gpu_mem_mb()
    yield

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
                "[GPU_MEM] %s %s: alloc %.0f->%.0f MB (d%+.0f), "
                "reserved %.0f->%.0f MB (d%+.0f)",
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
