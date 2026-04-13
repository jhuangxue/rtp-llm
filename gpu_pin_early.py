"""Earliest-possible GPU pinning for xdist workers.

Loaded via ``-p gpu_pin_early`` (from addopts or explicit CLI).
pytest processes ``-p`` flags in ``_preparse()`` **before**
``consider_setuptools_entrypoints()``, so this module-level code
sets CUDA_VISIBLE_DEVICES before any entry-point plugin can trigger
``import rtp_llm`` -> ``import torch`` -> ``cuInit()``.

Uses importlib.util to load device_resource.py directly so that
rtp_llm/__init__.py is never imported.

IMPORTANT: Always acquires a fresh GPU lock via DeviceResource, even
if CUDA_VISIBLE_DEVICES is already set. In xdist mode, any pre-existing
CVD is inherited from the parent process or REAPI framework, not from
our isolation code, and must be overridden with a properly locked GPU.
"""
import os as _os
import sys as _sys

_xdist_worker = _os.environ.get("PYTEST_XDIST_WORKER")
if _xdist_worker:
    import importlib.util as _ilu

    _gpu_count = int(_os.environ.get("GPU_COUNT_PER_WORKER", "1"))
    _old_cvd = _os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    _old_hvd = _os.environ.pop("HIP_VISIBLE_DEVICES", None)

    _dr_path = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "rtp_llm", "test", "utils", "device_resource.py",
    )
    if not _os.path.isfile(_dr_path):
        for _sp in _sys.path:
            _c = _os.path.join(_sp, "rtp_llm", "test", "utils", "device_resource.py")
            if _os.path.isfile(_c):
                _dr_path = _c
                break
    if not _os.path.isfile(_dr_path):
        raise RuntimeError(
            "gpu_pin_early: cannot find rtp_llm/test/utils/device_resource.py "
            "(searched CWD + sys.path)"
        )

    _spec = _ilu.spec_from_file_location("_dr_early", _dr_path)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    _info = _mod.get_device_info()
    if not _info:
        raise RuntimeError(
            f"gpu_pin_early: {_xdist_worker}: get_device_info() returned None — "
            "nvidia-smi / rocm-smi not found or not working"
        )

    _dr = _mod.DeviceResource(required_gpu_count=_gpu_count)
    _dr.__enter__()  # flock held until worker process exits (OS auto-releases)

    _env = _mod._get_visible_devices_env(_info[0])
    _os.environ[_env] = ",".join(_dr.gpu_ids)

    _sys.stderr.write(
        f"[gpu_pin_early] {_xdist_worker}: {_env}={_os.environ[_env]} "
        f"locked={_dr.gpu_ids}"
        + (f" (overrode inherited CVD={_old_cvd})" if _old_cvd else "")
        + "\n"
    )
    _sys.stderr.flush()
