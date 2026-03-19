from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

from small_target_optimizer import (
    _is_yolo_checkpoint_compat_error,
    _load_yolo_model_with_checkpoint_compat,
)


def test_detects_weights_only_checkpoint_error() -> None:
    message = (
        'Weights only load failed. Unsupported global: GLOBAL '
        'torch.nn.modules.container.Sequential was not an allowed global by default.'
    )

    assert _is_yolo_checkpoint_compat_error(message) is True
    assert _is_yolo_checkpoint_compat_error('CUDA error: out of memory') is False


def test_load_yolo_model_retries_with_local_checkpoint_compat(tmp_path: Path) -> None:
    checkpoint = tmp_path / 'trusted-local-model.pt'
    _ = checkpoint.write_bytes(b'checkpoint')

    added_safe_globals: list[type[object]] = []
    load_calls: list[dict[str, object]] = []

    class DetectionModel:
        pass

    class Sequential:
        pass

    def original_torch_load(**kwargs: object) -> object:
        load_calls.append(dict(kwargs))
        if kwargs.get('weights_only') is False:
            return {'ok': True}
        raise RuntimeError(
            'Weights only load failed. WeightsUnpickler error: Unsupported global: '
            'GLOBAL torch.nn.modules.container.Sequential'
        )

    torch_module = types.ModuleType('torch')
    setattr(torch_module, 'load', lambda *args, **kwargs: original_torch_load(**kwargs))
    original_module_torch_load = torch_module.load
    setattr(
        torch_module,
        'serialization',
        types.SimpleNamespace(add_safe_globals=lambda values: added_safe_globals.extend(values)),
    )
    setattr(
        torch_module,
        'nn',
        types.SimpleNamespace(modules=types.SimpleNamespace(container=types.SimpleNamespace(Sequential=Sequential))),
    )

    ultralytics_tasks_module = types.ModuleType('ultralytics.nn.tasks')
    setattr(ultralytics_tasks_module, 'DetectionModel', DetectionModel)
    setattr(ultralytics_tasks_module, 'torch_load', lambda *args, **kwargs: original_torch_load(**kwargs))
    original_tasks_torch_load = ultralytics_tasks_module.torch_load

    ultralytics_nn_module = types.ModuleType('ultralytics.nn')
    setattr(ultralytics_nn_module, 'tasks', ultralytics_tasks_module)

    yolo_state = {'calls': 0}

    class FakeYOLO:
        def __init__(self, model_path: str) -> None:
            yolo_state['calls'] += 1
            ultralytics_tasks_module.torch_load(model_path)

    ultralytics_module = types.ModuleType('ultralytics')
    setattr(ultralytics_module, 'YOLO', FakeYOLO)

    with patch.dict(
        sys.modules,
        {
            'torch': torch_module,
            'ultralytics': ultralytics_module,
            'ultralytics.nn': ultralytics_nn_module,
            'ultralytics.nn.tasks': ultralytics_tasks_module,
        },
    ):
        model = _load_yolo_model_with_checkpoint_compat(checkpoint, FakeYOLO)

    assert isinstance(model, FakeYOLO)
    assert yolo_state['calls'] == 2
    assert load_calls[0].get('weights_only') is None
    assert load_calls[1].get('weights_only') is False
    assert DetectionModel in added_safe_globals
    assert Sequential in added_safe_globals
    assert ultralytics_tasks_module.torch_load is original_tasks_torch_load
    assert torch_module.load is original_module_torch_load


def test_load_yolo_model_does_not_retry_on_unrelated_error(tmp_path: Path) -> None:
    checkpoint = tmp_path / 'trusted-local-model.pt'
    _ = checkpoint.write_bytes(b'checkpoint')

    class FakeYOLO:
        def __init__(self, model_path: str) -> None:
            raise RuntimeError(f'generic failure for {model_path}')

    try:
        _load_yolo_model_with_checkpoint_compat(checkpoint, FakeYOLO)
    except RuntimeError as exc:
        assert 'generic failure' in str(exc)
    else:
        raise AssertionError('Expected RuntimeError for unrelated YOLO load failure')
