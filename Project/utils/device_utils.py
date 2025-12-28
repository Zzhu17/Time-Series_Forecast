from __future__ import annotations

from typing import Any, Optional

import torch


def resolve_device(device: Optional[str] = None) -> torch.device:
    d = (device or '').strip().lower()
    if d in ('', 'auto', 'default', 'none'):
        if getattr(torch.backends, 'mps', None) is not None:
            try:
                if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                    return torch.device('mps')
            except Exception:
                pass
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    # Common explicit values
    if d == 'mps':
        try:
            if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                return torch.device('mps')
        except Exception:
            pass
        raise RuntimeError('MPS was requested but is not available in this PyTorch build.')
    if d == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if d == 'cpu':
        return torch.device('cpu')

    # Allow advanced specs like 'cuda:0'
    try:
        dev = torch.device(d)
        # Validate non-cpu devices
        if dev.type == 'cuda' and not torch.cuda.is_available():
            return torch.device('cpu')
        if dev.type == 'mps':
            try:
                if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
                    return torch.device('cpu')
            except Exception:
                return torch.device('cpu')
        return dev
    except Exception:
        return torch.device('cpu')


def get_device_from_config(config: Any) -> torch.device:
    dev: Optional[str] = None
    if isinstance(config, dict):
        dev = config.get('device')
        if not dev:
            dev = (config.get('default') or {}).get('device')
    return resolve_device(dev)
