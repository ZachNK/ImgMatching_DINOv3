# project/imatch/models.py
import os
import torch
from pathlib import Path
from typing import Optional, Tuple

from .env import DINOV3_BLOCK_NET

def _block_hub_net_if_needed():
    if DINOV3_BLOCK_NET:
        import torch.hub as _th
        def _no_dl(*a, **k):
            raise RuntimeError("Blocked torch.hub.load_state_dict_from_url (offline)")
        _th.load_state_dict_from_url = _no_dl  # type: ignore

def load_model(repo_dir: Path, device: str, hub_name: Optional[str], ckpt: Optional[Path]) -> Tuple[torch.nn.Module, str]:
    """
    torch.hub 로 로컬 리포에서 모델 생성 후, ckpt state_dict 로딩.
    hub_name이 None이면 dinov3_vitl16 → vitb16 → vits16 순으로 시도.
    """
    _block_hub_net_if_needed()

    if hub_name:
        print(f"[model] hub.load entry='{hub_name}' from {repo_dir}")
        hub_kwargs = {}
        if ckpt is not None:
            hub_kwargs["pretrained"] = False
        try:
            model = torch.hub.load(str(repo_dir), hub_name, source="local", trust_repo=True, **hub_kwargs)
        except TypeError:
            model = torch.hub.load(str(repo_dir), hub_name, source="local", trust_repo=True)
    else:
        tried = ["dinov3_vitl16", "dinov3_vitb16", "dinov3_vits16"]
        last_err = None
        model = None
        for name in tried:
            try:
                model = torch.hub.load(str(repo_dir), name, source="local", trust_repo=True,
                                       pretrained=False if ckpt else True)
                hub_name = name
                break
            except TypeError:
                model = torch.hub.load(str(repo_dir), name, source="local", trust_repo=True)
                hub_name = name
                break
            except Exception as e:
                last_err = e
        if model is None:
            raise SystemExit(f"Failed to load hub model. Last error: {last_err}")

    model.eval().to(device)

    if ckpt is not None:
        print(f"[ckpt] loading: {ckpt}")
        # PyTorch 2.0+ 안전 로딩: weights_only=True
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        new_state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        if missing:    print(f"[ckpt][warn] missing={len(missing)}")
        if unexpected: print(f"[ckpt][warn] unexpected={len(unexpected)}")
    return model, (hub_name or "hub_model")
