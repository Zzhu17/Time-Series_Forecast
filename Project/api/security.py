from __future__ import annotations

import os
from typing import Optional

from fastapi import Header, HTTPException


def verify_api_token(authorization: Optional[str] = Header(default=None)) -> None:
    token = os.getenv("TSF_API_TOKEN") or os.getenv("API_TOKEN")
    if not token:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="missing API token")
    parts = authorization.split()
    provided = parts[-1] if len(parts) >= 2 else authorization
    if provided != token:
        raise HTTPException(status_code=403, detail="invalid API token")
