from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from src.config import BASE_DIR, get_settings


def _build_logger() -> logging.Logger:
    settings = get_settings()
    logs_dir = Path(os.getenv("RETENTION_LOG_DIR", BASE_DIR / "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s:%(lineno)d - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger("retention_ai")


logger = _build_logger()
