"""In-memory project store.

A dict behind a lock, and that is the right call for a prototype. The PDF asks
for saving and re-editing projects and for concurrent users; a dict keyed by
project id gives both, because each request only touches its own project and
the lock makes the dict itself safe.

What this deliberately does not do is persist across a restart. Adding SQLite
would be a schema, a migration and a serialisation layer for an assessment
prototype that gains nothing from surviving a reboot. Swapping this module for
a real store later touches nothing else - every caller goes through these four
functions.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .schemas import Analysis


@dataclass
class Project:
    id: str
    created_at: str

    original_bytes: bytes = b""
    original_mime: str = "image/jpeg"
    image_w: int = 0
    image_h: int = 0
    filename: str = ""

    analysis: Analysis | None = None
    redesign_bytes: bytes | None = None
    redesign_mime: str = "image/png"
    redesign_engine: str = ""  # "gemini" | "local" - shown in the UI and report

    selections: dict[str, str] = field(default_factory=dict)
    rate_overrides: dict[str, dict[str, float]] = field(default_factory=dict)
    scale_override: dict[str, Any] | None = None

    def summary(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "filename": self.filename,
            "has_analysis": self.analysis is not None,
            "has_redesign": self.redesign_bytes is not None,
            "redesign_engine": self.redesign_engine,
            "region_count": len(self.analysis.regions) if self.analysis else 0,
        }


_lock = threading.Lock()
_projects: dict[str, Project] = {}


def create(image_bytes: bytes, mime: str, w: int, h: int, filename: str) -> Project:
    pid = uuid.uuid4().hex[:12]
    p = Project(
        id=pid,
        created_at=datetime.now().isoformat(timespec="seconds"),
        original_bytes=image_bytes,
        original_mime=mime,
        image_w=w,
        image_h=h,
        filename=filename,
    )
    with _lock:
        _projects[pid] = p
    return p


def get(pid: str) -> Project | None:
    with _lock:
        return _projects.get(pid)


def all_projects() -> list[Project]:
    with _lock:
        return sorted(_projects.values(), key=lambda p: p.created_at, reverse=True)


def delete(pid: str) -> bool:
    with _lock:
        return _projects.pop(pid, None) is not None
