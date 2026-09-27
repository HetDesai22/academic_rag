"""Stage-level caching: a stage is skipped when its input fingerprint is unchanged and all of
its outputs still exist. Fingerprints are stored in reports/.pipeline_state.json."""
from pathlib import Path

from .common import now_iso, read_json, write_json


class PipelineState:
    def __init__(self, path, force=False):
        self.path = Path(path)
        self.force = force
        self.state = read_json(self.path, default={}) or {}

    def is_fresh(self, stage, fp, outputs):
        if self.force:
            return False
        rec = self.state.get(stage)
        return bool(rec) and rec.get("fingerprint") == fp and all(Path(o).exists() for o in outputs)

    def mark(self, stage, fp, info=None):
        self.state[stage] = {"fingerprint": fp, "completed_at": now_iso(), **(info or {})}
        write_json(self.state, self.path)

    def info(self, stage):
        return self.state.get(stage, {})
