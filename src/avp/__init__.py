"""AUT Video Pipeline (avp) — a local, open-source engine for faceless short-form videos.

Pipeline stages: script -> voice (TTS) -> footage -> captions -> assemble.
Everything runs locally on Apple Silicon; all default models are commercial-license-clean.
"""

__version__ = "0.1.0"

# v3 editorial layer: keep the public `stages.stage_script` entry point used by the CLI and
# unattended runner, but replace its implementation with the new story-selection/review engine.
# The import is intentionally last so normal package initialisation remains unchanged.
from . import stages as _stages
from . import editorial_engine as _editorial_engine
_stages.stage_script = _editorial_engine.stage_script
