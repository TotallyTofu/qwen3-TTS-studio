"""Podcast quality presets - the single source of truth for the UI and the orchestrator.

The UI dropdown only passes the preset *name* to the orchestrator, so both
sides must read the same table. (They used to diverge: the UI advertised
temperature 1.0 for "premium" while the orchestrator ran it at 0.2.)

Sampling note: Qwen3-TTS is a codec language model. Near-greedy decoding
(temperature <= ~0.3, no repetition penalty) lets it get stuck repeating a
silence frame for many seconds, producing long dead-air holes mid-sentence.
In A/B tests on the same voice prompt and text, temperature 0.2 with
repetition_penalty 1.0 left 58% of chunks with >=1.5s silences, while the
model's own generation_config (temperature 0.9, top_k 50, top_p 1.0,
repetition_penalty 1.05) left none. All presets therefore share those
sampling values and differ only in episode length.
"""

from __future__ import annotations

# Mirrors generation_config.json shipped with the Qwen3-TTS-12Hz checkpoints.
MODEL_SAMPLING_DEFAULTS: dict[str, float | int] = {
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
    "subtalker_temperature": 0.9,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
}

PODCAST_QUALITY_PRESETS: dict[str, dict[str, object]] = {
    "quick": {
        "num_segments": 2,
        **MODEL_SAMPLING_DEFAULTS,
        "max_new_tokens": 768,
        "duration_estimate": "2-3 min",
        "tooltip": "Fast generation with 2-3 segments. Best for quick demos and testing. ~2-3 minutes total.",
    },
    "standard": {
        "num_segments": 4,
        **MODEL_SAMPLING_DEFAULTS,
        "max_new_tokens": 1024,
        "duration_estimate": "5-7 min",
        "tooltip": "Balanced quality and speed with 4-5 segments. Recommended for most podcasts. ~5-7 minutes total.",
    },
    "premium": {
        "num_segments": 6,
        **MODEL_SAMPLING_DEFAULTS,
        "max_new_tokens": 1400,
        "duration_estimate": "10-15 min",
        "tooltip": "High quality with 6-8 segments. Best for professional podcasts. ~10-15 minutes total.",
    },
}

# Older preset names still found in saved sessions.
LEGACY_PRESET_ALIASES = {"draft": "quick", "high": "premium"}

TTS_PARAM_KEYS = (
    "temperature",
    "top_k",
    "top_p",
    "repetition_penalty",
    "max_new_tokens",
    "subtalker_temperature",
    "subtalker_top_k",
    "subtalker_top_p",
)


def get_tts_params(preset_name: str) -> dict[str, object]:
    """Return the TTS sampling params for a preset name ({} if unknown)."""
    name = preset_name.strip().lower()
    name = LEGACY_PRESET_ALIASES.get(name, name)
    preset = PODCAST_QUALITY_PRESETS.get(name)
    if preset is None:
        return {}
    return {key: preset[key] for key in TTS_PARAM_KEYS}
