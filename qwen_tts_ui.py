#!/usr/bin/env python3
import os

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import torch
import gradio as gr
import soundfile as sf
import numpy as np
import tempfile
import os
import json
import pickle
import shutil
import zipfile
import time
import copy
import gc
import queue
import re
import threading
import traceback
import html as html_escape
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from podcast import orchestrator as podcast_orchestrator
from podcast.presets import PODCAST_QUALITY_PRESETS
from podcast.script_parser import parse_script
from ui.content_input import (
    get_content_components,
    update_topic_char_count,
    validate_content,
    get_content_dict,
)
from ui.voice_cards import (
    get_selection_summary,
    validate_selections,
    ROLES,
    MAX_VOICES,
    generate_preview,
)
from ui.progress import (
    GenerationStep,
    ProgressState,
    ProgressTracker,
    create_step_indicator_html,
    create_status_text,
    calculate_overall_progress,
    format_time_remaining,
    PROGRESS_CSS,
)
from ui.draft_preview import (
    build_outline_html,
    render_dialogues_html,
    get_segment_dialogues,
    DRAFT_PREVIEW_CSS,
)
from ui.theme import APP_CSS, APP_JS, HEADER_HTML, build_theme
from storage.persona_models import (
    ALLOWED_PERSONALITIES,
    ALLOWED_SPEAKING_STYLES,
    Persona,
)
from storage.persona import delete_persona, list_personas, load_persona, save_persona
from podcast.models import Outline, Transcript, SpeakerProfile
from storage.voice import get_available_voices, get_saved_voices, create_speaker_profile
from storage.history import read_json_file
from api.openai_server import (
    MAX_LINKS as API_MAX_LINKS,
    MAX_SPEED as API_MAX_SPEED,
    MIN_SPEED as API_MIN_SPEED,
    OPENAI_VOICES,
    api_server,
    change_speed,
    client_base_url,
    find_link,
    load_api_settings,
    save_api_settings,
    synthesize_link,
    voice_display_name,
)
from config import get_openai_api_key
from podcast.llm_client import (
    LLMConfig,
    LLMProvider,
    PROVIDER_MODEL_OPTIONS,
    DEFAULT_MODELS,
    get_default_config,
    validate_connection,
)


def auto_transcribe_audio(audio_path: str | None) -> str:
    """
    Transcribe audio using OpenAI Whisper API.

    Args:
        audio_path: Path to the audio file to transcribe

    Returns:
        Transcribed text or error message
    """
    if not audio_path:
        return "Error: No audio file provided. Please upload or record audio first."

    if not os.path.exists(audio_path):
        return f"Error: Audio file not found at {audio_path}"

    try:
        from openai import OpenAI, APIError, APITimeoutError, RateLimitError
    except ImportError:
        return "Error: OpenAI package not installed. Please install it with: pip install openai"

    try:
        api_key = get_openai_api_key()
    except ValueError as e:
        return f"Error: {str(e)}"

    try:
        client = OpenAI(api_key=api_key)

        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )

        return transcript.text

    except RateLimitError:
        return (
            "Error: OpenAI API rate limit exceeded. Please wait a moment and try again."
        )
    except APITimeoutError:
        return "Error: OpenAI API request timed out. Please try again."
    except APIError as e:
        return f"Error: OpenAI API error - {str(e)}"
    except Exception as e:
        return f"Error: Failed to transcribe audio - {str(e)}"


def format_user_error(error: Exception) -> str:
    error_messages = {
        "CUDA out of memory": "Not enough GPU memory. Try reducing text length or use a smaller model.",
        "Connection refused": "Cannot connect to server. Please check if the service is running.",
        "Rate limit": "Too many requests. Please wait a moment and try again.",
        "Invalid audio": "The audio file could not be processed. Please try a different file.",
    }
    error_str = str(error)
    for key, msg in error_messages.items():
        if key.lower() in error_str.lower():
            return msg
    return f"An error occurred: {error_str[:200]}"


SAVED_VOICES_DIR = Path("saved_voices")
SAVED_VOICES_DIR.mkdir(exist_ok=True)
HISTORY_DIR = Path("generation_history")
HISTORY_DIR.mkdir(exist_ok=True)
SETTINGS_FILE = Path("tts_settings.json")
FAVORITES_FILE = Path("favorites.json")

from audio.model_loader import MODEL_PATHS, get_model, loaded_models, _gpu_cleanup
from audio.embedding_utils import (
    AudioSampleInfo,
    analyze_audio_samples,
    combine_speaker_embeddings,
    create_combined_voice_clone_prompt,
    format_samples_summary,
    get_sample_warnings,
    get_audio_duration as get_audio_duration_util,
    estimate_snr,
)

DEFAULT_PARAMS = {
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
    "max_new_tokens": 2048,
    "subtalker_temperature": 0.9,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
}

PARAM_PRESETS = {
    "fast": {
        "temperature": 0.7,
        "top_k": 30,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "max_new_tokens": 1024,
        "subtalker_temperature": 0.7,
        "subtalker_top_k": 30,
        "subtalker_top_p": 0.9,
    },
    "balanced": DEFAULT_PARAMS.copy(),
    "quality": {
        "temperature": 1.0,
        "top_k": 80,
        "top_p": 1.0,
        "repetition_penalty": 1.1,
        "max_new_tokens": 4096,
        "subtalker_temperature": 1.0,
        "subtalker_top_k": 80,
        "subtalker_top_p": 1.0,
    },
}

PARAM_TOOLTIPS = {
    "temperature": "Lower = consistent pronunciation, Higher = varied intonation. Natural speech: 0.7-0.9, Precise reading: 0.3-0.5",
    "top_k": "Number of candidates for next token. Lower = stable, Higher = diverse. Recommended: 30-50",
    "top_p": "Probability-based token selection range. 1.0 = full range, lower = more certain. Recommended: 0.9-1.0",
    "repetition_penalty": "Prevents sound/word repetition. 1.0 = no penalty, higher = less repetition. Recommended: 1.0-1.1",
    "max_new_tokens": "Auto-calculated from text length at generation time. Shown value is for reference only",
    "subtalker_temperature": "Voice rhythm/accent control. Default recommended, adjust if needed",
    "subtalker_top_k": "Intonation diversity control. Default recommended",
    "subtalker_top_p": "Intonation selection range. Default recommended",
}

MAX_CHARS = 2000
CHAR_WARNING_THRESHOLD = 1500


def _prompt_to_cpu(prompt_items):
    """Move voice_clone_prompt tensors to CPU to reduce MPS memory pressure."""
    if prompt_items is None:
        return None
    out = []
    for it in prompt_items:
        try:
            new_item = type(it)(
                ref_code=None if it.ref_code is None else it.ref_code.detach().cpu(),
                ref_spk_embedding=it.ref_spk_embedding.detach().cpu(),
                x_vector_only_mode=it.x_vector_only_mode,
                icl_mode=it.icl_mode,
                ref_text=getattr(it, "ref_text", None),
            )
            out.append(new_item)
        except Exception:
            out.append(it)
    return out


def _guess_script_language(text: str | None) -> str | None:
    """Best-effort script guess from Unicode codepoints.

    Returns one of: korean, japanese, cjk, russian, latin, or None.

    Notes:
    - "japanese" is only returned when Kana is present (strong indicator).
    - Han-only text is treated as "cjk" (ambiguous between zh/ja). We avoid
      using cjk vs japanese mismatches to auto-toggle ICL.
    """

    if not text:
        return None
    t = text.strip()
    if not t:
        return None

    counts: dict[str, int] = {
        "korean": 0,
        "japanese": 0,
        "cjk": 0,
        "russian": 0,
        "latin": 0,
    }

    for ch in t:
        if ch.isspace() or ch.isdigit():
            continue
        o = ord(ch)

        # Hangul syllables
        if 0xAC00 <= o <= 0xD7A3:
            counts["korean"] += 1
            continue
        # Hiragana / Katakana (incl. extensions)
        if (0x3040 <= o <= 0x30FF) or (0x31F0 <= o <= 0x31FF):
            counts["japanese"] += 1
            continue
        # CJK Unified Ideographs (rough)
        if 0x4E00 <= o <= 0x9FFF:
            counts["cjk"] += 1
            continue
        # Cyrillic
        if 0x0400 <= o <= 0x04FF:
            counts["russian"] += 1
            continue
        # Basic Latin / Latin-1 supplement / Latin Extended
        if (0x0041 <= o <= 0x007A) or (0x00C0 <= o <= 0x024F):
            counts["latin"] += 1
            continue
        # ignore punctuation/symbols/other scripts

    total = sum(counts.values())
    if total < 2:
        return None

    # Kana presence is a strong Japanese signal even in mixed Kanji/Kana text.
    if counts["japanese"] >= 1 and counts["japanese"] / total >= 0.2:
        return "japanese"

    # Otherwise require a clear majority.
    best = max(counts, key=lambda k: counts[k])
    best_ratio = counts[best] / total if total else 0.0

    # Latin is easy to appear as acronyms inside CJK; be stricter.
    if best == "latin" and (counts[best] < 3 or best_ratio < 0.7):
        return None
    if best_ratio < 0.6:
        return None

    return best


def _ui_language_to_script(lang: str | None) -> str | None:
    if not lang:
        return None
    l = str(lang).strip().lower()
    if not l or l == "auto":
        return None
    if l == "korean":
        return "korean"
    if l == "japanese":
        return "japanese"
    if l == "chinese":
        return "cjk"
    if l == "russian":
        return "russian"
    # Treat remaining supported languages as Latin-script.
    if l in {
        "english",
        "french",
        "german",
        "italian",
        "portuguese",
        "spanish",
    }:
        return "latin"
    return None


def _should_use_xvector_only(
    *,
    ref_text: str | None,
    out_language: str | None,
    out_text: str | None,
) -> bool:
    """Decide whether to disable ICL (use x-vector only) for cross-lingual output."""

    ref_script = _guess_script_language(ref_text)

    out_script = _ui_language_to_script(out_language)
    if out_script is None:
        out_script = _guess_script_language(out_text)

    if ref_script is None or out_script is None:
        return False

    # Avoid auto-toggling between Han-only and Japanese; Han is ambiguous.
    if (ref_script == "cjk" and out_script == "japanese") or (
        ref_script == "japanese" and out_script == "cjk"
    ):
        return False

    return ref_script != out_script


def _should_use_xvector_only_multi(
    *,
    ref_texts: list[str | None],
    out_language: str | None,
    out_text: str | None,
) -> bool:
    out_script = _ui_language_to_script(out_language)
    if out_script is None:
        out_script = _guess_script_language(out_text)
    if out_script is None:
        return False

    ref_scripts = []
    for t in ref_texts:
        s = _guess_script_language(t)
        if s is not None:
            ref_scripts.append(s)

    if not ref_scripts:
        return False

    # Conservative: if references are mixed and any differs from output, disable ICL.
    for rs in ref_scripts:
        if (rs == "cjk" and out_script == "japanese") or (
            rs == "japanese" and out_script == "cjk"
        ):
            continue
        if rs != out_script:
            return True
    return False


def _make_xvector_only_prompt(prompt_items: Any) -> Any:
    """Return a prompt equivalent with ICL disabled.

    Keeps speaker embedding but strips ref_code/ref_text and sets flags.
    """

    if prompt_items is None:
        return None

    # Dict-shaped prompts (rare in this repo, but handle defensively)
    if isinstance(prompt_items, dict):
        out_dict = prompt_items.copy()
        if "ref_code" in out_dict:
            out_dict["ref_code"] = None
        if "ref_text" in out_dict:
            out_dict["ref_text"] = None
        out_dict["x_vector_only_mode"] = True
        out_dict["icl_mode"] = False
        return out_dict

    prompt_items_list = (
        prompt_items if isinstance(prompt_items, list) else [prompt_items]
    )

    out: list[Any] = []
    for it in prompt_items_list:
        # Preserve unknown objects as-is.
        if it is None:
            out.append(it)
            continue

        # If it doesn't look like a qwen-tts prompt item, don't touch it.
        if not hasattr(it, "ref_spk_embedding"):
            out.append(it)
            continue

        try:
            new_it = copy.copy(it)
            if hasattr(new_it, "ref_code"):
                setattr(new_it, "ref_code", None)
            if hasattr(new_it, "ref_text"):
                setattr(new_it, "ref_text", None)
            if hasattr(new_it, "x_vector_only_mode"):
                setattr(new_it, "x_vector_only_mode", True)
            if hasattr(new_it, "icl_mode"):
                setattr(new_it, "icl_mode", False)
            out.append(new_it)
        except Exception as e:
            print(f"[Prompt] Warning: failed to strip ICL ({type(e).__name__})")
            out.append(it)

    return out if isinstance(prompt_items, list) else out[0]


def estimate_max_tokens(
    text: str,
    tokens_per_char: float = 2.5,
    safety: float = 1.3,
    min_tokens: int = 256,
    max_cap: int = 4096,
) -> int:
    """
    Estimate appropriate max_new_tokens based on text length.

    At 12Hz TTS with ~6.5 Korean chars/sec:
    - 1 char ≈ 0.15 sec of audio
    - 1 sec of audio ≈ 12 tokens
    - So 1 char ≈ 1.8-2.5 tokens (with safety margin)
    """
    import math

    char_count = len(text)
    estimated = math.ceil(char_count * tokens_per_char * safety)
    return max(min_tokens, min(estimated, max_cap))


def load_settings():
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return DEFAULT_PARAMS.copy()


def save_settings(params):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(params, f, indent=2)
    return True


def load_favorites():
    if FAVORITES_FILE.exists():
        with open(FAVORITES_FILE) as f:
            return set(json.load(f))
    return set()


def save_favorites(favorites):
    with open(FAVORITES_FILE, "w") as f:
        json.dump(list(favorites), f)


def toggle_favorite(item_id):
    favorites = load_favorites()
    if item_id in favorites:
        favorites.discard(item_id)
        status = "Removed from favorites"
    else:
        favorites.add(item_id)
        status = "Added to favorites"
    save_favorites(favorites)
    return status, format_history_for_display(), get_history_choices()


def get_audio_duration(audio_path):
    """Get duration of audio file in seconds."""
    try:
        data, sr = sf.read(audio_path)
        return len(data) / sr
    except Exception:
        return 0


def format_duration(seconds):
    """Format seconds as mm:ss or ss.xs."""
    if seconds >= 60:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"
    return f"{seconds:.1f}s"


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as human-readable string for podcast progress."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins}m"


def _podcast_completion_status(
    failed_clips: list[dict] | None, success_text: str, success_html: str
) -> tuple[str, str]:
    """Status text and HTML for a finished podcast, warning about missing lines."""
    if not failed_clips:
        return success_text, success_html
    lines = ", ".join(
        f"#{int(clip.get('index', 0)) + 1} {clip.get('speaker', '')}"
        for clip in failed_clips
    )
    text = (
        f"Podcast generated, but {len(failed_clips)} line(s) failed and are "
        f"missing from the audio: {lines}. Edit them and use 'Regenerate Audio "
        f"from Edits' to retry."
    )
    return text, f'<div style="color: #b8860b;">{html_escape.escape(text)}</div>'


_SPEAKER_TAG_RE = re.compile(r"\[([^\]\n]{1,40})\]")
_TRANSCRIPT_PREVIEW_LIMIT = 100


def _render_podcast_transcript_html(transcript_data: dict) -> str:
    """Render transcript data as styled HTML for podcast preview."""
    dialogues = transcript_data.get("dialogues", [])
    if not dialogues:
        return '<div class="empty-state">No dialogues</div>'
    speaker_index: dict[str, int] = {}
    html_parts = []
    for dlg in dialogues[:_TRANSCRIPT_PREVIEW_LIMIT]:
        name = str(dlg.get("speaker", "Speaker"))
        color = speaker_index.setdefault(name, len(speaker_index) % 4)
        initial = html_escape.escape(
            "".join(w[0] for w in name.split()[:2]).upper() or "?"
        )
        speaker = html_escape.escape(name)
        # Delivery cues such as [laughs] / [excited] become small chips.
        text = _SPEAKER_TAG_RE.sub(
            r'<span class="dlg-tag">\1</span>',
            html_escape.escape(str(dlg.get("text", ""))),
        )
        html_parts.append(
            f'<div class="dlg s{color}"><div class="dlg-avatar">{initial}</div>'
            f'<div><div class="dlg-speaker">{speaker}</div>'
            f'<div class="dlg-text">{text}</div></div></div>'
        )
    if len(dialogues) > _TRANSCRIPT_PREVIEW_LIMIT:
        html_parts.append(
            f'<div class="dlg-more">... and {len(dialogues) - _TRANSCRIPT_PREVIEW_LIMIT} more lines</div>'
        )
    return f'<div class="dlg-list">{"".join(html_parts)}</div>'


def save_to_history(
    audio_path, text, voice_info, tab_type, gen_time=None, model_name=None, params=None
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    history_id = f"{timestamp}_{tab_type}"

    item_dir = HISTORY_DIR / history_id
    item_dir.mkdir(exist_ok=True)

    audio_dest = item_dir / "audio.wav"
    shutil.copy(audio_path, audio_dest)

    duration = get_audio_duration(str(audio_dest))

    meta = {
        "id": history_id,
        "text": text[:100] + "..." if len(text) > 100 else text,
        "full_text": text,
        "voice_info": voice_info,
        "tab_type": tab_type,
        "created": datetime.now().isoformat(),
        "duration": duration,
        "generation_time": gen_time,
        "model": model_name,
        "params": params or {},
    }
    with open(item_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return str(audio_dest)


def get_history_items(
    limit=50, search_query="", favorites_only=False, tab_type_filter=None
):
    items = []
    favorites = load_favorites()

    # Collect items from generation_history (single voice generations)
    for item_dir in sorted(HISTORY_DIR.iterdir(), reverse=True):
        if item_dir.is_dir():
            meta_path = item_dir / "metadata.json"
            audio_path = item_dir / "audio.wav"
            if meta_path.exists() and audio_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    meta["audio_path"] = str(audio_path)
                    meta["is_favorite"] = meta["id"] in favorites

                    if search_query:
                        search_lower = search_query.lower()
                        text_match = search_lower in meta.get("full_text", "").lower()
                        voice_match = search_lower in meta.get("voice_info", "").lower()
                        if not (text_match or voice_match):
                            continue

                    if favorites_only and not meta["is_favorite"]:
                        continue

                    items.append(meta)

    # Collect items from podcasts directory (podcast generations)
    podcasts_dir = Path("podcasts")
    if podcasts_dir.exists():
        for podcast_dir in sorted(podcasts_dir.iterdir(), reverse=True):
            if podcast_dir.is_dir():
                meta_path = podcast_dir / "metadata.json"
                audio_path = podcast_dir / "final_podcast.mp3"
                if meta_path.exists() and audio_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                        # Ensure podcast has type="podcast"
                        meta["type"] = "podcast"
                        meta["tab_type"] = "podcast"
                        meta["audio_path"] = str(audio_path)
                        meta["id"] = podcast_dir.name
                        meta["is_favorite"] = (
                            meta.get("id", podcast_dir.name) in favorites
                        )

                        # Load podcast artifacts
                        outline_path = podcast_dir / "outline.json"
                        transcript_path = podcast_dir / "transcript.json"
                        if outline_path.exists():
                            meta["outline"] = read_json_file(outline_path)
                        if transcript_path.exists():
                            meta["transcript"] = read_json_file(transcript_path)

                        if search_query:
                            search_lower = search_query.lower()
                            topic_match = search_lower in meta.get("topic", "").lower()
                            voice_match = (
                                search_lower in str(meta.get("speakers", "")).lower()
                            )
                            if not (topic_match or voice_match):
                                continue

                        if favorites_only and not meta["is_favorite"]:
                            continue

                        items.append(meta)

    # Apply tab_type filter
    if tab_type_filter:
        if tab_type_filter == "voice":
            items = [i for i in items if i.get("tab_type", "") != "podcast"]
        else:
            items = [i for i in items if i.get("tab_type", "") == tab_type_filter]

    # Sort all items by creation time (newest first)
    items.sort(key=lambda x: x.get("created", ""), reverse=True)

    # Return limited results
    return items[:limit]


def format_history_for_display(
    search_query="", favorites_only=False, tab_type_filter=None
):
    items = get_history_items(50, search_query, favorites_only, tab_type_filter)
    if not items:
        if search_query:
            return '<div class="empty-state">No results found for your search.</div>'
        if favorites_only:
            return '<div class="empty-state">No favorites yet. Star items to save them here!</div>'
        return '<div class="empty-state">No generation history yet. Generate some audio to see it here!</div>'

    html_parts = []
    for item in items:
        created = item.get("created", "")[:19].replace("T", " ")
        tab = item.get("tab_type", "")
        item_id = item.get("id", "")
        duration = item.get("duration", 0)
        gen_time = item.get("generation_time")
        is_favorite = item.get("is_favorite", False)

        if tab == "podcast":
            icon = "🎙️"
            text_preview = item.get("topic", "Untitled Podcast")
            voice = f"{len(item.get('speakers', []))} speakers"
        else:
            icon = "🎤" if tab == "preset" else ("🎭" if tab == "clone" else "📚")
            text_preview = item.get("text", "")
            voice = item.get("voice_info", "")

        star = "⭐" if is_favorite else "☆"
        duration_str = format_duration(duration) if duration else "—"
        gen_time_str = f"{gen_time:.1f}s" if gen_time else "—"

        safe_id = html_escape.escape(str(item_id))
        safe_text = html_escape.escape(str(text_preview))
        safe_voice = html_escape.escape(str(voice))
        safe_created = html_escape.escape(str(created))

        html_parts.append(f"""
<div class="history-card" data-id="{safe_id}">
  <div class="history-card-header">
    <span class="history-icon">{icon}</span>
    <span class="history-time">{safe_created}</span>
    <span class="history-star" title="Toggle favorite">{star}</span>
  </div>
  <div class="history-text">{safe_text}</div>
  <div class="history-meta">
    <span class="history-voice">{safe_voice}</span>
    <span class="history-duration">🎵 {duration_str}</span>
    <span class="history-gentime">⏱ {gen_time_str}</span>
  </div>
  <div class="history-id">ID: {safe_id}</div>
</div>
""")

    return "".join(html_parts)


def get_history_choices(tab_type_filter=None, search_query="", favorites_only=False):
    items = get_history_items(50, search_query, favorites_only, tab_type_filter)
    choices = []
    for item in items:
        created = item.get("created", "")[:16].replace("T", " ")
        tab = item.get("tab_type", "")
        duration = item.get("duration", 0)
        dur_str = (
            f"{int(duration // 60)}:{int(duration % 60):02d}"
            if duration >= 60
            else f"{duration:.0f}s"
            if duration
            else ""
        )

        if tab == "podcast":
            icon = "🎙️"
            topic = item.get("topic", "Untitled")[:25]
            speakers = len(item.get("speakers", []))
            label = f"{icon} {created} | {speakers} speakers | {dur_str} | {topic}..."
        else:
            icon = "🎤" if tab == "preset" else ("🎭" if tab == "clone" else "📚")
            text_preview = item.get("text", "")[:25]
            voice = item.get("voice_info", "").split(" ")[0][:10]
            label = f"{icon} {created} | {voice} | {dur_str} | {text_preview}..."

        value = item["id"]
        choices.append((label, value))
    return choices


def get_history_initial(tab_type_filter=None, search_query="", favorites_only=False):
    choices = get_history_choices(tab_type_filter, search_query, favorites_only)
    if not choices:
        return choices, None, None, "", ""
    first_value = choices[0][1]
    audio, text, params = play_history_item_with_details(first_value)
    return choices, first_value, audio, text, params


def play_history_item(choice):
    if not choice:
        return None
    audio_path = HISTORY_DIR / choice / "audio.wav"
    if audio_path.exists():
        return str(audio_path)
    podcast_audio_path = Path("podcasts") / choice / "final_podcast.mp3"
    if podcast_audio_path.exists():
        return str(podcast_audio_path)
    return None


def play_history_item_with_details(choice):
    if not choice:
        return None, "", ""

    history_audio_path = HISTORY_DIR / choice / "audio.wav"
    history_meta_path = HISTORY_DIR / choice / "metadata.json"

    podcast_dir = Path("podcasts") / choice
    podcast_audio_path = podcast_dir / "final_podcast.mp3"
    podcast_meta_path = podcast_dir / "metadata.json"

    audio = None
    full_text = ""
    params_str = ""

    if history_meta_path.exists():
        audio = str(history_audio_path) if history_audio_path.exists() else None
        with open(history_meta_path) as f:
            meta = json.load(f)

        full_text = meta.get("full_text", "")
        params = meta.get("params", {})
        model = meta.get("model", "")
        duration = meta.get("duration", 0)
        gen_time = meta.get("generation_time", 0)
        max_tokens = params.get("max_new_tokens", 0) if params else 0

        info_parts = []
        if model:
            info_parts.append(f"Model: {model}")
        if duration:
            dur_str = (
                f"{int(duration // 60)}:{int(duration % 60):02d}"
                if duration >= 60
                else f"{duration:.1f}s"
            )
            info_parts.append(f"Duration: {dur_str}")
        if gen_time:
            info_parts.append(f"GenTime: {gen_time:.1f}s")
        if max_tokens:
            info_parts.append(f"Tokens: {max_tokens}")

        if params:
            param_names = {
                "temperature": "T",
                "top_k": "K",
                "top_p": "P",
                "repetition_penalty": "Rep",
            }
            for key, label in param_names.items():
                if key in params and params[key] is not None:
                    val = params[key]
                    info_parts.append(
                        f"{label}:{val:.2f}"
                        if isinstance(val, float)
                        else f"{label}:{val}"
                    )

        params_str = " | ".join(info_parts) if info_parts else "No info recorded"

    elif podcast_meta_path.exists():
        audio = str(podcast_audio_path) if podcast_audio_path.exists() else None
        with open(podcast_meta_path) as f:
            meta = json.load(f)

        topic = meta.get("topic", "")
        speakers = meta.get("speakers", [])
        duration = meta.get("duration", 0)

        info_parts = []
        info_parts.append(f"Topic: {topic}")
        if speakers:
            speaker_names = (
                [s.get("name", "Unknown") for s in speakers]
                if isinstance(speakers, list)
                else []
            )
            info_parts.append(f"Speakers: {', '.join(speaker_names)}")
        if duration:
            dur_str = (
                f"{int(duration // 60)}:{int(duration % 60):02d}"
                if duration >= 60
                else f"{duration:.1f}s"
            )
            info_parts.append(f"Duration: {dur_str}")

        full_text = topic
        params_str = " | ".join(info_parts) if info_parts else "No info recorded"

    return audio, full_text, params_str


def get_history_item_details(choice):
    if not choice:
        return "", "", ""

    history_meta_path = HISTORY_DIR / choice / "metadata.json"
    podcast_meta_path = Path("podcasts") / choice / "metadata.json"

    if history_meta_path.exists():
        with open(history_meta_path) as f:
            meta = json.load(f)
            params = meta.get("params", {})
            model = meta.get("model", "")

            if params:
                params_lines = [f"Model: {model}"] if model else []
                param_names = {
                    "temperature": "Temp",
                    "top_k": "Top-K",
                    "top_p": "Top-P",
                    "repetition_penalty": "Rep.Pen",
                    "max_new_tokens": "MaxTok",
                    "subtalker_temperature": "SubTemp",
                    "subtalker_top_k": "SubTop-K",
                    "subtalker_top_p": "SubTop-P",
                    "speaker": "Speaker",
                    "language": "Lang",
                }
                for key, label in param_names.items():
                    if key in params and params[key] is not None:
                        val = params[key]
                        if isinstance(val, float):
                            params_lines.append(f"{label}: {val:.2f}")
                        else:
                            params_lines.append(f"{label}: {val}")
                params_str = (
                    " | ".join(params_lines) if params_lines else "No params recorded"
                )
            else:
                params_str = "No params recorded"

            return meta.get("full_text", ""), meta.get("voice_info", ""), params_str

    elif podcast_meta_path.exists():
        with open(podcast_meta_path) as f:
            meta = json.load(f)
            topic = meta.get("topic", "")
            speakers = meta.get("speakers", [])

            speaker_names = []
            if isinstance(speakers, list):
                speaker_names = [s.get("name", "Unknown") for s in speakers]

            params_lines = [f"Topic: {topic}"]
            if speaker_names:
                params_lines.append(f"Speakers: {', '.join(speaker_names)}")

            params_str = (
                " | ".join(params_lines) if params_lines else "No info recorded"
            )
            return topic, ", ".join(speaker_names), params_str

    return "", "", ""


def apply_history_params(choice):
    if not choice:
        return tuple([gr.update()] * 8) + ("Select an item first",)

    meta_path = HISTORY_DIR / choice / "metadata.json"

    if not meta_path.exists():
        return tuple([gr.update()] * 8) + ("Item not found",)

    with open(meta_path) as f:
        meta = json.load(f)

    params = meta.get("params", {})
    if not params:
        return tuple([gr.update()] * 8) + ("No params recorded for this item",)

    return (
        params.get("temperature", gr.update()),
        params.get("top_k", gr.update()),
        params.get("top_p", gr.update()),
        params.get("repetition_penalty", gr.update()),
        params.get("max_new_tokens", gr.update()),
        params.get("subtalker_temperature", gr.update()),
        params.get("subtalker_top_k", gr.update()),
        params.get("subtalker_top_p", gr.update()),
        f"✓ Applied params from {choice[:20]}...",
    )


def delete_history_item(choice, confirm_state=False):
    if not choice:
        return "Select an item first", gr.update(), gr.update(), False

    if not confirm_state:
        gr.Warning(f"⚠️ Click Delete again to confirm deletion of '{choice}'")
        return (
            f"⚠️ Click Delete again to confirm deletion of '{choice}'",
            gr.update(),
            gr.update(),
            True,
        )

    history_item_dir = HISTORY_DIR / choice
    podcast_item_dir = Path("podcasts") / choice

    deleted = False
    if history_item_dir.exists():
        shutil.rmtree(history_item_dir)
        deleted = True

    if podcast_item_dir.exists():
        shutil.rmtree(podcast_item_dir)
        deleted = True

    if deleted:
        return (
            "Deleted",
            gr.update(choices=get_history_choices(), value=None),
            None,
            False,
        )
    return "Item not found", gr.update(), gr.update(), False


def clear_all_history():
    count = 0
    for item_dir in HISTORY_DIR.iterdir():
        if item_dir.is_dir():
            shutil.rmtree(item_dir)
            count += 1
    return (
        f"✓ Cleared {count} items",
        gr.update(choices=[], value=None),
        None,
        gr.update(value=format_history_for_display()),
    )


def export_history_to_zip():
    """Export all history items to a ZIP file."""
    items = get_history_items(100)
    if not items:
        return None, "No history to export"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = f"/tmp/tts_history_export_{timestamp}.zip"

    with zipfile.ZipFile(zip_path, "w") as zf:
        for item in items:
            item_id = item["id"]
            audio_path = item.get("audio_path")
            if audio_path and os.path.exists(audio_path):
                zf.write(audio_path, f"{item_id}/audio.wav")
                meta_content = json.dumps(item, indent=2, ensure_ascii=False)
                zf.writestr(f"{item_id}/metadata.json", meta_content)

    return zip_path, f"✓ Exported {len(items)} items"


def get_podcast_history_items(limit=20):
    """Get podcast-specific history items."""
    items = []
    podcasts_dir = Path("podcasts")
    if podcasts_dir.exists():
        for podcast_dir in sorted(podcasts_dir.iterdir(), reverse=True):
            if podcast_dir.is_dir():
                meta_path = podcast_dir / "metadata.json"
                audio_path = podcast_dir / "final_podcast.mp3"
                if meta_path.exists() and audio_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                        meta["id"] = podcast_dir.name
                        meta["audio_path"] = str(audio_path)
                        items.append(meta)
    return items[:limit]


def get_podcast_history_choices():
    """Get podcast history as dropdown choices."""
    items = get_podcast_history_items(20)
    choices = []
    for item in items:
        created = item.get("created", "")[:16].replace("T", " ")
        topic = item.get("topic", "Untitled")[:30]
        speakers = item.get("speakers", [])
        speaker_count = len(speakers) if speakers else 0
        duration = item.get("duration", 0)
        dur_str = (
            f"{int(duration // 60)}:{int(duration % 60):02d}"
            if duration >= 60
            else f"{duration:.0f}s"
            if duration
            else ""
        )
        label = f"🎙️ {created} | {speaker_count} voices | {dur_str} | {topic}..."
        choices.append((label, item["id"]))
    return choices


def get_podcast_history_initial():
    """Get initial podcast history state."""
    choices = get_podcast_history_choices()
    if not choices:
        return choices, None, None, ""
    first_value = choices[0][1]
    audio, metadata = load_podcast_history_item(first_value)
    return choices, first_value, audio, metadata


def load_podcast_history_item(podcast_id):
    """Load a podcast from history."""
    if not podcast_id:
        return None, ""

    podcast_dir = Path("podcasts") / podcast_id
    audio_path = podcast_dir / "final_podcast.mp3"
    meta_path = podcast_dir / "metadata.json"

    audio = str(audio_path) if audio_path.exists() else None
    metadata = ""

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            topic = meta.get("topic", "")
            speakers = meta.get("speakers", [])
            duration = meta.get("duration", 0)
            created = meta.get("created", "")[:19].replace("T", " ")

            speaker_names = (
                [s.get("name", "Unknown") for s in speakers]
                if isinstance(speakers, list)
                else []
            )
            dur_str = (
                f"{int(duration // 60)}:{int(duration % 60):02d}"
                if duration >= 60
                else f"{duration:.1f}s"
                if duration
                else "—"
            )

            metadata = f"Topic: {topic}\nVoices: {', '.join(speaker_names)}\nDuration: {dur_str}\nCreated: {created}"

    return audio, metadata


def delete_podcast_history_item(podcast_id, confirm_state=False):
    """Delete a podcast from history."""
    if not podcast_id:
        return "Select a podcast first", gr.update(), gr.update(), False

    if not confirm_state:
        gr.Warning(f"⚠️ Click Delete again to confirm deletion of '{podcast_id}'")
        return (
            f"⚠️ Click Delete again to confirm deletion of '{podcast_id}'",
            gr.update(),
            gr.update(),
            True,
        )

    podcast_dir = Path("podcasts") / podcast_id
    if podcast_dir.exists():
        shutil.rmtree(podcast_dir)
        return (
            "Deleted",
            gr.update(choices=get_podcast_history_choices(), value=None),
            None,
            False,
        )
    return "Podcast not found", gr.update(), gr.update(), False


def get_saved_voices():
    voices = []
    for voice_dir in SAVED_VOICES_DIR.iterdir():
        if voice_dir.is_dir():
            meta_path = voice_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    meta["id"] = voice_dir.name
                    voices.append(meta)
    return sorted(voices, key=lambda x: x.get("created", ""), reverse=True)


def get_saved_voice_choices():
    voices = get_saved_voices()
    choices = []
    for v in voices:
        vid = v["id"]
        name = v.get("name", vid)
        model = v.get("model", "")
        desc = v.get("description", "")
        parts = [name]
        if model:
            parts.append(model)
        if desc:
            for ch in '\n\r|':
                desc = desc.replace(ch, ' ')
            desc = ' '.join(desc.split())
            if len(desc) > 30:
                parts.append(desc[:30] + '...')
            else:
                parts.append(desc)
        label = " | ".join(parts)
        choices.append((label, vid))
    return choices


def update_char_count(text):
    count = len(text)
    if count > MAX_CHARS:
        return f'<span class="char-count char-error">{count:,} / {MAX_CHARS:,} characters (too long)</span>'
    elif count > CHAR_WARNING_THRESHOLD:
        return f'<span class="char-count char-warning">{count:,} / {MAX_CHARS:,} characters</span>'
    else:
        return f'<span class="char-count">{count:,} / {MAX_CHARS:,} characters</span>'


def generate_custom_voice(
    text,
    model_name,
    speaker,
    language,
    instruct,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    max_new_tokens,
    sub_temp,
    sub_top_k,
    sub_top_p,
    progress=gr.Progress(),
):
    if not text.strip():
        raise gr.Error("Please enter text to generate")

    if len(text) > MAX_CHARS:
        raise gr.Error(f"Text too long ({len(text)} chars). Maximum is {MAX_CHARS}.")

    save_settings(
        {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "subtalker_temperature": sub_temp,
            "subtalker_top_k": sub_top_k,
            "subtalker_top_p": sub_top_p,
        }
    )

    start_time = time.time()
    char_count = len(text)
    auto_max_tokens = estimate_max_tokens(text)
    est_time = max(10, char_count * 0.15)

    wavs = None
    try:
        progress(0.1, desc="Loading model...")
        model = get_model(model_name)
        load_time = time.time() - start_time

        progress(
            0.2,
            desc=f"Model loaded ({load_time:.1f}s). Generating ~{est_time:.0f}s for {char_count} chars (max {auto_max_tokens} tokens)...",
        )

        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct if instruct and instruct.strip() else None,
            non_streaming_mode=True,
            temperature=temperature,
            top_k=int(top_k),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=auto_max_tokens,
            subtalker_temperature=sub_temp,
            subtalker_top_k=int(sub_top_k),
            subtalker_top_p=sub_top_p,
        )

        gen_time = time.time() - start_time

        progress(0.9, desc=f"Saving audio ({gen_time:.1f}s)...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wavs[0], sr)

            history_path = save_to_history(
                f.name,
                text,
                f"{speaker} ({model_name})",
                "preset",
                gen_time,
                model_name=model_name,
                params={
                    "temperature": temperature,
                    "top_k": int(top_k),
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "max_new_tokens": auto_max_tokens,
                    "subtalker_temperature": sub_temp,
                    "subtalker_top_k": int(sub_top_k),
                    "subtalker_top_p": sub_top_p,
                    "speaker": speaker,
                    "language": language,
                    "instruct": instruct if instruct and instruct.strip() else None,
                },
            )

            duration = get_audio_duration(history_path)
            status = f"Done in {gen_time:.1f}s | Duration: {format_duration(duration)} | Tokens: {auto_max_tokens} • Saved to History ✓"

            progress(1.0, desc="Complete!")
            return history_path, status
    except Exception as e:
        gr.Warning(format_user_error(e))
        return None, f"❌ Error: {format_user_error(e)}"
    finally:
        del wavs
        _gpu_cleanup()


def generate_voice_design(
    text,
    voice_description,
    language,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    max_new_tokens,
    sub_temp,
    sub_top_k,
    sub_top_p,
    progress=gr.Progress(),
):
    if not text.strip():
        raise gr.Error("Please enter text to generate")

    if not voice_description.strip():
        raise gr.Error("Please enter a voice description")

    MAX_VOICE_DESC_LENGTH = 500
    if len(voice_description) > MAX_VOICE_DESC_LENGTH:
        raise gr.Error(
            f"Voice description too long ({len(voice_description)} chars). Maximum is {MAX_VOICE_DESC_LENGTH}."
        )

    if len(text) > MAX_CHARS:
        raise gr.Error(f"Text too long ({len(text)} chars). Maximum is {MAX_CHARS}.")

    save_settings(
        {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "subtalker_temperature": sub_temp,
            "subtalker_top_k": sub_top_k,
            "subtalker_top_p": sub_top_p,
        }
    )

    start_time = time.time()
    char_count = len(text)
    auto_max_tokens = estimate_max_tokens(text)
    est_time = max(10, char_count * 0.15)

    wavs = None
    try:
        progress(0.1, desc="Loading VoiceDesign model...")
        model = get_model("1.7B-VoiceDesign")

        if not hasattr(model, "generate_voice_design"):
            raise gr.Error(
                "VoiceDesign model doesn't support generate_voice_design(). "
                "Please ensure you have the correct model downloaded."
            )

        load_time = time.time() - start_time

        progress(
            0.2,
            desc=f"Model loaded ({load_time:.1f}s). Generating ~{est_time:.0f}s for {char_count} chars...",
        )

        actual_language = language if language and language != "auto" else None

        gen_kwargs = {
            "text": text,
            "instruct": voice_description.strip(),
            "non_streaming_mode": True,
            "temperature": temperature,
            "top_k": int(top_k),
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": auto_max_tokens,
            "subtalker_temperature": sub_temp,
            "subtalker_top_k": int(sub_top_k),
            "subtalker_top_p": sub_top_p,
        }
        if actual_language:
            gen_kwargs["language"] = actual_language

        wavs, sr = model.generate_voice_design(**gen_kwargs)

        # Guard against empty result
        if wavs is None or len(wavs) == 0:
            raise gr.Error(
                "Model returned empty audio. Try different parameters or voice description."
            )

        gen_time = time.time() - start_time

        progress(0.9, desc=f"Saving audio ({gen_time:.1f}s)...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, wavs[0], sr)

        try:
            history_path = save_to_history(
                temp_path,
                text,
                f"VoiceDesign",
                "design",
                gen_time,
                model_name="1.7B-VoiceDesign",
                params={
                    "temperature": temperature,
                    "top_k": int(top_k),
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "max_new_tokens": auto_max_tokens,
                    "subtalker_temperature": sub_temp,
                    "subtalker_top_k": int(sub_top_k),
                    "subtalker_top_p": sub_top_p,
                    "language": actual_language,
                    "voice_description": voice_description.strip(),
                },
            )

            duration = get_audio_duration(history_path)
            status = f"Done in {gen_time:.1f}s | Duration: {format_duration(duration)} | Tokens: {auto_max_tokens} • Saved to History ✓"

            progress(1.0, desc="Complete!")
            return history_path, status
        finally:
            # Clean up temp file to prevent leak
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    except Exception as e:
        gr.Warning(format_user_error(e))
        return None, f"❌ Error: {format_user_error(e)}"
    finally:
        del wavs
        _gpu_cleanup()


def clone_voice_multi(
    audio_files: list,
    transcripts_json: str,
    model_name: str,
    test_text: str,
    language: str,
    ref_language: str,
    crosslingual_opt: bool,
    combine_samples: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    sub_temp: float,
    sub_top_k: int,
    sub_top_p: float,
    progress=gr.Progress(),
):
    if not audio_files:
        raise gr.Error("Please upload at least one reference audio sample")

    if len(audio_files) > 3:
        raise gr.Error("Maximum 3 audio samples allowed")

    try:
        transcripts = json.loads(transcripts_json) if transcripts_json else {}
    except json.JSONDecodeError:
        transcripts = {}

    audio_paths = []
    audio_transcripts = []
    for af in audio_files[:3]:
        if af is None:
            continue
        path = extract_file_path(af)
        if not path:
            continue
        audio_paths.append(path)
        audio_transcripts.append(transcripts.get(path, ""))

    if not audio_paths:
        raise gr.Error("No valid audio files provided")

    has_transcripts = any(t.strip() for t in audio_transcripts)
    x_vector_only_mode = not has_transcripts

    save_settings(
        {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "subtalker_temperature": sub_temp,
            "subtalker_top_k": sub_top_k,
            "subtalker_top_p": sub_top_p,
        }
    )

    start_time = time.time()
    wavs = None
    voice_clone_prompt = None

    try:
        progress(0.1, desc="Loading model...")
        model = get_model(model_name)
        load_time = time.time() - start_time

        progress(
            0.2,
            desc=f"Model loaded ({load_time:.1f}s). Analyzing {len(audio_paths)} sample(s)...",
        )

        sample_infos = analyze_audio_samples(audio_paths, audio_transcripts)

        if combine_samples and len(sample_infos) > 1:
            progress(0.3, desc="Combining voice embeddings...")
            voice_clone_prompt = create_combined_voice_clone_prompt(
                model=model,
                sample_infos=sample_infos,
                x_vector_only_mode=x_vector_only_mode,
            )
            primary_info = next(
                (s for s in sample_infos if s.is_primary), sample_infos[0]
            )
        else:
            primary_info = sample_infos[0]
            voice_clone_prompt = model.create_voice_clone_prompt(
                ref_audio=primary_info.path,
                ref_text=primary_info.transcript,
                x_vector_only_mode=x_vector_only_mode,
            )

        # Keep the stored prompt as-is, but optionally disable ICL at generation
        # time for cross-lingual output when ICL is available.
        runtime_prompt = voice_clone_prompt
        runtime_xvector_only = bool(x_vector_only_mode)
        if test_text.strip() and not x_vector_only_mode and bool(crosslingual_opt):
            ref_texts = [
                s.transcript for s in sample_infos if getattr(s, "transcript", None)
            ]
            if not ref_texts and primary_info is not None:
                ref_texts = [getattr(primary_info, "transcript", None)]

            try:
                ref_lang = (ref_language or "").strip().lower()
                out_lang = (language or "").strip().lower()
                if ref_lang and ref_lang != "auto" and out_lang and out_lang != "auto":
                    runtime_xvector_only = ref_lang != out_lang
                else:
                    runtime_xvector_only = _should_use_xvector_only_multi(
                        ref_texts=ref_texts,
                        out_language=language,
                        out_text=test_text,
                    )
            except Exception:
                runtime_xvector_only = False

            if runtime_xvector_only:
                runtime_prompt = _make_xvector_only_prompt(voice_clone_prompt)

        output_audio = None
        if test_text.strip():
            if len(test_text) > MAX_CHARS:
                raise gr.Error(f"Test text too long ({len(test_text)} chars)")

            char_count = len(test_text)
            auto_max_tokens = estimate_max_tokens(test_text)
            est_time = max(10, char_count * 0.15)
            progress(0.4, desc=f"Generating ~{est_time:.0f}s for {char_count} chars...")

            wavs, sr = model.generate_voice_clone(
                text=test_text,
                language=language,
                voice_clone_prompt=runtime_prompt,
                non_streaming_mode=True,
                temperature=temperature,
                top_k=int(top_k),
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=auto_max_tokens,
                subtalker_temperature=sub_temp,
                subtalker_top_k=int(sub_top_k),
                subtalker_top_p=sub_top_p,
            )

            gen_time = time.time() - start_time
            progress(0.9, desc=f"Saving audio ({gen_time:.1f}s)...")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, wavs[0], sr)
                n_samples = len(sample_infos) if combine_samples else 1
                output_audio = save_to_history(
                    f.name,
                    test_text,
                    f"Clone ({model_name}, {n_samples} samples)",
                    "clone",
                    gen_time,
                    model_name=model_name,
                    params={
                        "temperature": temperature,
                        "top_k": int(top_k),
                        "top_p": top_p,
                        "repetition_penalty": repetition_penalty,
                        "max_new_tokens": auto_max_tokens,
                        "subtalker_temperature": sub_temp,
                        "subtalker_top_k": int(sub_top_k),
                        "subtalker_top_p": sub_top_p,
                        "language": language,
                        "ref_language": ref_language,
                        "crosslingual_opt": bool(crosslingual_opt),
                        "num_samples": n_samples,
                        "x_vector_only_mode_runtime": runtime_xvector_only,
                    },
                )
            duration = get_audio_duration(output_audio)
            audio_info = (
                f" | Duration: {format_duration(duration)} | Tokens: {auto_max_tokens}"
            )
        else:
            gen_time = time.time() - start_time
            audio_info = ""

        prompt_temp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        cpu_prompt = _prompt_to_cpu(voice_clone_prompt)
        with open(prompt_temp.name, "wb") as f:
            pickle.dump(cpu_prompt, f)

        samples_meta = [
            {
                "path": s.path,
                "duration": float(s.duration),
                "transcript": s.transcript,
                "snr_estimate": float(s.snr_estimate)
                if s.snr_estimate is not None
                else None,
                "is_primary": s.is_primary,
                "weight": float(s.weight),
            }
            for s in sample_infos
        ]
        samples_meta_json = json.dumps(samples_meta)

        n_samples = len(sample_infos) if combine_samples else 1
        total_duration = sum(s.duration for s in sample_infos)
        progress(1.0, desc="Complete!")
        status = f"Done in {gen_time:.1f}s{audio_info} | {n_samples} sample(s), {total_duration:.1f}s total ref audio"

        return output_audio, prompt_temp.name, model_name, samples_meta_json, status
    except Exception as e:
        traceback.print_exc()
        gr.Warning(format_user_error(e))
        return None, gr.skip(), gr.skip(), gr.skip(), f"❌ Error: {format_user_error(e)}"
    finally:
        del wavs
        del voice_clone_prompt
        _gpu_cleanup()


def analyze_uploaded_samples(audio_files: list) -> tuple[str, str]:
    if not audio_files:
        return "", ""

    audio_paths = []
    for af in audio_files[:3]:
        if af is None:
            continue
        path = af if isinstance(af, str) else af.name
        audio_paths.append(path)

    if not audio_paths:
        return "", ""

    sample_infos = analyze_audio_samples(audio_paths)
    summary = format_samples_summary(sample_infos)
    warnings = get_sample_warnings(sample_infos)

    if len(audio_files) > 3:
        warnings.insert(0, "Only the first 3 samples will be used.")

    warnings_html = "<br>".join(warnings) if warnings else ""

    return summary, warnings_html


def extract_file_path(file_obj) -> str | None:
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name")
    return getattr(file_obj, "path", None) or getattr(file_obj, "name", None)


def flush_transcripts_to_state(
    transcript_state: dict,
    current_paths: list,
    t1: str,
    t2: str,
    t3: str,
) -> dict:
    if not current_paths:
        return transcript_state if isinstance(transcript_state, dict) else {}

    new_state = dict(transcript_state) if isinstance(transcript_state, dict) else {}
    transcripts = [t1, t2, t3]

    for i, path in enumerate(current_paths[:3]):
        if path and i < len(transcripts):
            new_state[path] = transcripts[i] or ""

    return new_state


def update_transcript_fields(audio_files: list, transcript_state: dict):
    empty_result = (
        gr.update(value="*Upload audio samples to enter transcripts.*"),
        gr.update(visible=False, label="Sample 1 (Primary)", value=""),
        gr.update(visible=False, label="Sample 2", value=""),
        gr.update(visible=False, label="Sample 3", value=""),
        gr.update(visible=False),
        transcript_state if isinstance(transcript_state, dict) else {},
        [],
    )

    if not audio_files:
        return empty_result

    file_paths = []
    file_names = []
    for af in audio_files[:3]:
        path = extract_file_path(af)
        if path:
            file_paths.append(path)
            file_names.append(Path(path).name)

    num_files = len(file_paths)
    if num_files == 0:
        return empty_result

    prev_state = transcript_state if isinstance(transcript_state, dict) else {}

    transcript_values = []
    for path in file_paths:
        transcript_values.append(prev_state.get(path, ""))

    new_state = dict(prev_state)
    for i, path in enumerate(file_paths):
        if path not in new_state:
            new_state[path] = transcript_values[i]

    max_state_entries = 20
    if len(new_state) > max_state_entries:
        current_set = set(file_paths)
        keys_to_remove = [k for k in new_state if k not in current_set]
        for k in keys_to_remove[: len(new_state) - max_state_entries]:
            del new_state[k]

    info_text = f"*Enter transcript for each sample. {num_files} file(s) uploaded.*"

    t1_update = gr.update(
        visible=True,
        label=f"Sample 1 (Primary): {file_names[0]}",
        value=transcript_values[0] if num_files >= 1 else "",
    )
    t2_update = gr.update(
        visible=num_files >= 2,
        label=f"Sample 2: {file_names[1]}" if num_files >= 2 else "Sample 2",
        value=transcript_values[1] if num_files >= 2 else "",
    )
    t3_update = gr.update(
        visible=num_files >= 3,
        label=f"Sample 3: {file_names[2]}" if num_files >= 3 else "Sample 3",
        value=transcript_values[2] if num_files >= 3 else "",
    )
    btn_update = gr.update(visible=num_files >= 1)

    return info_text, t1_update, t2_update, t3_update, btn_update, new_state, file_paths


def save_cloned_voice_multi(
    voice_name: str,
    description: str,
    style_note: str,
    ref_language: str,
    audio_files: list,
    transcripts_json: str,
    prompt_path: str,
    model_name: str,
    samples_meta_json: str,
):
    if not voice_name.strip():
        gr.Warning("Please enter a name for this voice")
        return "Please enter a name for this voice", gr.update()
    if not prompt_path or not os.path.exists(prompt_path):
        gr.Warning("Clone a voice first before saving")
        return "Clone a voice first before saving", gr.update()

    try:
        safe_name = "".join(c for c in voice_name if c.isalnum() or c in "_-").strip()
        if not safe_name or safe_name in (".", ".."):
            gr.Warning(
                "Invalid voice name - use only letters, numbers, underscores, hyphens"
            )
            return "Invalid voice name", gr.update()

        voice_dir = (SAVED_VOICES_DIR / safe_name).resolve()
        try:
            voice_dir.relative_to(SAVED_VOICES_DIR.resolve())
        except ValueError:
            gr.Warning("Invalid voice name")
            return "Invalid voice name", gr.update()

        voice_dir.mkdir(exist_ok=True)
        shutil.copy(prompt_path, voice_dir / "prompt.pkl")

        try:
            transcripts = json.loads(transcripts_json) if transcripts_json else {}
        except json.JSONDecodeError:
            transcripts = {}

        try:
            samples_meta = json.loads(samples_meta_json) if samples_meta_json else []
        except json.JSONDecodeError:
            samples_meta = []

        ref_audios_dir = voice_dir / "ref_audios"
        ref_audios_dir.mkdir(exist_ok=True)
        saved_ref_paths = []

        if audio_files:
            for i, af in enumerate(audio_files):
                if af is None:
                    continue
                src_path = af if isinstance(af, str) else af.name
                fname = Path(src_path).name
                dest_path = ref_audios_dir / f"sample_{i:02d}_{fname}"
                shutil.copy(src_path, dest_path)
                saved_ref_paths.append(str(dest_path.relative_to(voice_dir)))

        primary_transcript = ""
        if samples_meta:
            primary_sample = next(
                (s for s in samples_meta if s.get("is_primary")),
                samples_meta[0] if samples_meta else None,
            )
            if primary_sample and primary_sample.get("transcript"):
                primary_transcript = primary_sample.get("transcript", "")
        if transcripts:
            primary_transcript = next(iter(transcripts.values()), "")

        ref_language_ui = (ref_language or "").strip().lower()
        if not ref_language_ui:
            ref_language_ui = "auto"
        if ref_language_ui not in set(LANGUAGES):
            ref_language_ui = "auto"

        ref_script = _guess_script_language(primary_transcript)

        metadata = {
            "name": voice_name,
            "description": description,
            "style_note": style_note,
            "ref_text": primary_transcript,
            # Back-compat: historically this stored a script guess.
            "ref_language": ref_script,
            "ref_language_script": ref_script,
            "ref_language_ui": ref_language_ui,
            "model": model_name or "1.7B-Base",
            "created": datetime.now().isoformat(),
            "multi_sample": True,
            "num_samples": len(saved_ref_paths),
            "samples": samples_meta,
            "ref_audio_paths": saved_ref_paths,
        }
        with open(voice_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        gr.Info(f"Voice '{voice_name}' saved with {len(saved_ref_paths)} sample(s)!")
        return f"Saved voice: {safe_name}", gr.update(choices=get_saved_voice_choices())
    except Exception as e:
        error_msg = format_user_error(e)
        gr.Warning(f"Failed to save voice: {error_msg}")
        return f"Error: {error_msg}", gr.update()


def save_cloned_voice(
    voice_name,
    description,
    style_note,
    ref_language,
    ref_audio,
    ref_text,
    prompt_path,
    model_name,
):
    if not voice_name.strip():
        gr.Warning("Please enter a name for this voice")
        return "Please enter a name for this voice", gr.update()
    if not prompt_path or not os.path.exists(prompt_path):
        gr.Warning("Clone a voice first before saving")
        return "Clone a voice first before saving", gr.update()

    try:
        # Strict sanitization: only alphanumeric, underscore, hyphen
        safe_name = "".join(c for c in voice_name if c.isalnum() or c in "_-").strip()
        if not safe_name or safe_name in (".", ".."):
            gr.Warning(
                "Invalid voice name - use only letters, numbers, underscores, hyphens"
            )
            return "Invalid voice name", gr.update()

        voice_dir = (SAVED_VOICES_DIR / safe_name).resolve()
        # Ensure we're still within SAVED_VOICES_DIR (proper path containment check)
        try:
            voice_dir.relative_to(SAVED_VOICES_DIR.resolve())
        except ValueError:
            gr.Warning("Invalid voice name")
            return "Invalid voice name", gr.update()

        voice_dir.mkdir(exist_ok=True)

        shutil.copy(prompt_path, voice_dir / "prompt.pkl")

        if ref_audio and isinstance(ref_audio, str):
            shutil.copy(ref_audio, voice_dir / "ref_audio.wav")

        ref_language_ui = (ref_language or "").strip().lower()
        if not ref_language_ui:
            ref_language_ui = "auto"
        if ref_language_ui not in set(LANGUAGES):
            ref_language_ui = "auto"

        ref_script = _guess_script_language(ref_text)

        metadata = {
            "name": voice_name,
            "description": description,
            "style_note": style_note,
            "ref_text": ref_text,
            # Back-compat: historically this stored a script guess.
            "ref_language": ref_script,
            "ref_language_script": ref_script,
            "ref_language_ui": ref_language_ui,
            "model": model_name or "1.7B-Base",
            "created": datetime.now().isoformat(),
        }
        with open(voice_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        gr.Info(f"✅ Voice '{voice_name}' saved successfully!")
        return f"Saved voice: {safe_name}", gr.update(choices=get_saved_voice_choices())
    except Exception as e:
        error_msg = format_user_error(e)
        gr.Warning(f"❌ Failed to save voice: {error_msg}")
        return f"Error: {error_msg}", gr.update()


def generate_with_saved_voice(
    text,
    saved_voice_id,
    language,
    crosslingual_opt,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    max_new_tokens,
    sub_temp,
    sub_top_k,
    sub_top_p,
    progress=gr.Progress(),
):
    if not text.strip():
        raise gr.Error("Please enter text to generate")
    if not saved_voice_id:
        raise gr.Error("Please select a saved voice")

    if len(text) > MAX_CHARS:
        raise gr.Error(f"Text too long ({len(text)} chars). Maximum is {MAX_CHARS}.")

    save_settings(
        {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "subtalker_temperature": sub_temp,
            "subtalker_top_k": sub_top_k,
            "subtalker_top_p": sub_top_p,
        }
    )

    start_time = time.time()
    char_count = len(text)
    auto_max_tokens = estimate_max_tokens(text)
    est_time = max(10, char_count * 0.15)
    wavs = None
    voice_clone_prompt = None

    try:
        voice_dir = SAVED_VOICES_DIR / saved_voice_id
        prompt_path = voice_dir / "prompt.pkl"
        meta_path = voice_dir / "metadata.json"

        if not prompt_path.exists():
            raise gr.Error(f"Voice not found: {saved_voice_id}")

        with open(meta_path) as f:
            meta = json.load(f)
        model_name = meta.get("model", "1.7B-Base")

        progress(0.05, desc="Loading voice profile...")

        with open(prompt_path, "rb") as f:
            voice_clone_prompt = pickle.load(f)

        # Optionally disable ICL at generation time for cross-lingual output.
        runtime_prompt = voice_clone_prompt
        runtime_xvector_only = False
        if bool(crosslingual_opt):
            try:
                ref_lang_raw = meta.get("ref_language_ui")
                if not ref_lang_raw and meta.get("ref_language") in set(LANGUAGES):
                    ref_lang_raw = meta.get("ref_language")
                ref_lang = (ref_lang_raw or "").strip().lower()
                out_lang = (language or "").strip().lower()

                if ref_lang and ref_lang != "auto" and out_lang and out_lang != "auto":
                    runtime_xvector_only = ref_lang != out_lang
                else:
                    ref_texts: list[str | None] = []
                    ref_texts.append(meta.get("ref_text"))
                    samples = meta.get("samples")
                    if isinstance(samples, list):
                        for s in samples:
                            if isinstance(s, dict):
                                ref_texts.append(s.get("transcript"))

                    runtime_xvector_only = _should_use_xvector_only_multi(
                        ref_texts=ref_texts,
                        out_language=language,
                        out_text=text,
                    )
            except Exception:
                runtime_xvector_only = False

            if runtime_xvector_only:
                runtime_prompt = _make_xvector_only_prompt(voice_clone_prompt)

        progress(0.1, desc=f"Loading {model_name}...")
        model = get_model(model_name)
        load_time = time.time() - start_time

        progress(
            0.2,
            desc=f"Model loaded ({load_time:.1f}s). Generating ~{est_time:.0f}s for {char_count} chars (max {auto_max_tokens} tokens)...",
        )

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=runtime_prompt,
            non_streaming_mode=True,
            temperature=temperature,
            top_k=int(top_k),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=auto_max_tokens,
            subtalker_temperature=sub_temp,
            subtalker_top_k=int(sub_top_k),
            subtalker_top_p=sub_top_p,
        )

        gen_time = time.time() - start_time

        progress(0.9, desc=f"Saving audio ({gen_time:.1f}s)...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wavs[0], sr)
            history_path = save_to_history(
                f.name,
                text,
                f"{saved_voice_id} ({model_name})",
                "saved",
                gen_time,
                model_name=model_name,
                params={
                    "temperature": temperature,
                    "top_k": int(top_k),
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "max_new_tokens": auto_max_tokens,
                    "subtalker_temperature": sub_temp,
                    "subtalker_top_k": int(sub_top_k),
                    "subtalker_top_p": sub_top_p,
                    "language": language,
                    "saved_voice_id": saved_voice_id,
                    "crosslingual_opt": bool(crosslingual_opt),
                    "x_vector_only_mode_runtime": runtime_xvector_only,
                },
            )

            duration = get_audio_duration(history_path)
            status = f"Done in {gen_time:.1f}s | Duration: {format_duration(duration)} | Tokens: {auto_max_tokens} • Saved to History ✓"

            progress(1.0, desc="Complete!")
            return history_path, status
    except Exception as e:
        gr.Warning(format_user_error(e))
        return None, f"❌ Error: {format_user_error(e)}"
    finally:
        del wavs
        del voice_clone_prompt
        _gpu_cleanup()


def get_voice_details(saved_voice_id):
    if not saved_voice_id:
        return "", "", "", "", "", None

    voice_dir = SAVED_VOICES_DIR / saved_voice_id
    meta_path = voice_dir / "metadata.json"

    if not meta_path.exists():
        return "Not found", "", "", "", "", None

    with open(meta_path) as f:
        meta = json.load(f)

    ref_audio_path = voice_dir / "ref_audio.wav"
    ref_audio = str(ref_audio_path) if ref_audio_path.exists() else None

    ref_lang_ui = meta.get("ref_language_ui")
    if not ref_lang_ui:
        ref_lang_ui = meta.get("ref_language")
    if not ref_lang_ui:
        ref_lang_ui = "auto"

    ref_lang_display = str(ref_lang_ui)
    if ref_lang_display not in set(LANGUAGES):
        # Back-compat where we stored a coarse script guess.
        if ref_lang_display in {"latin", "cjk", "korean", "japanese", "russian"}:
            ref_lang_display = f"auto ({ref_lang_display})"
        else:
            ref_lang_display = "auto"

    return (
        meta.get("description", ""),
        meta.get("style_note", ""),
        meta.get("ref_text", ""),
        ref_lang_display,
        meta.get("model", "Unknown"),
        ref_audio,
    )


def delete_saved_voice(saved_voice_id, confirm_state):
    if not saved_voice_id:
        return "Select a voice first", gr.update(), gr.update(), False

    if not confirm_state:
        gr.Warning(f"⚠️ Click Delete again to confirm deletion of '{saved_voice_id}'")
        return (
            f"⚠️ Click Delete again to confirm deletion of '{saved_voice_id}'",
            gr.update(),
            gr.update(),
            True,
        )

    voice_dir = SAVED_VOICES_DIR / saved_voice_id
    if voice_dir.exists():
        shutil.rmtree(voice_dir)
        gr.Info(f"✅ Voice '{saved_voice_id}' deleted successfully")
        return (
            f"✅ Deleted: {saved_voice_id}",
            gr.update(choices=get_saved_voice_choices(), value=None),
            gr.update(value=None),
            False,
        )
    return "Voice not found", gr.update(), gr.update(), False


def apply_preset(preset_name):
    if preset_name not in PARAM_PRESETS:
        return tuple([gr.update()] * 8) + ("Unknown preset",)

    p = PARAM_PRESETS[preset_name]
    save_settings(p)

    return (
        p["temperature"],
        p["top_k"],
        p["top_p"],
        p["repetition_penalty"],
        p["max_new_tokens"],
        p["subtalker_temperature"],
        p["subtalker_top_k"],
        p["subtalker_top_p"],
        f'<span class="save-indicator show">Current: {preset_name.title()} preset</span>',
    )


def reset_params():
    save_settings(DEFAULT_PARAMS)
    return tuple(DEFAULT_PARAMS.values()) + (
        '<span class="save-indicator show">Reset to defaults</span>',
    )


def apply_podcast_preset(preset_name):
    """Apply podcast quality preset and return updated parameters and num_segments."""
    if preset_name not in PODCAST_QUALITY_PRESETS:
        return tuple([gr.update()] * 8) + (2, "Unknown preset")

    p = PODCAST_QUALITY_PRESETS[preset_name]
    save_settings(
        {
            "temperature": p["temperature"],
            "top_k": p["top_k"],
            "top_p": p["top_p"],
            "repetition_penalty": p["repetition_penalty"],
            "max_new_tokens": p["max_new_tokens"],
            "subtalker_temperature": p["subtalker_temperature"],
            "subtalker_top_k": p["subtalker_top_k"],
            "subtalker_top_p": p["subtalker_top_p"],
        }
    )

    return (
        p["temperature"],
        p["top_k"],
        p["top_p"],
        p["repetition_penalty"],
        p["max_new_tokens"],
        p["subtalker_temperature"],
        p["subtalker_top_k"],
        p["subtalker_top_p"],
        p["num_segments"],
        f'<span class="save-indicator show">Applied {preset_name} preset ({p["duration_estimate"]})</span>',
    )


def update_podcast_preset_info(preset_name):
    if preset_name not in PODCAST_QUALITY_PRESETS:
        return gr.update(value="Unknown preset")
    p = PODCAST_QUALITY_PRESETS[preset_name]
    return gr.update(
        value=f'<div class="hint">{p["tooltip"]}</div>'
    )


def on_param_change(*args):
    params = {
        "temperature": args[0],
        "top_k": args[1],
        "top_p": args[2],
        "repetition_penalty": args[3],
        "max_new_tokens": args[4],
        "subtalker_temperature": args[5],
        "subtalker_top_k": args[6],
        "subtalker_top_p": args[7],
    }
    save_settings(params)
    return '<span class="save-indicator show">Settings saved</span>'


def search_history(query, favorites_only):
    """Search/filter history items."""
    return format_history_for_display(query, favorites_only)


def search_history_filtered(query, favorites_only, tab_filter):
    """Search/filter history - updates display, dropdown, and clears details."""
    filter_val = tab_filter.lower() if tab_filter and tab_filter != "All" else "voice"
    display = format_history_for_display(query, favorites_only, filter_val)
    choices = get_history_choices(filter_val, query, favorites_only)
    return display, gr.update(choices=choices, value=None), None, "", ""


def history_tab_favorite(choice, query, favorites_only, tab_filter):
    """Toggle favorite with filtered refresh for History tab."""
    if not choice:
        return "Select an item first", gr.update(), gr.update(), gr.update(), gr.update()

    toggle_favorite(choice)
    filter_val = tab_filter.lower() if tab_filter and tab_filter != "All" else "voice"
    display = format_history_for_display(query, favorites_only, filter_val)
    choices = get_history_choices(filter_val, query, favorites_only)
    selected = choice if any(value == choice for _, value in choices) else None
    if selected is None:
        return "★ Toggled favorite", display, gr.update(choices=choices, value=None), None, ""
    return "★ Toggled favorite", display, gr.update(choices=choices, value=selected), gr.update(), gr.update()


def history_tab_delete(choice, confirm_state, query, favorites_only, tab_filter):
    """Delete with filtered refresh for History tab."""
    result = delete_history_item(choice, confirm_state)
    if confirm_state and result[0] == "Deleted":
        filter_val = tab_filter.lower() if tab_filter and tab_filter != "All" else "voice"
        display = format_history_for_display(query, favorites_only, filter_val)
        choices = get_history_choices(filter_val, query, favorites_only)
        return result[0], gr.update(choices=choices, value=None), None, False, display, ""
    return result[0], result[1], result[2], result[3], gr.update(), gr.update()


def _disable_btn():
    """Return a gr.update that disables a button with a processing label."""
    return gr.update(interactive=False, value="\u23f3 Processing...")


def _enable_btn(label="Generate Speech"):
    """Return a gr.update that re-enables a button with its original label."""
    return gr.update(interactive=True, value=label)


def _refresh_history_on_success(audio_path, search, favorites, tab_filter):
    """Refresh history tab only when generation succeeded (audio_path is truthy)."""
    if not audio_path:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    return search_history_filtered(search, favorites, tab_filter)

LANGUAGES = [
    "auto",
    "chinese",
    "english",
    "french",
    "german",
    "italian",
    "japanese",
    "korean",
    "portuguese",
    "russian",
    "spanish",
]

custom_css = APP_CSS

settings = load_settings()


PRESET_VOICES = [
    {"voice_id": "aiden", "name": "Aiden", "desc": "Bright American male, clear midrange"},
    {"voice_id": "dylan", "name": "Dylan", "desc": ""},
    {"voice_id": "eric", "name": "Eric", "desc": ""},
    {"voice_id": "ono_anna", "name": "Ono Anna", "desc": "Lively Japanese female"},
    {"voice_id": "ryan", "name": "Ryan", "desc": "Dynamic male with strong rhythm"},
    {"voice_id": "serena", "name": "Serena", "desc": "Warm, soft young female"},
    {"voice_id": "sohee", "name": "Sohee", "desc": "Warm Korean female, rich emotion"},
    {"voice_id": "uncle_fu", "name": "Uncle Fu", "desc": ""},
    {"voice_id": "vivian", "name": "Vivian", "desc": "Bright, slightly sharp young female"},
]


def _get_preset_speaker_choices() -> list[tuple[str, str]]:
    """Build (label, value) tuples for the Custom Voice speaker dropdown."""
    choices = []
    for v in PRESET_VOICES:
        label = v["name"]
        if v.get("desc"):
            label += f" \u2014 {v['desc']}"
        choices.append((label, v["voice_id"]))
    return choices


def _get_podcast_voice_choices() -> list[tuple[str, str]]:
    saved = get_saved_voices()
    preset_choices = []
    for v in PRESET_VOICES:
        label = v["name"]
        if v.get("desc"):
            label += f" \u2014 {v['desc']}"
        label += " (Preset)"
        preset_choices.append((label, f"preset:{v['voice_id']}"))
    saved_choices = []
    for v in saved:
        name = v.get("name", v.get("id"))
        model = v.get("model", "")
        desc = v.get("description", "")
        parts = [name]
        if model:
            parts.append(model)
        if desc:
            for ch in '\n\r|':
                desc = desc.replace(ch, ' ')
            desc = ' '.join(desc.split())
            if len(desc) > 30:
                parts.append(desc[:30] + '...')
            else:
                parts.append(desc)
        parts.append("Saved")
        label = " | ".join(parts)
        saved_choices.append((label, f"saved:{v.get('id')}"))
    return [("-- Select --", "")] + preset_choices + saved_choices


def _get_persona_voice_choices() -> list[tuple[str, str]]:
    saved = get_saved_voices()
    preset_choices = []
    for v in PRESET_VOICES:
        label = v["name"]
        if v.get("desc"):
            label += f" \u2014 {v['desc']}"
        label += " (Preset)"
        preset_choices.append((label, f"{v['voice_id']}|preset"))
    saved_choices = []
    for v in saved:
        name = v.get("name", v.get("id"))
        model = v.get("model", "")
        desc = v.get("description", "")
        parts = [name]
        if model:
            parts.append(model)
        if desc:
            for ch in '\n\r|':
                desc = desc.replace(ch, ' ')
            desc = ' '.join(desc.split())
            if desc:
                if len(desc) > 30:
                    parts.append(desc[:30] + '...')
                else:
                    parts.append(desc)
        parts.append("Saved")
        label = " | ".join(parts)
        saved_choices.append((label, f"{v.get('id')}|saved"))
    return preset_choices + saved_choices


def _parse_persona_voice_value(value: str) -> tuple[str, str]:
    if not value or "|" not in value:
        return "", ""
    parts = value.split("|", 1)
    return parts[0], parts[1]


def _render_persona_cards(personas: list) -> str:
    if not personas:
        return '<div class="persona-gallery-empty">No personas saved yet. Create one above!</div>'

    cards_html = []
    for voice_id, voice_type, persona in personas:
        traits_html = f"""
            <span class="persona-trait">{persona.personality}</span>
            <span class="persona-trait">{persona.speaking_style}</span>
        """
        if persona.expertise:
            for exp in persona.expertise[:2]:
                traits_html += f'<span class="persona-trait">{exp}</span>'

        bio_preview = (
            persona.bio[:100] + "..." if len(persona.bio) > 100 else persona.bio
        )

        card_html = f"""
        <div class="persona-card">
            <div class="persona-card-header">
                <span class="persona-name">{persona.character_name}</span>
                <span class="persona-voice-badge">{voice_type.upper()}</span>
            </div>
            <div class="persona-traits">{traits_html}</div>
            <div class="persona-bio">{bio_preview or "No bio"}</div>
            <div style="font-size: 0.7rem; color: #555570; margin-top: 0.5rem;">
                Voice: {voice_id}
            </div>
        </div>
        """
        cards_html.append(card_html)

    return f'<div class="persona-cards-grid">{"".join(cards_html)}</div>'


def _generate_persona_voice_preview(voice_id: str, voice_type: str) -> str | None:
    try:
        from ui.voice_cards import generate_preview

        return generate_preview(voice_id, voice_type)
    except Exception as e:
        print(f"Voice preview generation failed: {format_user_error(e)}")
        return None


# ---------------------------------------------------------------------------
# OpenAI API tab
# ---------------------------------------------------------------------------

# OpenAI voice names suggested for empty link slots, in order.
_API_DEFAULT_NAMES = ("alloy", "echo", "sage", "nova")


def _get_api_voice_choices() -> list[tuple[str, str]]:
    """Studio voices for API links; voices with a persona show its name first."""
    persona_names = {
        f"{voice_type}:{voice_id}": persona.character_name
        for voice_id, voice_type, persona in list_personas()
    }
    choices = [("-- Not linked --", "")]
    for label, value in _get_podcast_voice_choices():
        if not value:
            continue
        if value in persona_names:
            label = f"{persona_names[value]} (persona) | {label}"
        choices.append((label, value))
    return choices


def _api_link_rows(settings: dict, voice_values: set[str]) -> list[tuple[str, str, str]]:
    """(OpenAI name, studio voice, language) per link slot, padded with free names."""
    rows = []
    for link in settings.get("links", [])[:API_MAX_LINKS]:
        name = link.get("openai_voice", "")
        if name not in OPENAI_VOICES:
            continue
        voice = link.get("voice", "")
        language = link.get("language") or "auto"
        rows.append(
            (
                name,
                voice if voice in voice_values else "",
                language if language in LANGUAGES else "auto",
            )
        )
    used = {row[0] for row in rows}
    spare = [n for n in dict.fromkeys(_API_DEFAULT_NAMES + OPENAI_VOICES) if n not in used]
    while len(rows) < API_MAX_LINKS:
        rows.append((spare.pop(0), "", "auto"))
    return rows


def _api_linked_names(settings: dict) -> list[str]:
    return [l["openai_voice"] for l in settings.get("links", []) if l.get("voice")]


def _api_test_voice_update(settings: dict, current: str | None = None):
    names = _api_linked_names(settings)
    return gr.update(
        choices=names, value=current if current in names else (names[0] if names else None)
    )


def _api_status_html() -> str:
    if api_server.running:
        url = html_escape.escape(api_server.base_url)
        return (
            '<div class="api-status api-status-on"><span class="api-dot"></span>'
            f"Running at <code>{url}</code></div>"
        )
    return '<div class="api-status"><span class="api-dot"></span>Stopped</div>'


def _api_message(text: str, ok: bool = True) -> str:
    css_class = "api-msg" if ok else "api-msg api-msg-error"
    return f'<div class="{css_class}">{html_escape.escape(text)}</div>'


def _api_usage_markdown(settings: dict) -> str:
    base_url = client_base_url(settings["host"], int(settings["port"]))
    linked = [l for l in settings.get("links", []) if l.get("voice")]
    example_voice = linked[0]["openai_voice"] if linked else "alloy"
    key = "YOUR_API_KEY" if settings.get("api_key") else "not-needed"
    voices = ", ".join(
        f"`{l['openai_voice']}` → {voice_display_name(l['voice'])}" for l in linked
    )
    lines = [
        f"**Base URL:** `{base_url}`  ",
        "**API key:** "
        + ("the key set on the left" if settings.get("api_key") else "any value (no key set)")
        + "  ",
        "**Model:** `tts-1` (any model name is accepted)  ",
        f"**Voices:** {voices or '*none linked yet*'}",
    ]
    if settings["host"] in ("0.0.0.0", "::"):
        lines += [
            "",
            "From other devices on your network, replace `127.0.0.1` with this PC's IP address.",
        ]
    lines += [
        "",
        "In apps with an OpenAI text-to-speech option (Open WebUI, SillyTavern, ...), "
        "choose OpenAI as the TTS provider, enter the base URL above, and pick a linked voice.",
        "",
        "**Python** (`pip install openai`)",
        "```python",
        "from openai import OpenAI",
        "",
        "client = OpenAI(",
        f'    base_url="{base_url}",',
        f'    api_key="{key}",',
        ")",
        "with client.audio.speech.with_streaming_response.create(",
        '    model="tts-1",',
        f'    voice="{example_voice}",',
        '    input="Hello from Qwen3-TTS Studio!",',
        ") as response:",
        '    response.stream_to_file("speech.mp3")',
        "```",
        "",
        "**PowerShell**",
        "```powershell",
        f'$body = @{{ model = "tts-1"; voice = "{example_voice}"; input = "Hello!" }}',
        "Invoke-RestMethod -Method Post `",
        f'  -Uri "{base_url}/audio/speech" `',
        f'  -Headers @{{ Authorization = "Bearer {key}" }} `',
        '  -ContentType "application/json" `',
        "  -Body ($body | ConvertTo-Json) -OutFile speech.mp3",
        "```",
    ]
    return "\n".join(lines)


def _api_settings_from_inputs(values) -> dict:
    """Build API settings from the tab's inputs; raises gr.Error if invalid."""
    n_link_inputs = API_MAX_LINKS * 3
    host, port, api_key, autostart = values[n_link_inputs:]
    links = []
    seen = set()
    for i in range(API_MAX_LINKS):
        name, voice, language = values[i * 3 : i * 3 + 3]
        if not voice:
            continue
        if not name:
            raise gr.Error(f"Link {i + 1}: choose an OpenAI voice name.")
        if name in seen:
            raise gr.Error(
                f"'{name}' is linked twice. Each OpenAI voice name can only be used once."
            )
        seen.add(name)
        links.append({"openai_voice": name, "voice": voice, "language": language or "auto"})

    try:
        port = int(port)
    except (TypeError, ValueError):
        port = 0
    if not 1 <= port <= 65535:
        raise gr.Error("Port must be a number between 1 and 65535.")

    return {
        "host": (host or "").strip() or "127.0.0.1",
        "port": port,
        "api_key": (api_key or "").strip(),
        "autostart": bool(autostart),
        "links": links,
    }


def api_save_settings(*values):
    """Inputs: link dropdowns, host, port, key, autostart, then the test voice."""
    settings = _api_settings_from_inputs(values[:-1])
    save_api_settings(settings)
    count = len(settings["links"])
    message = f"Saved {count} voice link{'' if count == 1 else 's'}."
    if api_server.running and (settings["host"], settings["port"]) != (
        api_server.host,
        api_server.port,
    ):
        message += " Restart the server to use the new host and port."
    return (
        _api_message(message),
        _api_usage_markdown(settings),
        _api_test_voice_update(settings, values[-1]),
    )


def api_start_server(*values):
    """Save the settings, then (re)start the server on the saved host and port."""
    settings = _api_settings_from_inputs(values[:-1])
    save_api_settings(settings)
    api_server.stop()
    try:
        api_server.start(settings["host"], settings["port"])
        message = _api_message(f"Server running at {api_server.base_url}")
    except RuntimeError as e:
        message = _api_message(str(e), ok=False)
    return (
        _api_status_html(),
        message,
        _api_usage_markdown(settings),
        _api_test_voice_update(settings, values[-1]),
    )


def api_stop_server():
    was_running = api_server.running
    api_server.stop()
    message = "Server stopped." if was_running else "Server is not running."
    return _api_status_html(), _api_message(message)


def api_refresh_voices(*current_values):
    choices = _get_api_voice_choices()
    valid = {value for _, value in choices}
    return [
        gr.update(choices=choices, value=v if v in valid else "") for v in current_values
    ]


def api_on_tab_select(*current_values):
    return [_api_status_html(), *api_refresh_voices(*current_values)]


def api_test_link(openai_voice, text, speed, progress=gr.Progress()):
    """Render text through a saved link, exactly as an API request would."""
    if not openai_voice:
        raise gr.Error("Link a studio voice and save your settings first.")
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please enter text to generate")
    link = find_link(load_api_settings(), openai_voice)
    if link is None:
        raise gr.Error(f"'{openai_voice}' is not linked. Save your settings first.")

    voice_name = voice_display_name(link["voice"])
    progress(0.1, desc=f"Generating with {voice_name}...")
    start_time = time.time()
    try:
        audio, sr = synthesize_link(link, text)
        audio = change_speed(audio, sr, float(speed))
    except Exception as e:
        gr.Warning(format_user_error(e))
        return None, f"❌ Error: {format_user_error(e)}"
    finally:
        _gpu_cleanup()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name
    sf.write(out_path, audio, sr)
    gen_time = time.time() - start_time
    duration = len(audio) / sr
    return (
        out_path,
        f"Done in {gen_time:.1f}s | Duration: {format_duration(duration)} | "
        f"{openai_voice} → {voice_name}",
    )


with gr.Blocks(
    title="Qwen3-TTS Studio", theme=build_theme(), css=custom_css, js=APP_JS
) as demo:
    gr.HTML(HEADER_HTML)

    current_prompt_data = gr.State(None)
    current_clone_model = gr.State(None)

    with gr.Row(elem_classes=["app-body"]):
        with gr.Column(scale=5, elem_classes=["main-col"]):
            with gr.Tabs(elem_classes=["main-tabs"]) as tabs:
                with gr.TabItem("Preset Voices", id="preset"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML('<div class="section-header">Voice Settings</div>')

                            cv_model = gr.Radio(
                                ["1.7B-CustomVoice", "0.6B-CustomVoice"],
                                value="1.7B-CustomVoice",
                                label="Model",
                                info="1.7B: Higher quality | 0.6B: Faster",
                            )
                            cv_speaker = gr.Dropdown(
                                choices=_get_preset_speaker_choices(),
                                value="serena",
                                label="Voice Preset",
                                info="Select a built-in voice character",
                            )
                            cv_language = gr.Dropdown(
                                choices=LANGUAGES,
                                value="auto",
                                label="Language",
                                info="Auto-detect or specify language",
                            )
                            cv_instruct = gr.Textbox(
                                label="Voice Style (1.7B only)",
                                placeholder="e.g., Speak warmly and enthusiastically",
                                lines=2,
                                info="Optional instruction to guide voice style",
                            )

                        with gr.Column(scale=2):
                            gr.HTML('<div class="section-header">Text Input</div>')

                            cv_text = gr.Textbox(
                                label="Text to Speak",
                                placeholder="Enter the text you want to convert to speech...",
                                lines=4,
                                max_lines=8,
                            )
                            cv_char_count = gr.HTML(value=update_char_count(""))

                            cv_btn = gr.Button(
                                "Generate Speech",
                                variant="primary",
                                elem_classes=["generate-btn"],
                                size="lg",
                            )

                            cv_status = gr.Textbox(
                                label="Status", interactive=False, show_label=True,
                                value="Ready to generate...",
                            )

                            cv_audio = gr.Audio(
                                label="Generated Audio", type="filepath", interactive=False
                            )

                with gr.TabItem("Clone Voice", id="clone"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML(
                                '<div class="section-header">Reference Samples</div>'
                            )
                            gr.Markdown(
                                "*Upload multiple voice samples for better quality. "
                                "More samples = more consistent voice cloning.*",
                                elem_classes=["info-text"],
                            )

                            vc_ref_audio = gr.File(
                                label="Upload Audio Samples (max 3)",
                                file_count="multiple",
                                file_types=["audio"],
                                type="filepath",
                            )

                            vc_samples_summary = gr.Textbox(
                                label="Sample Analysis",
                                interactive=False,
                                lines=5,
                                placeholder="Upload samples to see analysis...",
                            )
                            vc_samples_warnings = gr.HTML(value="")

                            vc_combine_samples = gr.Checkbox(
                                label="Combine samples for better quality",
                                value=True,
                                info="Merges voice characteristics from all samples",
                            )

                            gr.HTML(
                                '<div class="section-header" style="margin-top:1rem;">Transcripts</div>'
                            )

                            with gr.Accordion("Sample Transcripts", open=True):
                                vc_transcripts_info = gr.Markdown(
                                    "*Upload audio samples to enter transcripts.*"
                                )
                                vc_transcript_1 = gr.Textbox(
                                    label="Sample 1 (Primary)",
                                    placeholder="Enter the exact words spoken...",
                                    lines=2,
                                    visible=False,
                                )
                                vc_transcript_2 = gr.Textbox(
                                    label="Sample 2",
                                    placeholder="Optional: transcript for sample 2...",
                                    lines=2,
                                    visible=False,
                                )
                                vc_transcript_3 = gr.Textbox(
                                    label="Sample 3",
                                    placeholder="Optional: transcript for sample 3...",
                                    lines=2,
                                    visible=False,
                                )
                                with gr.Row():
                                    vc_auto_transcribe_btn = gr.Button(
                                        "Auto-transcribe Primary",
                                        size="sm",
                                        visible=False,
                                    )

                            vc_transcripts_json = gr.State(value="{}")
                            vc_transcript_state = gr.State(value={})
                            vc_current_file_paths = gr.State(value=[])

                            with gr.Accordion("Generation Settings", open=False):
                                vc_model = gr.Radio(
                                    ["1.7B-Base", "0.6B-Base"],
                                    value="1.7B-Base",
                                    label="Model",
                                )
                                vc_language = gr.Dropdown(
                                    choices=LANGUAGES, value="auto", label="Output Language"
                                )
                                vc_ref_language = gr.Dropdown(
                                    choices=LANGUAGES,
                                    value="auto",
                                    label="Reference Language",
                                    info="Language spoken in the reference samples (recommended if you generate in a different language)",
                                )
                                vc_crosslingual_opt = gr.Checkbox(
                                    label="Prioritize cross-lingual pronunciation",
                                    value=True,
                                    info="Keeps voice identity but may reduce transcript-based style transfer when languages differ",
                                )

                        with gr.Column(scale=2):
                            gr.HTML(
                                '<div class="section-header">Test Cloned Voice</div>'
                            )

                            vc_test_text = gr.Textbox(
                                label="Test Text",
                                placeholder="Enter text to test the cloned voice...",
                                lines=3,
                                info="Leave empty to just create the voice profile",
                            )
                            vc_test_char_count = gr.HTML(value=update_char_count(""))

                            vc_clone_btn = gr.Button(
                                "Clone & Generate",
                                variant="primary",
                                elem_classes=["generate-btn"],
                                size="lg",
                            )

                            vc_status = gr.Textbox(label="Status", interactive=False, value="Ready to generate...")
                            vc_output = gr.Audio(label="Test Output", type="filepath", interactive=False)

                            gr.HTML(
                                '<div class="section-header" style="margin-top:1rem;">Save Cloned Voice</div>'
                            )

                            with gr.Row(elem_classes=["align-end"]):
                                vc_name = gr.Textbox(
                                    label="Voice Name",
                                    placeholder="my_custom_voice",
                                    scale=2,
                                )
                                vc_save_btn = gr.Button("Save Voice", scale=1)

                            vc_description = gr.Textbox(
                                label="Description",
                                placeholder="Deep male voice with British accent...",
                                lines=2,
                            )
                            vc_style_note = gr.Textbox(
                                label="Usage Notes",
                                placeholder="Best for narration, avoid singing...",
                                lines=1,
                            )
                            vc_save_status = gr.Textbox(
                                label="", interactive=False, show_label=False
                            )

                    vc_samples_meta_json = gr.State(value="")

                with gr.TabItem("Voice Design", id="design"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML(
                                '<div class="section-header">Voice Description</div>'
                            )
                            gr.Markdown(
                                "*Describe the voice you want in natural language. "
                                "Include age, gender, tone, emotion, accent, etc.*"
                            )

                            vd_description = gr.Textbox(
                                label="Voice Description",
                                placeholder="A warm female voice with a British accent, speaking softly and calmly...",
                                lines=4,
                                info="Describe the desired voice characteristics",
                            )

                            vd_language = gr.Dropdown(
                                choices=LANGUAGES,
                                value="auto",
                                label="Language",
                                info="Auto-detect or specify language",
                            )

                            gr.HTML(
                                '<div class="section-header" style="margin-top:1rem;">Example Descriptions</div>'
                            )
                            gr.Markdown(
                                """
**Examples:**
- "A cheerful young female voice with high pitch and energetic tone"
- "Deep male voice, mature, authoritative, speaking slowly and clearly"
- "Elderly woman, warm and gentle, with a slight tremor in voice"
                                """,
                                elem_classes=["info-text"],
                            )

                        with gr.Column(scale=2):
                            gr.HTML('<div class="section-header">Text Input</div>')

                            vd_text = gr.Textbox(
                                label="Text to Speak",
                                placeholder="Enter the text you want to convert to speech...",
                                lines=4,
                                max_lines=8,
                            )
                            vd_char_count = gr.HTML(value=update_char_count(""))

                            vd_btn = gr.Button(
                                "Generate Speech",
                                variant="primary",
                                elem_classes=["generate-btn"],
                                size="lg",
                            )

                            vd_status = gr.Textbox(
                                label="Status", interactive=False, show_label=True,
                                value="Ready to generate...",
                            )

                            vd_audio = gr.Audio(
                                label="Generated Audio", type="filepath", interactive=False
                            )

                with gr.TabItem("Saved Voices", id="saved"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML('<div class="section-header">Your Voices</div>')
                            gr.Markdown(
                                "*Voices saved from the Clone Voice tab appear here for reuse.*",
                                elem_classes=["info-text"],
                            )

                            sv_voice_dropdown = gr.Dropdown(
                                choices=get_saved_voice_choices(),
                                label="Select Voice",
                                info="Choose from your saved voice clones",
                            )

                            with gr.Row():
                                sv_refresh_btn = gr.Button("Refresh", size="sm")
                                sv_delete_btn = gr.Button(
                                    "Delete", variant="stop", size="sm"
                                )

                            sv_model_info = gr.Textbox(
                                label="Model Used", interactive=False
                            )
                            sv_ref_language_info = gr.Textbox(
                                label="Reference Language", interactive=False
                            )
                            sv_description = gr.Textbox(
                                label="Description", interactive=False, lines=2
                            )
                            sv_ref_audio = gr.Audio(
                                label="Original Reference",
                                type="filepath",
                                interactive=False,
                            )

                        with gr.Column(scale=2):
                            gr.HTML(
                                '<div class="section-header">Generate with Saved Voice</div>'
                            )

                            sv_text = gr.Textbox(
                                label="Text to Speak",
                                placeholder="Enter text to generate with the saved voice...",
                                lines=4,
                            )
                            sv_char_count = gr.HTML(value=update_char_count(""))

                            sv_language = gr.Dropdown(
                                choices=LANGUAGES, value="auto", label="Language"
                            )

                            sv_crosslingual_opt = gr.Checkbox(
                                label="Prioritize cross-lingual pronunciation",
                                value=True,
                                info="If output language differs from the reference language, prioritize pronunciation; may reduce style transfer",
                            )

                            sv_generate_btn = gr.Button(
                                "Generate Speech",
                                variant="primary",
                                elem_classes=["generate-btn"],
                                size="lg",
                            )

                            sv_status = gr.Textbox(label="Status", interactive=False, value="Ready to generate...")
                            sv_audio = gr.Audio(
                                label="Generated Audio", type="filepath", interactive=False
                            )

                    sv_style_note = gr.Textbox(visible=False)
                    sv_ref_text = gr.Textbox(visible=False)
                    sv_delete_status = gr.Textbox(visible=False)
                    sv_delete_confirm = gr.State(False)

                with gr.TabItem("Personas", id="personas"):
                    gr.Markdown("## Persona Management")
                    gr.Markdown("*Define character personas for your podcast voices*")

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML('<div class="section-header">Voice Selection</div>')

                            persona_voice_dropdown = gr.Dropdown(
                                label="Select Voice",
                                choices=_get_persona_voice_choices(),
                                value=None,
                                interactive=True,
                                allow_custom_value=False,
                                info="Choose a voice to create or edit its persona",
                            )

                            persona_refresh_voices_btn = gr.Button(
                                "Refresh Voices", size="sm"
                            )

                        with gr.Column(scale=2):
                            gr.HTML(
                                '<div class="section-header">Character Definition</div>'
                            )

                            persona_character_name = gr.Textbox(
                                label="Character Name",
                                placeholder="e.g., Dr. Sarah Chen, The Wise Narrator",
                                info="Display name for this character",
                            )

                            with gr.Row():
                                persona_personality = gr.Dropdown(
                                    label="Personality",
                                    choices=sorted(ALLOWED_PERSONALITIES),
                                    value=None,
                                    interactive=True,
                                    info="Core personality trait",
                                )
                                persona_speaking_style = gr.Dropdown(
                                    label="Speaking Style",
                                    choices=sorted(ALLOWED_SPEAKING_STYLES),
                                    value=None,
                                    interactive=True,
                                    info="How they communicate",
                                )

                            persona_expertise = gr.Textbox(
                                label="Expertise (comma-separated)",
                                placeholder="e.g., AI Ethics, Philosophy, Technology",
                                info="Areas of knowledge or expertise",
                            )

                            persona_background = gr.Textbox(
                                label="Background",
                                placeholder="Brief background information about the character...",
                                lines=2,
                                info="Character's history, role, or context",
                            )

                            persona_bio = gr.Textbox(
                                label="Bio / Personality Notes",
                                placeholder="Detailed character description, personality quirks, mannerisms...",
                                lines=3,
                                info="Extended character description for transcript generation",
                            )

                            with gr.Row():
                                persona_save_btn = gr.Button(
                                    "Save Persona", variant="primary", size="lg"
                                )
                                persona_delete_btn = gr.Button(
                                    "Delete Persona", variant="stop", size="lg"
                                )
                                persona_preview_btn = gr.Button(
                                    "Preview Voice", size="lg"
                                )

                            persona_status_text = gr.Textbox(
                                label="Status", interactive=False, show_label=True
                            )

                            persona_preview_audio = gr.Audio(
                                label="Voice Preview", type="filepath", visible=True, interactive=False
                            )

                    gr.HTML(
                        '<div class="section-header" style="margin-top: 2rem;">Saved Personas Gallery</div>'
                    )

                    personas_gallery = gr.HTML(
                        value=_render_persona_cards(list_personas())
                    )

                    persona_refresh_gallery_btn = gr.Button(
                        "Refresh Gallery", size="sm"
                    )

                    persona_selected_voice_state = gr.State(value=None)
                    persona_delete_confirm_state = gr.State(value=False)

                    def on_persona_voice_select(voice_value: str):
                        if not voice_value:
                            return (
                                "",
                                "",
                                None,
                                None,
                                "",
                                "",
                                "Select a voice to begin",
                                voice_value,
                                False,
                            )

                        voice_id, voice_type = _parse_persona_voice_value(voice_value)

                        if not voice_id:
                            return (
                                "",
                                "",
                                None,
                                None,
                                "",
                                "",
                                "Invalid voice selection",
                                voice_value,
                                False,
                            )

                        existing = load_persona(voice_id, voice_type)

                        if existing:
                            expertise_str = (
                                ", ".join(existing.expertise)
                                if existing.expertise
                                else ""
                            )
                            return (
                                existing.character_name,
                                expertise_str,
                                existing.personality,
                                existing.speaking_style,
                                existing.background,
                                existing.bio,
                                f"Loaded persona for {voice_id}",
                                voice_value,
                                False,
                            )
                        else:
                            return (
                                "",
                                "",
                                None,
                                None,
                                "",
                                "",
                                f"No persona found for {voice_id}. Create one!",
                                voice_value,
                                False,
                            )

                    def on_persona_save(
                        voice_value, char_name, pers, style, exp, bg, bio_text
                    ):
                        if not voice_value:
                            gr.Warning("Please select a voice first")
                            return "Error: No voice selected", _render_persona_cards(
                                list_personas()
                            )

                        voice_id, voice_type = _parse_persona_voice_value(voice_value)

                        if not voice_id:
                            gr.Warning("Invalid voice selection")
                            return "Error: Invalid voice", _render_persona_cards(
                                list_personas()
                            )

                        if not char_name or not char_name.strip():
                            gr.Warning("Character name is required")
                            return (
                                "Error: Character name required",
                                _render_persona_cards(list_personas()),
                            )

                        if not pers:
                            gr.Warning("Personality is required")
                            return "Error: Personality required", _render_persona_cards(
                                list_personas()
                            )

                        if not style:
                            gr.Warning("Speaking style is required")
                            return (
                                "Error: Speaking style required",
                                _render_persona_cards(list_personas()),
                            )

                        expertise_list = (
                            [e.strip() for e in exp.split(",") if e.strip()]
                            if exp
                            else []
                        )

                        try:
                            persona = Persona(
                                voice_id=voice_id,
                                voice_type=voice_type,
                                character_name=char_name.strip(),
                                personality=pers,
                                speaking_style=style,
                                expertise=expertise_list,
                                background=bg.strip() if bg else "",
                                bio=bio_text.strip() if bio_text else "",
                            )

                            save_persona(persona)
                            gr.Info(f"Persona saved for {char_name}")
                            return f"Saved persona: {char_name}", _render_persona_cards(
                                list_personas()
                            )

                        except ValueError as e:
                            error_msg = format_user_error(e)
                            gr.Warning(f"Validation error: {error_msg}")
                            return f"Error: {error_msg}", _render_persona_cards(
                                list_personas()
                            )
                        except Exception as e:
                            error_msg = format_user_error(e)
                            gr.Warning(f"Failed to save: {error_msg}")
                            return (
                                f"Error saving persona: {error_msg}",
                                _render_persona_cards(list_personas()),
                            )

                    def on_persona_delete(voice_value, confirm_state):
                        gallery_html = _render_persona_cards(list_personas())

                        if not voice_value:
                            gr.Warning("Please select a voice first")
                            return (
                                "Error: No voice selected",
                                gallery_html,
                                "",
                                None,
                                None,
                                "",
                                "",
                                "",
                                False,
                            )

                        voice_id, voice_type = _parse_persona_voice_value(voice_value)

                        if not voice_id:
                            gr.Warning("Invalid voice selection")
                            return (
                                "Error: Invalid voice",
                                gallery_html,
                                "",
                                None,
                                None,
                                "",
                                "",
                                "",
                                False,
                            )

                        if not confirm_state:
                            gr.Warning(
                                f"Click Delete again to confirm deletion of persona for '{voice_id}'"
                            )
                            return (
                                f"Click Delete again to confirm deletion for {voice_id}",
                                gallery_html,
                                gr.update(),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                                True,
                            )

                        try:
                            deleted = delete_persona(voice_id, voice_type)

                            if deleted:
                                gr.Info(f"Persona deleted for {voice_id}")
                                return (
                                    f"Deleted persona for {voice_id}",
                                    _render_persona_cards(list_personas()),
                                    "",
                                    None,
                                    None,
                                    "",
                                    "",
                                    "",
                                    False,
                                )
                            else:
                                gr.Warning(f"No persona found for {voice_id}")
                                return (
                                    f"No persona found for {voice_id}",
                                    gallery_html,
                                    "",
                                    None,
                                    None,
                                    "",
                                    "",
                                    "",
                                    False,
                                )

                        except Exception as e:
                            error_msg = format_user_error(e)
                            gr.Warning(f"Failed to delete: {error_msg}")
                            return (
                                f"Error deleting persona: {error_msg}",
                                gallery_html,
                                "",
                                None,
                                None,
                                "",
                                "",
                                "",
                                False,
                            )

                    def on_persona_preview_voice(voice_value):
                        if not voice_value:
                            gr.Warning("Please select a voice first")
                            return None, "Select a voice to preview"

                        voice_id, voice_type = _parse_persona_voice_value(voice_value)

                        if not voice_id:
                            gr.Warning("Invalid voice selection")
                            return None, "Invalid voice selection"

                        audio_path = _generate_persona_voice_preview(
                            voice_id, voice_type
                        )

                        if audio_path:
                            gr.Info("Voice preview generated!")
                            return audio_path, f"Preview generated for {voice_id}"
                        else:
                            gr.Warning("Failed to generate preview")
                            return None, "Failed to generate voice preview"

                    def on_persona_refresh_voices():
                        choices = _get_persona_voice_choices()
                        return gr.update(choices=choices, value=None)

                    def on_persona_refresh_gallery():
                        return _render_persona_cards(list_personas())

                    persona_voice_dropdown.change(
                        fn=on_persona_voice_select,
                        inputs=[persona_voice_dropdown],
                        outputs=[
                            persona_character_name,
                            persona_expertise,
                            persona_personality,
                            persona_speaking_style,
                            persona_background,
                            persona_bio,
                            persona_status_text,
                            persona_selected_voice_state,
                            persona_delete_confirm_state,
                        ],
                    )

                    persona_save_btn.click(
                        fn=on_persona_save,
                        inputs=[
                            persona_voice_dropdown,
                            persona_character_name,
                            persona_personality,
                            persona_speaking_style,
                            persona_expertise,
                            persona_background,
                            persona_bio,
                        ],
                        outputs=[persona_status_text, personas_gallery],
                    )

                    persona_delete_btn.click(
                        fn=on_persona_delete,
                        inputs=[persona_voice_dropdown, persona_delete_confirm_state],
                        outputs=[
                            persona_status_text,
                            personas_gallery,
                            persona_character_name,
                            persona_personality,
                            persona_speaking_style,
                            persona_expertise,
                            persona_background,
                            persona_bio,
                            persona_delete_confirm_state,
                        ],
                    )

                    persona_preview_btn.click(
                        fn=on_persona_preview_voice,
                        inputs=[persona_voice_dropdown],
                        outputs=[persona_preview_audio, persona_status_text],
                    )

                    persona_refresh_voices_btn.click(
                        fn=on_persona_refresh_voices, outputs=[persona_voice_dropdown]
                    )

                    persona_refresh_gallery_btn.click(
                        fn=on_persona_refresh_gallery, outputs=[personas_gallery]
                    )

                with gr.TabItem("Podcast", id="podcast"):
                    podcast_outline_state = gr.State(None)
                    podcast_transcript_state = gr.State(None)
                    podcast_speaker_profile_state = gr.State(None)
                    podcast_voice_selections_state = gr.State({})
                    podcast_session_state = gr.State(
                        None
                    )  # Stores {podcast_dir, quality_preset, language}

                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Tabs(elem_classes=["sub-tabs"]):
                                with gr.TabItem("AI Generated"):
                                    gr.HTML(
                                        '<div class="section-header">Topic & Style</div>'
                                    )

                                    podcast_topic = gr.Textbox(
                                        label="Podcast Topic",
                                        placeholder="Enter your podcast topic or main subject...\n\nExample: The future of artificial intelligence",
                                        lines=3,
                                        max_lines=3,
                                        info="Required. What is your podcast about?",
                                    )
                                    podcast_topic_chars = gr.HTML(
                                        value=update_topic_char_count("")
                                    )

                                    podcast_key_points = gr.Textbox(
                                        label="Key Points (Optional)",
                                        placeholder="List the main points you want to cover...\n\n- Point 1\n- Point 2\n- Point 3",
                                        lines=4,
                                        info="Bullet points or key topics to discuss",
                                    )

                                    podcast_briefing = gr.Textbox(
                                        label="Style & Tone (Optional)",
                                        placeholder="Describe the desired style and tone...\n\nExample: Conversational and engaging",
                                        lines=2,
                                        info="How should the podcast sound?",
                                    )

                                    with gr.Row():
                                        podcast_quality_preset = gr.Dropdown(
                                            choices=list(
                                                PODCAST_QUALITY_PRESETS.keys()
                                            ),
                                            value="standard",
                                            label="Quality",
                                            info="Select quality level",
                                            scale=1,
                                        )
                                        podcast_num_segments = gr.Slider(
                                            minimum=2,
                                            maximum=8,
                                            value=4,
                                            step=1,
                                            label="Segments",
                                            info="Outline segments",
                                            scale=1,
                                        )

                                    podcast_language = gr.Dropdown(
                                        label="Language",
                                        choices=[
                                            "English",
                                            "Korean",
                                            "Chinese",
                                            "Japanese",
                                            "Spanish",
                                            "French",
                                            "German",
                                        ],
                                        value="English",
                                        info="Language for script generation and voice synthesis",
                                    )
                                    with gr.Accordion("LLM Provider", open=False):
                                        podcast_llm_provider = gr.Dropdown(
                                            choices=[
                                                "Unsloth Desktop",
                                                "OpenAI",
                                                "Ollama",
                                                "OpenRouter",
                                                "Claude",
                                            ],
                                            value="Unsloth Desktop",
                                            label="Provider",
                                            info="LLM service for script generation",
                                        )
                                        podcast_llm_model = gr.Dropdown(
                                            label="Model",
                                            choices=PROVIDER_MODEL_OPTIONS[
                                                LLMProvider.UNSLOTH
                                            ],
                                            value=DEFAULT_MODELS[LLMProvider.UNSLOTH],
                                            info="Model for generation (Unsloth Desktop: type the model name loaded in Unsloth)",
                                            allow_custom_value=True,
                                        )
                                        podcast_llm_api_key = gr.Textbox(
                                            label="API Key",
                                            value="",
                                            placeholder="Not required for Unsloth Desktop / Ollama",
                                            type="password",
                                            visible=True,
                                            info="API key (not needed for Unsloth Desktop or Ollama)",
                                        )
                                        podcast_llm_base_url = gr.Textbox(
                                            label="Base URL",
                                            value="http://localhost:8888/v1",
                                            placeholder="Custom API endpoint",
                                            info="API endpoint URL (Unsloth Desktop default: http://localhost:8888/v1)",
                                        )
                                        podcast_llm_status = gr.HTML(value="")
                                        podcast_llm_test_btn = gr.Button(
                                            "Test Connection", size="sm"
                                        )

                                    with gr.Row(elem_classes=["speakers-head"]):
                                        gr.HTML(
                                            '<div class="section-header">Speakers (1-4)</div>'
                                        )
                                        podcast_refresh_voices_btn = gr.Button(
                                            "↻ Refresh",
                                            size="sm",
                                            scale=0,
                                            min_width=80,
                                        )

                                    podcast_voice_summary = gr.HTML(
                                        value='<div style="color:#888; font-size:0.9em;">Select 1-4 speakers below</div>'
                                    )

                                    podcast_speaker_slots = []
                                    podcast_preview_buttons = []
                                    _slot_roles = ["Host", "Guest", "Guest", "Guest"]
                                    _initial_voice_choices = (
                                        _get_podcast_voice_choices()
                                    )  # Cache once
                                    for i in range(4):
                                        with gr.Row(elem_classes=["speaker-row", "ai-slot"]):
                                            slot_role = gr.Dropdown(
                                                choices=ROLES,
                                                value=_slot_roles[i],
                                                label=f"Speaker {i + 1}",
                                                scale=1,
                                                min_width=100,
                                                interactive=True,
                                            )
                                            slot_voice = gr.Dropdown(
                                                choices=_initial_voice_choices,
                                                value="",
                                                label="Voice",
                                                scale=2,
                                                min_width=150,
                                                interactive=True,
                                                allow_custom_value=False,
                                            )
                                            slot_preview = gr.Button(
                                                "▶",
                                                size="sm",
                                                scale=0,
                                                min_width=40,
                                            )
                                            podcast_preview_buttons.append(slot_preview)
                                            podcast_speaker_slots.append(
                                                (slot_role, slot_voice)
                                            )

                                    podcast_preview_audio = gr.Audio(
                                        label="Preview",
                                        visible=True,
                                        interactive=False,
                                    )

                                    podcast_voice_status = gr.HTML(value="")

                                    podcast_generate_btn = gr.Button(
                                        "Generate Podcast",
                                        variant="primary",
                                        elem_classes=["generate-btn"],
                                        size="lg",
                                    )

                                with gr.TabItem("Custom Script"):
                                    gr.HTML(
                                        '<div class="section-header">Paste Your Script</div>'
                                    )
                                    custom_script_input = gr.Textbox(
                                        label="Script",
                                        placeholder="Speaker One: Welcome to the show!\nSpeaker Two: Thanks for having me.\nSpeaker One: Let's dive in.",
                                        lines=12,
                                        info="Use 'Speaker Name: dialogue text' format",
                                    )
                                    custom_script_parse_btn = gr.Button(
                                        "Parse Script", size="sm"
                                    )
                                    custom_script_status = gr.HTML(value="")

                                    with gr.Row(elem_classes=["speakers-head"]):
                                        gr.HTML(
                                            '<div class="section-header">Speakers (1-4)</div>'
                                        )
                                        custom_refresh_voices_btn = gr.Button(
                                            "↻ Refresh",
                                            size="sm",
                                            scale=0,
                                            min_width=80,
                                        )

                                    custom_voice_summary = gr.HTML(
                                        value='<div style="color:#888; font-size:0.9em;">'
                                        "Name each speaker exactly as in the script (or click "
                                        '"Parse Script" to fill names), then pick a voice</div>'
                                    )

                                    custom_speaker_slots = []
                                    custom_preview_buttons = []
                                    for i in range(4):
                                        with gr.Row(elem_classes=["speaker-row", "custom-slot"]):
                                            csn = gr.Textbox(
                                                label=f"Speaker {i + 1}",
                                                placeholder="Name in script",
                                                scale=1,
                                                min_width=100,
                                            )
                                            csr_role = gr.Dropdown(
                                                choices=ROLES,
                                                value=_slot_roles[i],
                                                label="Role",
                                                scale=1,
                                                min_width=100,
                                                interactive=True,
                                            )
                                            csv = gr.Dropdown(
                                                choices=_initial_voice_choices,
                                                value="",
                                                label="Voice",
                                                scale=2,
                                                min_width=150,
                                                interactive=True,
                                                allow_custom_value=False,
                                            )
                                            cs_preview = gr.Button(
                                                "▶",
                                                size="sm",
                                                scale=0,
                                                min_width=40,
                                            )
                                            custom_preview_buttons.append(cs_preview)
                                        custom_speaker_slots.append(
                                            (csn, csr_role, csv)
                                        )

                                    custom_preview_audio = gr.Audio(
                                        label="Preview",
                                        visible=True,
                                        interactive=False,
                                    )

                                    custom_episode_title = gr.Textbox(
                                        label="Episode Title (Optional)",
                                        placeholder="Leave blank to auto-generate from script",
                                        lines=1,
                                    )
                                    custom_quality_preset = gr.Dropdown(
                                        choices=list(PODCAST_QUALITY_PRESETS.keys()),
                                        value="standard",
                                        label="Quality",
                                    )
                                    custom_language = gr.Dropdown(
                                        label="Language",
                                        choices=[
                                            "English",
                                            "Korean",
                                            "Chinese",
                                            "Japanese",
                                            "Spanish",
                                            "French",
                                            "German",
                                        ],
                                        value="English",
                                    )
                                    custom_generate_btn = gr.Button(
                                        "Generate from Script",
                                        variant="primary",
                                        elem_classes=["generate-btn"],
                                        size="lg",
                                    )

                        with gr.Column(scale=2):
                            gr.HTML(
                                '<div class="section-header">Progress</div>',
                                elem_classes=["progress-anchor"],
                            )

                            podcast_step_indicator = gr.HTML(
                                value=create_step_indicator_html(
                                    GenerationStep.OUTLINE, 0.0
                                )
                            )

                            podcast_overall_progress = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=0,
                                label="Overall progress",
                                interactive=False,
                                elem_classes=["progress-bar-slider"],
                            )

                            with gr.Row(elem_classes=["status-row"]):
                                podcast_status = gr.Textbox(
                                    value="Ready to generate...",
                                    label="Status",
                                    interactive=False,
                                    scale=3,
                                )
                                podcast_time_remaining = gr.Textbox(
                                    value="",
                                    label="Time",
                                    interactive=False,
                                    scale=1,
                                    min_width=170,
                                )

                            podcast_error_display = gr.HTML(value="")

                            gr.HTML(
                                '<div class="section-header" style="margin-top:1rem;">Audio Output</div>'
                            )

                            podcast_final_audio = gr.Audio(
                                label="Generated Podcast",
                                type="filepath",
                                interactive=False,
                            )

                            podcast_download = gr.File(
                                label="Download Podcast", visible=False
                            )

                            gr.HTML(
                                '<div class="section-header" style="margin-top:1rem;">Draft Preview</div>'
                            )

                            with gr.Tabs(elem_classes=["sub-tabs"]):
                                with gr.TabItem("Transcript"):
                                    podcast_transcript_html = gr.HTML(
                                        value='<div class="empty-state">Generate a podcast to see the transcript</div>'
                                    )
                                with gr.TabItem("Outline"):
                                    podcast_outline_html = gr.HTML(
                                        value='<div class="empty-state">Generate a podcast to see the outline</div>'
                                    )

                            with gr.Accordion(
                                "Edit Transcript", open=False, visible=False
                            ) as podcast_edit_accordion:
                                gr.HTML(
                                    '<div class="hint">'
                                    'Edit dialogue text below, then click "Regenerate Audio" to apply changes.</div>'
                                )
                                podcast_transcript_editor = gr.Dataframe(
                                    headers=["Speaker", "Text"],
                                    datatype=["str", "str"],
                                    interactive=True,
                                    wrap=True,
                                    value=[],
                                )
                                podcast_regenerate_btn = gr.Button(
                                    "Regenerate Audio from Edits",
                                    variant="primary",
                                    size="sm",
                                )

                            with gr.Accordion("Podcast History", open=False):
                                with gr.Row():
                                    podcast_history_search = gr.Textbox(
                                        placeholder="Search history...",
                                        show_label=False,
                                        scale=3,
                                    )
                                    podcast_history_favorites = gr.Checkbox(
                                        label="Favorites only",
                                        value=False,
                                        scale=1,
                                    )
                                podcast_history_display = gr.HTML(
                                    value=format_history_for_display(),
                                    elem_classes=["history-display"],
                                )
                                podcast_hist_init = get_podcast_history_initial()
                                podcast_history_dropdown = gr.Dropdown(
                                    choices=podcast_hist_init[0],
                                    value=podcast_hist_init[1],
                                    label="Select to load",
                                    allow_custom_value=False,
                                )
                                podcast_history_metadata = gr.Textbox(
                                    label="Details",
                                    lines=3,
                                    interactive=False,
                                    value=podcast_hist_init[3],
                                )
                                podcast_history_audio = gr.Audio(
                                    label="Playback",
                                    type="filepath",
                                    interactive=False,
                                    value=podcast_hist_init[2],
                                )
                                with gr.Row(elem_classes=["mini-btn-row"]):
                                    podcast_history_refresh = gr.Button(
                                        "Refresh", size="sm"
                                    )
                                    podcast_history_favorite = gr.Button(
                                        "★ Favorite", size="sm"
                                    )
                                    podcast_history_delete = gr.Button(
                                        "Delete", size="sm", variant="stop"
                                    )
                                podcast_history_delete_confirm = gr.State(False)

                    def on_llm_provider_change(provider_name):
                        """Update defaults when LLM provider changes."""
                        provider_map = {
                            "OpenAI": LLMProvider.OPENAI,
                            "Ollama": LLMProvider.OLLAMA,
                            "Unsloth Desktop": LLMProvider.UNSLOTH,
                            "OpenRouter": LLMProvider.OPENROUTER,
                            "Claude": LLMProvider.CLAUDE,
                        }
                        provider = provider_map.get(provider_name, LLMProvider.UNSLOTH)
                        default_model = DEFAULT_MODELS[provider]
                        model_choices = PROVIDER_MODEL_OPTIONS[provider]

                        if provider == LLMProvider.UNSLOTH:
                            return (
                                gr.update(choices=model_choices, value=default_model),
                                gr.update(
                                    value="",
                                    placeholder="Not required for Unsloth Desktop",
                                ),
                                gr.update(value="http://localhost:8888/v1"),
                                "",
                            )
                        elif provider == LLMProvider.OLLAMA:
                            return (
                                gr.update(choices=model_choices, value=default_model),
                                gr.update(
                                    value="", placeholder="Not required for Ollama"
                                ),
                                gr.update(value="http://localhost:11434/v1"),
                                "",
                            )
                        elif provider == LLMProvider.OPENROUTER:
                            return (
                                gr.update(choices=model_choices, value=default_model),
                                gr.update(
                                    value="", placeholder="Enter OpenRouter API key"
                                ),
                                gr.update(value="https://openrouter.ai/api/v1"),
                                "",
                            )
                        elif provider == LLMProvider.CLAUDE:
                            return (
                                gr.update(choices=model_choices, value=default_model),
                                gr.update(
                                    value="", placeholder="Enter Anthropic API key"
                                ),
                                gr.update(
                                    value="", placeholder="Default Anthropic endpoint"
                                ),
                                "",
                            )
                        else:  # OpenAI
                            return (
                                gr.update(choices=model_choices, value=default_model),
                                gr.update(value="", placeholder="Enter OpenAI API key"),
                                gr.update(value=""),
                                "",
                            )

                    def test_llm_connection(provider_name, model, api_key, base_url):
                        """Test LLM provider connection."""
                        provider_map = {
                            "OpenAI": LLMProvider.OPENAI,
                            "Ollama": LLMProvider.OLLAMA,
                            "Unsloth Desktop": LLMProvider.UNSLOTH,
                            "OpenRouter": LLMProvider.OPENROUTER,
                            "Claude": LLMProvider.CLAUDE,
                        }
                        provider = provider_map.get(provider_name, LLMProvider.UNSLOTH)

                        if provider == LLMProvider.UNSLOTH and not (model or "").strip():
                            return (
                                '<div style="color: #dc3545;">'
                                "Enter the model name loaded in Unsloth Desktop "
                                "first (the test needs it to validate the endpoint).</div>"
                            )

                        if not api_key and provider not in (
                            LLMProvider.OLLAMA,
                            LLMProvider.UNSLOTH,
                        ):
                            try:
                                from config import get_api_key_for_provider

                                api_key = get_api_key_for_provider(provider.value)
                            except ValueError:
                                return '<div style="color: #dc3545;">API key required. Enter key above or set in .env file.</div>'

                        config = get_default_config(
                            provider=provider,
                            api_key=api_key or "",
                            base_url=base_url or "",
                            model=model or "",
                        )
                        success, message = validate_connection(config)
                        if success:
                            return f'<div style="color: #28a745;">{message}</div>'
                        return f'<div style="color: #dc3545;">{message}</div>'

                    def build_voice_selections_from_slots(*slot_values):
                        sels = {}
                        num_slots = len(slot_values) // 2
                        for i in range(num_slots):
                            role = slot_values[i * 2]
                            voice_val = slot_values[i * 2 + 1]
                            if voice_val and voice_val != "":
                                parts = voice_val.split(":", 1)
                                if len(parts) != 2:
                                    summary = (
                                        '<div style="color:#dc3545;">'
                                        "Invalid voice selection. Choose a preset or saved voice."
                                        "</div>"
                                    )
                                    return {}, summary
                                vtype, vid = parts
                                if vtype not in {"preset", "saved"} or not vid:
                                    summary = (
                                        '<div style="color:#dc3545;">'
                                        "Invalid voice selection. Choose a preset or saved voice."
                                        "</div>"
                                    )
                                    return {}, summary
                                sels[f"slot_{i}"] = {
                                    "voice_id": vid,
                                    "name": vid,
                                    "role": role,
                                    "type": vtype,
                                }
                        count = len(sels)
                        if count == 0:
                            summary = (
                                '<div style="color:#888;">Select 1-4 speakers</div>'
                            )
                        elif count == 1:
                            summary = '<div style="color:#28a745;">1 speaker selected ✓ (Narration mode)</div>'
                        else:
                            summary = f'<div style="color:#28a745;">{count} speakers selected ✓</div>'
                        return sels, summary

                    def play_podcast_preview(voice_value):
                        """Generate and play preview for selected voice."""
                        if not voice_value or voice_value.strip() == "":
                            return None

                        try:
                            # Parse voice_value format: "preset:serena" or "saved:my_voice"
                            parts = voice_value.split(":", 1)
                            if len(parts) != 2:
                                return None

                            voice_type, voice_id = parts
                            audio_path = generate_preview(voice_id, voice_type)

                            if audio_path and os.path.exists(audio_path):
                                return audio_path
                            return None
                        except Exception as e:
                            print(f"Error playing preview: {e}")
                            return None

                    def validate_podcast_voices(selections):
                        is_valid, message, _ = validate_selections(selections)
                        if is_valid:
                            return f'<div style="color: #28a745;">{message}</div>'
                        return f'<div style="color: #dc3545;">{message}</div>'

                    @dataclass
                    class _ProgressEvent:
                        step: GenerationStep
                        progress: float
                        status: str
                        detail: str
                        data: dict | None = None

                    @dataclass
                    class _DoneEvent:
                        result: dict[str, Any]

                    @dataclass
                    class _ErrorEvent:
                        error: str
                        tb: str

                    def run_podcast_generation(
                        topic,
                        key_points,
                        briefing,
                        num_segments,
                        voice_selections,
                        quality_preset,
                        language,
                        llm_provider_name,
                        llm_model,
                        llm_api_key,
                        llm_base_url,
                    ):
                        is_valid, error_msg = validate_content(
                            topic, key_points, briefing
                        )
                        if not is_valid:
                            yield (
                                create_step_indicator_html(GenerationStep.OUTLINE, 0.0),
                                0,
                                f"Error: {error_msg}",
                                "",
                                f'<div style="color: #dc3545;">{error_msg}</div>',
                                None,
                                None,
                                None,
                                gr.update(visible=False),
                                gr.update(value="Generate Podcast", interactive=True),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                            )
                            return

                        voice_valid, voice_msg, voice_output = validate_selections(
                            voice_selections
                        )
                        if not voice_valid:
                            yield (
                                create_step_indicator_html(GenerationStep.OUTLINE, 0.0),
                                0,
                                f"Error: {voice_msg}",
                                "",
                                f'<div style="color: #dc3545;">{voice_msg}</div>',
                                None,
                                None,
                                None,
                                gr.update(visible=False),
                                gr.update(value="Generate Podcast", interactive=True),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                            )
                            return

                        provider_map = {
                            "OpenAI": LLMProvider.OPENAI,
                            "Ollama": LLMProvider.OLLAMA,
                            "Unsloth Desktop": LLMProvider.UNSLOTH,
                            "OpenRouter": LLMProvider.OPENROUTER,
                            "Claude": LLMProvider.CLAUDE,
                        }
                        llm_provider = provider_map.get(
                            llm_provider_name, LLMProvider.UNSLOTH
                        )

                        if llm_provider == LLMProvider.UNSLOTH and not (
                            llm_model or ""
                        ).strip():
                            yield (
                                create_step_indicator_html(GenerationStep.OUTLINE, 0.0),
                                0,
                                "Error: Model name required for Unsloth Desktop",
                                "",
                                '<div style="color: #dc3545;">Enter the model name loaded in Unsloth Desktop in the LLM Provider section.</div>',
                                None,
                                None,
                                None,
                                gr.update(visible=False),
                                gr.update(value="Generate Podcast", interactive=True),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                            )
                            return

                        if not llm_api_key and llm_provider not in (
                            LLMProvider.OLLAMA,
                            LLMProvider.UNSLOTH,
                        ):
                            try:
                                from config import get_api_key_for_provider

                                llm_api_key = get_api_key_for_provider(
                                    llm_provider.value
                                )
                            except ValueError:
                                yield (
                                    create_step_indicator_html(
                                        GenerationStep.OUTLINE, 0.0
                                    ),
                                    0,
                                    f"Error: API key required for {llm_provider_name}",
                                    "",
                                    f'<div style="color: #dc3545;">API key required for {llm_provider_name}. Enter it in the LLM Provider section or set in .env file.</div>',
                                    None,
                                    None,
                                    None,
                                    gr.update(visible=False),
                                    gr.update(
                                        value="Generate Podcast", interactive=True
                                    ),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                )
                                return

                        llm_config = get_default_config(
                            provider=llm_provider,
                            api_key=llm_api_key or "",
                            base_url=llm_base_url or "",
                            model=llm_model or "",
                        )

                        print(f"[LANG] UI selected: {language}")
                        content_input = {
                            "topic": topic,
                            "key_points": key_points,
                            "briefing": briefing,
                            "num_segments": int(num_segments),
                            "language": language,
                        }

                        q: queue.Queue[_ProgressEvent | _DoneEvent | _ErrorEvent] = (
                            queue.Queue(maxsize=500)
                        )
                        cancel_event = threading.Event()

                        current_step = GenerationStep.OUTLINE
                        step_progress = 0.0
                        status_text = "Starting..."
                        last_emit_time = 0.0

                        generation_started = time.monotonic()
                        clip_durations: list[float] = []
                        current_clip_started: float = 0.0
                        current_clip_index: int = 0
                        total_clips: int = 0
                        eta_lock = threading.Lock()

                        def progress_callback(
                            step_name: str, detail: dict[str, Any] | None
                        ):
                            nonlocal \
                                last_emit_time, \
                                current_clip_started, \
                                current_clip_index, \
                                total_clips
                            if cancel_event.is_set():
                                return

                            if detail is None:
                                detail = {}

                            status = detail.get("status", "")
                            step = GenerationStep.OUTLINE

                            if step_name == "generate_clips":
                                with eta_lock:
                                    total_clips = detail.get("total", total_clips)
                                    clip_idx = detail.get("current", 0)
                                    if status == "clip_started":
                                        current_clip_started = time.monotonic()
                                        current_clip_index = clip_idx
                                    elif (
                                        status == "progress"
                                        and current_clip_started > 0
                                    ):
                                        clip_duration = (
                                            time.monotonic() - current_clip_started
                                        )
                                        if clip_duration > 0:
                                            clip_durations.append(clip_duration)
                                        current_clip_started = 0.0
                            progress = 0.0
                            status_msg = ""

                            if step_name == "generate_outline":
                                step = GenerationStep.OUTLINE
                                progress = 0.5 if status == "started" else 1.0
                                status_msg = (
                                    "Creating outline..."
                                    if status == "started"
                                    else "Outline complete"
                                )
                            elif step_name == "generate_transcript":
                                step = GenerationStep.TRANSCRIPT
                                progress = 0.5 if status == "started" else 1.0
                                status_msg = (
                                    "Generating transcript..."
                                    if status == "started"
                                    else "Transcript complete"
                                )
                            elif step_name == "generate_clips":
                                step = GenerationStep.AUDIO
                                current = detail.get("current", 0)
                                total = detail.get("total", 1)
                                segment = detail.get("segment", {})
                                speaker = segment.get("speaker", "")

                                if status == "clip_started":
                                    progress = (
                                        max(0.0, (current - 1) / total)
                                        if total > 0
                                        else 0.0
                                    )
                                    status_msg = f"Working on clip {current}/{total}: {speaker}..."
                                elif status == "progress":
                                    progress = (
                                        min(1.0, max(0.0, current / total))
                                        if total > 0
                                        else 0.0
                                    )
                                    clip_status = segment.get("status", "")
                                    if clip_status == "success":
                                        status_msg = f"Completed clip {current}/{total}"
                                    elif clip_status == "error":
                                        status_msg = f"Clip {current}/{total} failed, continuing..."
                                    else:
                                        status_msg = (
                                            f"Generating audio: {current}/{total} clips"
                                        )
                                elif status == "completed":
                                    progress = 1.0
                                    status_msg = "Audio generation complete"
                                else:
                                    progress = 0.0
                                    status_msg = "Starting audio generation..."
                            elif step_name == "combine_audio":
                                step = GenerationStep.COMBINE
                                progress = 0.5 if status == "started" else 1.0
                                status_msg = (
                                    "Combining audio..."
                                    if status == "started"
                                    else "Audio combined"
                                )

                            now = time.monotonic()
                            if (now - last_emit_time) < 0.2:
                                return
                            last_emit_time = now

                            event_data = None
                            if detail.get("outline"):
                                event_data = {"outline": detail["outline"]}
                            elif detail.get("transcript"):
                                event_data = {"transcript": detail["transcript"]}

                            evt = _ProgressEvent(
                                step=step,
                                progress=progress,
                                status=status_msg,
                                detail=str(detail),
                                data=event_data,
                            )
                            try:
                                q.put_nowait(evt)
                            except queue.Full:
                                try:
                                    q.get_nowait()
                                    q.put_nowait(evt)
                                except (queue.Empty, queue.Full):
                                    pass

                        def worker():
                            try:
                                result = podcast_orchestrator.generate_podcast(
                                    content_input=content_input,
                                    voice_selections=voice_output,
                                    quality_preset=quality_preset,
                                    progress_callback=progress_callback,
                                    llm_config=llm_config,
                                )
                                q.put(_DoneEvent(result=result))
                            except Exception as e:
                                q.put(
                                    _ErrorEvent(
                                        error=format_user_error(e),
                                        tb=traceback.format_exc(),
                                    )
                                )

                        worker_thread = threading.Thread(target=worker, daemon=True)
                        worker_thread.start()

                        last_yield_time = time.monotonic()
                        outline_html = (
                            '<div class="empty-state">Generating outline...</div>'
                        )
                        transcript_html = (
                            '<div class="empty-state">Waiting for transcript...</div>'
                        )

                        def render_outline_html(outline_data: dict) -> str:
                            segments = outline_data.get("segments", [])
                            if not segments:
                                return '<div class="empty-state">No segments</div>'
                            html_parts = []
                            for i, seg in enumerate(segments):
                                title = html_escape.escape(str(seg.get("title", "Segment")))
                                desc = html_escape.escape(str(seg.get("description", "")))
                                html_parts.append(
                                    f'<div class="outline-item"><div class="outline-num">{i + 1}</div>'
                                    f'<div><div class="outline-title">{title}</div>'
                                    f'<div class="outline-desc">{desc}</div></div></div>'
                                )
                            return f'<div class="outline-list">{"".join(html_parts)}</div>'

                        yield (
                            create_step_indicator_html(GenerationStep.OUTLINE, 0.0),
                            0,
                            "Starting podcast generation...",
                            "",
                            "",
                            None,
                            outline_html,
                            transcript_html,
                            gr.update(visible=False),
                            gr.update(value="⏳ Generating...", interactive=False),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(visible=False),
                        )

                        def get_eta_string() -> str:
                            with eta_lock:
                                if len(clip_durations) < 2 or total_clips == 0:
                                    return ""
                                avg_clip_time = sum(clip_durations) / len(
                                    clip_durations
                                )
                                completed = len(clip_durations)
                                remaining = total_clips - completed
                                if remaining <= 0:
                                    return ""
                                eta_seconds = avg_clip_time * remaining
                            return f"~{_format_elapsed(eta_seconds)} remaining"

                        try:
                            while True:
                                try:
                                    item = q.get(timeout=0.5)
                                except queue.Empty:
                                    now = time.monotonic()
                                    if (now - last_yield_time) >= 1.5:
                                        elapsed = now - generation_started
                                        elapsed_str = (
                                            f"Elapsed: {_format_elapsed(elapsed)}"
                                        )
                                        eta_str = get_eta_string()
                                        time_info = f"{elapsed_str}  {eta_str}".strip()

                                        yield (
                                            create_step_indicator_html(
                                                current_step, step_progress
                                            ),
                                            calculate_overall_progress(
                                                current_step, step_progress
                                            ),
                                            f"{status_text}",
                                            time_info,
                                            "",
                                            None,
                                            outline_html,
                                            transcript_html,
                                            gr.update(visible=False),
                                            gr.update(
                                                value="⏳ Generating...",
                                                interactive=False,
                                            ),
                                            gr.update(),
                                            gr.update(),
                                            gr.update(),
                                            gr.update(),
                                        )
                                        last_yield_time = now
                                    continue

                                if isinstance(item, _ProgressEvent):
                                    current_step = item.step
                                    step_progress = item.progress
                                    status_text = item.status

                                    if item.data:
                                        if "outline" in item.data:
                                            outline_html = render_outline_html(
                                                item.data["outline"]
                                            )
                                        if "transcript" in item.data:
                                            transcript_html = (
                                                _render_podcast_transcript_html(
                                                    item.data["transcript"]
                                                )
                                            )

                                    now = time.monotonic()
                                    elapsed = now - generation_started
                                    elapsed_str = f"Elapsed: {_format_elapsed(elapsed)}"
                                    eta_str = get_eta_string()
                                    time_info = f"{elapsed_str}  {eta_str}".strip()

                                    yield (
                                        create_step_indicator_html(
                                            current_step, step_progress
                                        ),
                                        calculate_overall_progress(
                                            current_step, step_progress
                                        ),
                                        status_text,
                                        time_info,
                                        "",
                                        None,
                                        outline_html,
                                        transcript_html,
                                        gr.update(visible=False),
                                        gr.update(
                                            value="⏳ Generating...", interactive=False
                                        ),
                                        gr.update(),
                                        gr.update(),
                                        gr.update(),
                                        gr.update(),
                                    )
                                    last_yield_time = now
                                    continue

                                if isinstance(item, _DoneEvent):
                                    result = item.result
                                    combined_audio_path = result.get(
                                        "combined_audio_path"
                                    )
                                    outline_path = result.get("outline_path")
                                    transcript_path = result.get("transcript_path")
                                    podcast_dir = result.get("podcast_dir")

                                    if (
                                        "empty-state" in outline_html
                                        and outline_path
                                        and Path(outline_path).exists()
                                    ):
                                        outline_html = render_outline_html(
                                            read_json_file(outline_path)
                                        )

                                    transcript_data = None
                                    editor_rows = []
                                    if (
                                        transcript_path
                                        and Path(transcript_path).exists()
                                    ):
                                        transcript_data = read_json_file(transcript_path)
                                        if "empty-state" in transcript_html:
                                            transcript_html = (
                                                _render_podcast_transcript_html(
                                                    transcript_data
                                                )
                                            )
                                        dialogues = transcript_data.get("dialogues", [])
                                        editor_rows = [
                                            [
                                                dlg.get("speaker", ""),
                                                dlg.get("text", ""),
                                            ]
                                            for dlg in dialogues
                                        ]

                                    session_info = {
                                        "podcast_dir": podcast_dir,
                                        "quality_preset": quality_preset,
                                        "language": language,
                                    }

                                    done_text, done_html = _podcast_completion_status(
                                        result.get("failed_clips"),
                                        "Podcast generated successfully!",
                                        '<div style="color: #28a745;">Generation complete!</div>',
                                    )
                                    yield (
                                        create_step_indicator_html(
                                            GenerationStep.COMBINE, 1.0
                                        ),
                                        100,
                                        done_text,
                                        "",
                                        done_html,
                                        combined_audio_path,
                                        outline_html,
                                        transcript_html,
                                        gr.update(
                                            value=combined_audio_path, visible=True
                                        )
                                        if combined_audio_path
                                        else gr.update(visible=False),
                                        gr.update(
                                            value="Generate Podcast", interactive=True
                                        ),
                                        transcript_data,
                                        session_info,
                                        gr.update(value=editor_rows),
                                        gr.update(visible=True),
                                    )
                                    return

                                if isinstance(item, _ErrorEvent):
                                    print(f"[Podcast Error] {item.error}\n{item.tb}")
                                    yield (
                                        create_step_indicator_html(
                                            current_step, step_progress
                                        ),
                                        calculate_overall_progress(
                                            current_step, step_progress
                                        ),
                                        f"Error: {item.error}",
                                        "",
                                        f'<div style="color: #dc3545;">Generation failed: {item.error}</div>',
                                        None,
                                        outline_html,
                                        transcript_html,
                                        gr.update(visible=False),
                                        gr.update(
                                            value="Generate Podcast", interactive=True
                                        ),
                                        gr.update(),
                                        gr.update(),
                                        gr.update(),
                                        gr.update(),
                                    )
                                    return

                        finally:
                            cancel_event.set()

                    def regenerate_audio_from_edits(
                        editor_data,
                        session_state,
                        voice_selections,
                    ):
                        if not session_state:
                            yield (
                                create_step_indicator_html(GenerationStep.AUDIO, 0.0),
                                0,
                                "Error: No session data available",
                                "",
                                '<div style="color: #dc3545;">Generate a podcast first</div>',
                                None,
                                gr.update(visible=False),
                                gr.update(
                                    value="Regenerate Audio from Edits",
                                    interactive=True,
                                ),
                            )
                            return

                        # Handle pandas DataFrame or list-of-lists from Gradio
                        import pandas as pd

                        if isinstance(editor_data, pd.DataFrame):
                            editor_data = editor_data.values.tolist()

                        if editor_data is None or len(editor_data) == 0:
                            yield (
                                create_step_indicator_html(GenerationStep.AUDIO, 0.0),
                                0,
                                "Error: No transcript data to regenerate",
                                "",
                                '<div style="color: #dc3545;">Transcript is empty</div>',
                                None,
                                gr.update(visible=False),
                                gr.update(
                                    value="Regenerate Audio from Edits",
                                    interactive=True,
                                ),
                            )
                            return

                        podcast_dir = session_state.get("podcast_dir")
                        quality_preset = session_state.get("quality_preset", "standard")
                        language = session_state.get("language", "English")

                        if not podcast_dir or not Path(podcast_dir).exists():
                            yield (
                                create_step_indicator_html(GenerationStep.AUDIO, 0.0),
                                0,
                                "Error: Podcast directory not found",
                                "",
                                '<div style="color: #dc3545;">Session expired, generate a new podcast</div>',
                                None,
                                gr.update(visible=False),
                                gr.update(
                                    value="Regenerate Audio from Edits",
                                    interactive=True,
                                ),
                            )
                            return

                        voice_valid, voice_msg, voice_output = validate_selections(
                            voice_selections
                        )
                        if not voice_valid:
                            yield (
                                create_step_indicator_html(GenerationStep.AUDIO, 0.0),
                                0,
                                f"Error: {voice_msg}",
                                "",
                                f'<div style="color: #dc3545;">{voice_msg}</div>',
                                None,
                                gr.update(visible=False),
                                gr.update(
                                    value="Regenerate Audio from Edits",
                                    interactive=True,
                                ),
                            )
                            return

                        dialogues = [
                            {"speaker": row[0], "text": row[1]}
                            for row in editor_data
                            if len(row) >= 2 and row[0] and row[1]
                        ]

                        if not dialogues:
                            yield (
                                create_step_indicator_html(GenerationStep.AUDIO, 0.0),
                                0,
                                "Error: No valid dialogue entries",
                                "",
                                '<div style="color: #dc3545;">Add speaker and text to dialogues</div>',
                                None,
                                gr.update(visible=False),
                                gr.update(
                                    value="Regenerate Audio from Edits",
                                    interactive=True,
                                ),
                            )
                            return

                        yield (
                            create_step_indicator_html(GenerationStep.AUDIO, 0.0),
                            30,
                            "Regenerating audio from edited transcript...",
                            "",
                            "",
                            None,
                            gr.update(visible=False),
                            gr.update(value="⏳ Regenerating...", interactive=False),
                        )

                        try:
                            transcript = podcast_orchestrator.transcript_from_struct(
                                dialogues
                            )
                            speaker_profile = create_speaker_profile(voice_output)

                            transcript_path = Path(podcast_dir) / "transcript.json"
                            with open(transcript_path, "w") as f:
                                json.dump(transcript.model_dump(), f, indent=2)

                            q: queue.Queue = queue.Queue(maxsize=100)
                            result_holder: list = []
                            error_holder: list = []

                            def progress_cb(step: str, detail: dict | None):
                                if detail:
                                    q.put({"step": step, "detail": detail})

                            def worker():
                                try:
                                    clips, combined, failed_clips = (
                                        podcast_orchestrator.generate_audio_only(
                                            transcript=transcript,
                                            speaker_profile=speaker_profile,
                                            podcast_dir=Path(podcast_dir),
                                            quality_preset=quality_preset,
                                            language=language,
                                            progress_callback=progress_cb,
                                        )
                                    )
                                    result_holder.append((str(combined), failed_clips))
                                except Exception as e:
                                    error_holder.append(str(e))

                            worker_thread = threading.Thread(target=worker, daemon=True)
                            worker_thread.start()

                            while worker_thread.is_alive():
                                try:
                                    item = q.get(timeout=0.5)
                                    detail = item.get("detail", {})
                                    status = detail.get("status", "")
                                    current = detail.get("current", 0)
                                    total = detail.get("total", 1)

                                    if item.get("step") == "generate_clips":
                                        progress = int(
                                            30 + (current / max(total, 1)) * 50
                                        )
                                        yield (
                                            create_step_indicator_html(
                                                GenerationStep.AUDIO,
                                                current / max(total, 1),
                                            ),
                                            progress,
                                            f"Generating clip {current}/{total}...",
                                            "",
                                            "",
                                            None,
                                            gr.update(visible=False),
                                            gr.update(
                                                value="⏳ Regenerating...",
                                                interactive=False,
                                            ),
                                        )
                                    elif item.get("step") == "combine_audio":
                                        yield (
                                            create_step_indicator_html(
                                                GenerationStep.COMBINE, 0.5
                                            ),
                                            85,
                                            "Combining audio clips...",
                                            "",
                                            "",
                                            None,
                                            gr.update(visible=False),
                                            gr.update(
                                                value="⏳ Regenerating...",
                                                interactive=False,
                                            ),
                                        )
                                except queue.Empty:
                                    continue

                            worker_thread.join()

                            if error_holder:
                                yield (
                                    create_step_indicator_html(
                                        GenerationStep.AUDIO, 0.0
                                    ),
                                    0,
                                    f"Error: {error_holder[0]}",
                                    "",
                                    f'<div style="color: #dc3545;">Regeneration failed: {error_holder[0]}</div>',
                                    None,
                                    gr.update(visible=False),
                                    gr.update(
                                        value="Regenerate Audio from Edits",
                                        interactive=True,
                                    ),
                                )
                                return

                            if result_holder:
                                combined_path, failed_clips = result_holder[0]
                                done_text, done_html = _podcast_completion_status(
                                    failed_clips,
                                    "Audio regenerated successfully!",
                                    '<div style="color: #28a745;">Regeneration complete!</div>',
                                )
                                yield (
                                    create_step_indicator_html(
                                        GenerationStep.COMBINE, 1.0
                                    ),
                                    100,
                                    done_text,
                                    "",
                                    done_html,
                                    combined_path,
                                    gr.update(value=combined_path, visible=True),
                                    gr.update(
                                        value="Regenerate Audio from Edits",
                                        interactive=True,
                                    ),
                                )
                            else:
                                yield (
                                    create_step_indicator_html(
                                        GenerationStep.AUDIO, 0.0
                                    ),
                                    0,
                                    "Error: No audio generated",
                                    "",
                                    '<div style="color: #dc3545;">Unknown error occurred</div>',
                                    None,
                                    gr.update(visible=False),
                                    gr.update(
                                        value="Regenerate Audio from Edits",
                                        interactive=True,
                                    ),
                                )

                        except Exception as e:
                            yield (
                                create_step_indicator_html(GenerationStep.AUDIO, 0.0),
                                0,
                                f"Error: {str(e)}",
                                "",
                                f'<div style="color: #dc3545;">Regeneration failed: {str(e)}</div>',
                                None,
                                gr.update(visible=False),
                                gr.update(
                                    value="Regenerate Audio from Edits",
                                    interactive=True,
                                ),
                            )

                    def parse_custom_script(script_text, *slot_values):
                        """Fill the speaker slots with the names found in the script.

                        Voices already picked are kept: a slot that already
                        carries a script speaker's name moves with that name,
                        and an unnamed slot keeps its voice for the speaker
                        that lands in it.
                        """
                        result = parse_script(script_text)

                        if not result.ok:
                            error_html = f'<div style="color: #dc3545;">{"; ".join(result.errors)}</div>'
                            return (error_html, *[gr.update()] * len(slot_values))

                        speakers = result.speakers
                        count = len(speakers)
                        status_html = (
                            f'<div style="color: #28a745;">Found {count} speaker'
                            f"{'s' if count != 1 else ''}: {', '.join(speakers)} "
                            f"({len(result.dialogues)} dialogue lines)</div>"
                        )

                        current = [
                            (
                                (slot_values[i * 3] or "").strip(),
                                slot_values[i * 3 + 1],
                                slot_values[i * 3 + 2] or "",
                            )
                            for i in range(len(slot_values) // 3)
                        ]
                        by_name = {
                            name.lower(): (role, voice)
                            for name, role, voice in current
                            if name
                        }

                        outputs = [status_html]
                        for i, (cur_name, cur_role, cur_voice) in enumerate(current):
                            if i < count:
                                name = speakers[i]
                                if name.lower() in by_name:
                                    role, voice = by_name[name.lower()]
                                elif not cur_name:
                                    role, voice = cur_role, cur_voice
                                else:
                                    role, voice = _slot_roles[i], ""
                            else:
                                name, role, voice = "", _slot_roles[i], ""
                            outputs.extend([name, role, voice])

                        return tuple(outputs)

                    def on_custom_script_change():
                        return '<div style="color: #888; font-size: 0.9em;">Click "Parse Script" to detect speakers and fill their names</div>'

                    def build_custom_voice_summary(*slot_values):
                        names = []
                        ready = []
                        for i in range(len(slot_values) // 3):
                            name = (slot_values[i * 3] or "").strip()
                            voice_val = slot_values[i * 3 + 2] or ""
                            if voice_val and not name:
                                return (
                                    '<div style="color:#dc3545;">'
                                    f"Speaker {i + 1} has a voice but no name. "
                                    "Enter the name used in the script."
                                    "</div>"
                                )
                            if name:
                                names.append(name)
                                if voice_val:
                                    ready.append(name)

                        lowered = [n.lower() for n in names]
                        if len(set(lowered)) != len(lowered):
                            return (
                                '<div style="color:#dc3545;">'
                                "Speaker names must be unique."
                                "</div>"
                            )

                        count = len(ready)
                        if count == 0:
                            return '<div style="color:#888;">Select 1-4 speakers</div>'
                        if count == 1:
                            return (
                                '<div style="color:#28a745;">'
                                f"1 speaker selected ✓ (Narration mode): {ready[0]}"
                                "</div>"
                            )
                        return (
                            '<div style="color:#28a745;">'
                            f"{count} speakers selected ✓: {', '.join(ready)}"
                            "</div>"
                        )

                    def run_custom_script_generation(
                        script_text,
                        quality_preset,
                        language,
                        episode_title,
                        *custom_slot_values,
                    ):
                        outline_html = '<div class="empty-state">Custom script mode — no outline</div>'
                        transcript_html = (
                            '<div class="empty-state">Waiting for transcript...</div>'
                        )

                        def error_tuple(message: str, detail: str = ""):
                            return (
                                create_step_indicator_html(
                                    GenerationStep.TRANSCRIPT, 0.0
                                ),
                                0,
                                f"Error: {message}",
                                "",
                                f'<div style="color: #dc3545;">{detail or message}</div>',
                                None,
                                outline_html,
                                transcript_html,
                                gr.update(visible=False),
                                gr.update(
                                    value="Generate from Script", interactive=True
                                ),
                                None,
                                gr.update(),
                                gr.update(value=[]),
                                gr.update(visible=False),
                                gr.update(),
                            )

                        result = parse_script(script_text)
                        if not result.ok:
                            yield error_tuple(
                                "Invalid script", "; ".join(result.errors)
                            )
                            return

                        # Match script speakers to slots by name (case-insensitive,
                        # same as the clip generator); unused slots are ignored.
                        slots_by_name = {}
                        for i in range(len(custom_slot_values) // 3):
                            slot_name = (custom_slot_values[i * 3] or "").strip()
                            if not slot_name:
                                continue
                            if slot_name.lower() in slots_by_name:
                                yield error_tuple(
                                    f"Duplicate speaker name '{slot_name}'",
                                    "Each speaker slot needs a unique name.",
                                )
                                return
                            slots_by_name[slot_name.lower()] = (
                                custom_slot_values[i * 3 + 1],
                                custom_slot_values[i * 3 + 2],
                            )

                        voice_sels = []
                        for name in result.speakers:
                            slot = slots_by_name.get(name.lower())
                            if slot is None:
                                yield error_tuple(
                                    f"No speaker slot for '{name}'",
                                    f'Click "Parse Script" to fill speaker names, or type \'{name}\' into an empty speaker slot.',
                                )
                                return
                            role, voice_val = slot
                            if not voice_val:
                                yield error_tuple(
                                    f"Missing voice for {name}",
                                    f"Select a voice for '{name}' before generating.",
                                )
                                return
                            parts = voice_val.split(":", 1)
                            if len(parts) != 2:
                                yield error_tuple(
                                    f"Invalid voice selection for {name}",
                                    "Choose a preset or saved voice from the dropdown.",
                                )
                                return
                            vtype, vid = parts
                            if vtype not in {"preset", "saved"} or not vid:
                                yield error_tuple(
                                    f"Invalid voice selection for {name}",
                                    "Choose a preset or saved voice from the dropdown.",
                                )
                                return
                            voice_sels.append(
                                {
                                    "voice_id": vid,
                                    "name": name,
                                    "role": role,
                                    "type": vtype,
                                }
                            )

                        q: queue.Queue[_ProgressEvent | _DoneEvent | _ErrorEvent] = (
                            queue.Queue(maxsize=500)
                        )
                        cancel_event = threading.Event()

                        current_step = GenerationStep.TRANSCRIPT
                        step_progress = 0.0
                        status_text = "Starting..."

                        generation_started = time.monotonic()

                        def progress_callback(
                            step_name: str, detail: dict[str, Any] | None
                        ):
                            if cancel_event.is_set():
                                return

                            detail = detail or {}
                            status = detail.get("status", "")
                            step = GenerationStep.TRANSCRIPT
                            progress = 0.0
                            status_msg = ""
                            event_data = None

                            if step_name in {
                                "create_directory",
                                "generate_transcript",
                                "save_artifacts",
                            }:
                                step = GenerationStep.TRANSCRIPT
                                if step_name == "create_directory":
                                    progress = 0.2 if status == "completed" else 0.05
                                    status_msg = "Preparing podcast workspace..."
                                elif step_name == "generate_transcript":
                                    progress = 1.0 if status == "completed" else 0.6
                                    status_msg = (
                                        "Building transcript from script..."
                                        if status == "started"
                                        else "Transcript ready"
                                    )
                                    if detail.get("transcript"):
                                        event_data = {
                                            "transcript": detail["transcript"]
                                        }
                                else:
                                    progress = 1.0
                                    status_msg = "Transcript saved"
                            elif step_name == "generate_clips":
                                step = GenerationStep.AUDIO
                                current = detail.get("current", 0)
                                total = detail.get("total", 1)
                                segment = detail.get("segment", {})
                                speaker = segment.get("speaker", "")

                                if status == "clip_started":
                                    progress = (
                                        max(0.0, (current - 1) / total)
                                        if total > 0
                                        else 0.0
                                    )
                                    status_msg = f"Working on clip {current}/{total}: {speaker}..."
                                elif status == "progress":
                                    progress = (
                                        min(1.0, max(0.0, current / total))
                                        if total > 0
                                        else 0.0
                                    )
                                    clip_status = segment.get("status", "")
                                    if clip_status == "success":
                                        status_msg = f"Completed clip {current}/{total}"
                                    elif clip_status == "error":
                                        status_msg = f"Clip {current}/{total} failed, continuing..."
                                    else:
                                        status_msg = (
                                            f"Generating audio: {current}/{total} clips"
                                        )
                                elif status == "completed":
                                    progress = 1.0
                                    status_msg = "Audio generation complete"
                                else:
                                    progress = 0.0
                                    status_msg = "Starting audio generation..."
                            elif step_name in {"combine_audio", "save_metadata"}:
                                step = GenerationStep.COMBINE
                                progress = 1.0 if status == "completed" else 0.5
                                status_msg = (
                                    "Combining audio..."
                                    if step_name == "combine_audio"
                                    else "Saving metadata..."
                                )

                            evt = _ProgressEvent(
                                step=step,
                                progress=progress,
                                status=status_msg,
                                detail=str(detail),
                                data=event_data,
                            )
                            try:
                                q.put_nowait(evt)
                            except queue.Full:
                                try:
                                    q.get_nowait()
                                    q.put_nowait(evt)
                                except (queue.Empty, queue.Full):
                                    pass

                        def worker():
                            try:
                                result_data = (
                                    podcast_orchestrator.generate_podcast_from_script(
                                        dialogues=result.dialogues,
                                        voice_selections=voice_sels,
                                        quality_preset=quality_preset,
                                        language=language,
                                        title=episode_title,
                                        progress_callback=progress_callback,
                                    )
                                )
                                try:
                                    q.put_nowait(_DoneEvent(result=result_data))
                                except queue.Full:
                                    try:
                                        q.get_nowait()
                                        q.put_nowait(_DoneEvent(result=result_data))
                                    except (queue.Empty, queue.Full):
                                        pass
                            except Exception as e:
                                if cancel_event.is_set():
                                    return
                                err_evt = _ErrorEvent(
                                    error=format_user_error(e),
                                    tb=traceback.format_exc(),
                                )
                                try:
                                    q.put_nowait(err_evt)
                                except queue.Full:
                                    try:
                                        q.get_nowait()
                                        q.put_nowait(err_evt)
                                    except (queue.Empty, queue.Full):
                                        pass

                        worker_thread = threading.Thread(target=worker, daemon=True)
                        worker_thread.start()

                        yield (
                            create_step_indicator_html(GenerationStep.TRANSCRIPT, 0.0),
                            0,
                            "Starting custom script generation...",
                            "",
                            "",
                            None,
                            outline_html,
                            transcript_html,
                            gr.update(visible=False),
                            gr.update(value="⏳ Generating...", interactive=False),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(visible=False),
                            gr.update(),
                        )

                        try:
                            while True:
                                try:
                                    item = q.get(timeout=0.5)
                                except queue.Empty:
                                    elapsed = time.monotonic() - generation_started
                                    yield (
                                        create_step_indicator_html(
                                            current_step, step_progress
                                        ),
                                        calculate_overall_progress(
                                            current_step, step_progress
                                        ),
                                        status_text,
                                        f"Elapsed: {_format_elapsed(elapsed)}",
                                        "",
                                        None,
                                        outline_html,
                                        transcript_html,
                                        gr.update(visible=False),
                                        gr.update(
                                            value="⏳ Generating...", interactive=False
                                        ),
                                        gr.update(),
                                        gr.update(),
                                        gr.update(),
                                        gr.update(),
                                        gr.update(),
                                    )
                                    continue

                                if isinstance(item, _ProgressEvent):
                                    current_step = item.step
                                    step_progress = item.progress
                                    status_text = item.status

                                    if item.data and "transcript" in item.data:
                                        transcript_html = (
                                            _render_podcast_transcript_html(
                                                item.data["transcript"]
                                            )
                                        )

                                    elapsed = time.monotonic() - generation_started
                                    yield (
                                        create_step_indicator_html(
                                            current_step, step_progress
                                        ),
                                        calculate_overall_progress(
                                            current_step, step_progress
                                        ),
                                        status_text,
                                        f"Elapsed: {_format_elapsed(elapsed)}",
                                        "",
                                        None,
                                        outline_html,
                                        transcript_html,
                                        gr.update(visible=False),
                                        gr.update(
                                            value="⏳ Generating...", interactive=False
                                        ),
                                        gr.update(),
                                        gr.update(),
                                        gr.update(),
                                        gr.update(),
                                        gr.update(),
                                    )
                                    continue

                                if isinstance(item, _DoneEvent):
                                    result_data = item.result
                                    combined_audio_path = result_data.get(
                                        "combined_audio_path"
                                    )
                                    transcript_path = result_data.get("transcript_path")
                                    podcast_dir = result_data.get("podcast_dir")

                                    transcript_data = None
                                    editor_rows = []
                                    if (
                                        transcript_path
                                        and Path(transcript_path).exists()
                                    ):
                                        transcript_data = read_json_file(transcript_path)
                                        transcript_html = (
                                            _render_podcast_transcript_html(
                                                transcript_data
                                            )
                                        )
                                        dialogues = transcript_data.get("dialogues", [])
                                        editor_rows = [
                                            [
                                                dlg.get("speaker", ""),
                                                dlg.get("text", ""),
                                            ]
                                            for dlg in dialogues
                                        ]

                                    session_info = {
                                        "podcast_dir": podcast_dir,
                                        "quality_preset": quality_preset,
                                        "language": language,
                                    }

                                    done_text, done_html = _podcast_completion_status(
                                        result_data.get("failed_clips"),
                                        "Podcast generated successfully!",
                                        '<div style="color: #28a745;">Generation complete!</div>',
                                    )
                                    yield (
                                        create_step_indicator_html(
                                            GenerationStep.COMBINE, 1.0
                                        ),
                                        100,
                                        done_text,
                                        "",
                                        done_html,
                                        combined_audio_path,
                                        outline_html,
                                        transcript_html,
                                        gr.update(
                                            value=combined_audio_path, visible=True
                                        )
                                        if combined_audio_path
                                        else gr.update(visible=False),
                                        gr.update(
                                            value="Generate from Script",
                                            interactive=True,
                                        ),
                                        transcript_data,
                                        session_info,
                                        gr.update(value=editor_rows),
                                        gr.update(visible=True),
                                        voice_sels,
                                    )
                                    return

                                if isinstance(item, _ErrorEvent):
                                    print(f"[Podcast Error] {item.error}\n{item.tb}")
                                    yield (
                                        create_step_indicator_html(
                                            current_step, step_progress
                                        ),
                                        calculate_overall_progress(
                                            current_step, step_progress
                                        ),
                                        f"Error: {item.error}",
                                        "",
                                        f'<div style="color: #dc3545;">Generation failed: {item.error}</div>',
                                        None,
                                        outline_html,
                                        transcript_html,
                                        gr.update(visible=False),
                                        gr.update(
                                            value="Generate from Script",
                                            interactive=True,
                                        ),
                                        None,
                                        gr.update(),
                                        gr.update(value=[]),
                                        gr.update(visible=False),
                                        gr.update(),
                                    )
                                    return
                        finally:
                            cancel_event.set()

                    podcast_topic.change(
                        fn=update_topic_char_count,
                        inputs=[podcast_topic],
                        outputs=[podcast_topic_chars],
                    )

                    all_slot_inputs = []
                    for slot_role, slot_voice in podcast_speaker_slots:
                        all_slot_inputs.extend([slot_role, slot_voice])

                    for slot_role, slot_voice in podcast_speaker_slots:
                        slot_role.change(
                            fn=build_voice_selections_from_slots,
                            inputs=all_slot_inputs,
                            outputs=[
                                podcast_voice_selections_state,
                                podcast_voice_summary,
                            ],
                        )
                        slot_voice.change(
                            fn=build_voice_selections_from_slots,
                            inputs=all_slot_inputs,
                            outputs=[
                                podcast_voice_selections_state,
                                podcast_voice_summary,
                            ],
                        )

                    podcast_llm_provider.change(
                        fn=on_llm_provider_change,
                        inputs=[podcast_llm_provider],
                        outputs=[
                            podcast_llm_model,
                            podcast_llm_api_key,
                            podcast_llm_base_url,
                            podcast_llm_status,
                        ],
                    )

                    podcast_llm_test_btn.click(
                        fn=test_llm_connection,
                        inputs=[
                            podcast_llm_provider,
                            podcast_llm_model,
                            podcast_llm_api_key,
                            podcast_llm_base_url,
                        ],
                        outputs=[podcast_llm_status],
                    )

                    podcast_generate_btn.click(
                        fn=run_podcast_generation,
                        inputs=[
                            podcast_topic,
                            podcast_key_points,
                            podcast_briefing,
                            podcast_num_segments,
                            podcast_voice_selections_state,
                            podcast_quality_preset,
                            podcast_language,
                            podcast_llm_provider,
                            podcast_llm_model,
                            podcast_llm_api_key,
                            podcast_llm_base_url,
                        ],
                        outputs=[
                            podcast_step_indicator,
                            podcast_overall_progress,
                            podcast_status,
                            podcast_time_remaining,
                            podcast_error_display,
                            podcast_final_audio,
                            podcast_outline_html,
                            podcast_transcript_html,
                            podcast_download,
                            podcast_generate_btn,
                            podcast_transcript_state,
                            podcast_session_state,
                            podcast_transcript_editor,
                            podcast_edit_accordion,
                        ],
                        concurrency_limit=1,
                        concurrency_id="podcast_generation",
                    )

                    podcast_regenerate_btn.click(
                        fn=regenerate_audio_from_edits,
                        inputs=[
                            podcast_transcript_editor,
                            podcast_session_state,
                            podcast_voice_selections_state,
                        ],
                        outputs=[
                            podcast_step_indicator,
                            podcast_overall_progress,
                            podcast_status,
                            podcast_time_remaining,
                            podcast_error_display,
                            podcast_final_audio,
                            podcast_download,
                            podcast_regenerate_btn,
                        ],
                        concurrency_limit=1,
                        concurrency_id="podcast_regeneration",
                    )

                    custom_slot_inputs = [
                        component
                        for slot in custom_speaker_slots
                        for component in slot
                    ]

                    custom_script_parse_btn.click(
                        fn=parse_custom_script,
                        inputs=[custom_script_input] + custom_slot_inputs,
                        outputs=[custom_script_status] + custom_slot_inputs,
                    )

                    custom_script_input.change(
                        fn=on_custom_script_change,
                        inputs=[],
                        outputs=[custom_script_status],
                    )

                    for csn, _, csv in custom_speaker_slots:
                        for component in (csn, csv):
                            component.change(
                                fn=build_custom_voice_summary,
                                inputs=custom_slot_inputs,
                                outputs=[custom_voice_summary],
                                show_progress="hidden",
                            )

                    custom_refresh_voices_btn.click(
                        fn=lambda: [
                            gr.update(choices=_get_podcast_voice_choices())
                            for _ in custom_speaker_slots
                        ],
                        outputs=[slot[2] for slot in custom_speaker_slots],
                    )

                    for i, preview_btn in enumerate(custom_preview_buttons):
                        preview_btn.click(
                            fn=play_podcast_preview,
                            inputs=[custom_speaker_slots[i][2]],
                            outputs=[custom_preview_audio],
                        )

                    custom_generate_btn.click(
                        fn=run_custom_script_generation,
                        inputs=[
                            custom_script_input,
                            custom_quality_preset,
                            custom_language,
                            custom_episode_title,
                        ]
                        + custom_slot_inputs,
                        outputs=[
                            podcast_step_indicator,
                            podcast_overall_progress,
                            podcast_status,
                            podcast_time_remaining,
                            podcast_error_display,
                            podcast_final_audio,
                            podcast_outline_html,
                            podcast_transcript_html,
                            podcast_download,
                            custom_generate_btn,
                            podcast_transcript_state,
                            podcast_session_state,
                            podcast_transcript_editor,
                            podcast_edit_accordion,
                            podcast_voice_selections_state,
                        ],
                        concurrency_limit=1,
                        concurrency_id="podcast_generation",
                    )

                    for i, preview_btn in enumerate(podcast_preview_buttons):
                        preview_btn.click(
                            fn=play_podcast_preview,
                            inputs=[podcast_speaker_slots[i][1]],
                            outputs=[podcast_preview_audio],
                        )

                    podcast_history_dropdown.change(
                        fn=load_podcast_history_item,
                        inputs=[podcast_history_dropdown],
                        outputs=[podcast_history_audio, podcast_history_metadata],
                    ).then(
                        fn=lambda: False,
                        outputs=[podcast_history_delete_confirm],
                    )

                    podcast_history_refresh.click(
                        fn=lambda: gr.update(choices=get_podcast_history_choices()),
                        inputs=[],
                        outputs=[podcast_history_dropdown],
                    )

                    podcast_history_delete.click(
                        fn=delete_podcast_history_item,
                        inputs=[
                            podcast_history_dropdown,
                            podcast_history_delete_confirm,
                        ],
                        outputs=[
                            podcast_history_metadata,
                            podcast_history_dropdown,
                            podcast_history_audio,
                            podcast_history_delete_confirm,
                        ],
                    )
                    podcast_history_favorite.click(
                        fn=toggle_favorite,
                        inputs=[podcast_history_dropdown],
                        outputs=[
                            podcast_history_metadata,
                            podcast_history_display,
                            podcast_history_dropdown,
                        ],
                    )
                    podcast_history_search.change(
                        fn=search_history,
                        inputs=[podcast_history_search, podcast_history_favorites],
                        outputs=[podcast_history_display],
                        show_progress="hidden",
                    )
                    podcast_history_favorites.change(
                        fn=search_history,
                        inputs=[podcast_history_search, podcast_history_favorites],
                        outputs=[podcast_history_display],
                        show_progress="hidden",
                    )

                with gr.TabItem("OpenAI API", id="openai_api") as api_tab:
                    _api_settings = load_api_settings()
                    _api_voice_choices = _get_api_voice_choices()
                    _api_rows = _api_link_rows(
                        _api_settings, {value for _, value in _api_voice_choices}
                    )

                    gr.HTML('<div class="section-header">Voice Links</div>')
                    gr.Markdown(
                        "*Link up to 4 of your voices or personas to OpenAI voice names. "
                        "Apps and chat tools that support OpenAI text-to-speech can then "
                        "speak with them: point the app at this server and pick a linked name. "
                        "API speech uses the model's recommended sampling settings, "
                        "not the Generation settings panel.*",
                        elem_classes=["info-text"],
                    )

                    api_link_inputs = []
                    for i, (row_name, row_voice, row_language) in enumerate(_api_rows):
                        with gr.Row(elem_classes=["speaker-row", "api-slot"]):
                            api_link_inputs += [
                                gr.Dropdown(
                                    choices=list(OPENAI_VOICES),
                                    value=row_name,
                                    label=f"OpenAI Voice {i + 1}",
                                    interactive=True,
                                ),
                                gr.Dropdown(
                                    choices=_api_voice_choices,
                                    value=row_voice,
                                    label="Studio Voice / Persona",
                                    interactive=True,
                                ),
                                gr.Dropdown(
                                    choices=LANGUAGES,
                                    value=row_language,
                                    label="Language",
                                    interactive=True,
                                ),
                            ]
                    api_voice_dropdowns = api_link_inputs[1::3]

                    with gr.Row():
                        api_save_btn = gr.Button("Save Settings", variant="primary")
                        api_refresh_btn = gr.Button("↻ Refresh Voices")
                    api_message = gr.HTML(value="")

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML('<div class="section-header">Server</div>')
                            api_status = gr.HTML(value=_api_status_html())
                            with gr.Row():
                                api_host = gr.Textbox(
                                    label="Host",
                                    value=_api_settings["host"],
                                    info="127.0.0.1: this PC only. 0.0.0.0: also other devices on your network.",
                                    scale=2,
                                )
                                api_port = gr.Number(
                                    label="Port",
                                    value=_api_settings["port"],
                                    precision=0,
                                    minimum=1,
                                    maximum=65535,
                                    scale=1,
                                )
                            api_key_box = gr.Textbox(
                                label="API Key (optional)",
                                value=_api_settings["api_key"],
                                type="password",
                                info="When set, apps must send this key. Leave empty to accept any key.",
                            )
                            api_autostart = gr.Checkbox(
                                label="Start the API server when the studio launches",
                                value=bool(_api_settings["autostart"]),
                            )
                            with gr.Row():
                                api_start_btn = gr.Button("Start Server", variant="primary")
                                api_stop_btn = gr.Button("Stop Server", variant="stop")

                        with gr.Column(scale=1):
                            gr.HTML('<div class="section-header">Connect Your App</div>')
                            api_usage = gr.Markdown(value=_api_usage_markdown(_api_settings))

                    gr.HTML('<div class="section-header">Test a Link</div>')
                    with gr.Row():
                        with gr.Column(scale=1):
                            _api_names = _api_linked_names(_api_settings)
                            api_test_voice = gr.Dropdown(
                                choices=_api_names,
                                value=_api_names[0] if _api_names else None,
                                label="Linked Voice",
                                info="Uses the saved links, like an API request",
                            )
                            api_test_speed = gr.Slider(
                                API_MIN_SPEED,
                                API_MAX_SPEED,
                                value=1.0,
                                step=0.05,
                                label="Speed",
                            )
                        with gr.Column(scale=2):
                            api_test_text = gr.Textbox(
                                label="Text to Speak",
                                value="Hello! This is my studio voice, speaking through the OpenAI-compatible API.",
                                lines=3,
                            )
                            api_test_btn = gr.Button(
                                "Generate Test",
                                variant="primary",
                                elem_classes=["generate-btn"],
                                size="lg",
                            )
                            api_test_status = gr.Textbox(
                                label="Status", interactive=False, value="Ready to generate..."
                            )
                            api_test_audio = gr.Audio(
                                label="Test Audio", type="filepath", interactive=False
                            )

                with gr.TabItem("History", id="history"):
                    gr.HTML('<div class="section-header">Generation History</div>')
                    gr.Markdown(
                        "*Browse, search, and replay your past voice generations.*"
                    )

                    with gr.Row():
                        hist_tab_filter = gr.Dropdown(
                            choices=["All", "Preset", "Clone", "Design", "Saved"],
                            value="All",
                            label="Source",
                            scale=1,
                        )
                        hist_search = gr.Textbox(
                            placeholder="Search history...",
                            show_label=False,
                            scale=3,
                        )
                        hist_favorites = gr.Checkbox(
                            label="Favorites only",
                            value=False,
                            scale=1,
                        )

                    hist_display = gr.HTML(
                        value=format_history_for_display(tab_type_filter="voice"),
                        elem_classes=["history-display"],
                    )
                    hist_init = get_history_initial(tab_type_filter="voice")
                    hist_dropdown = gr.Dropdown(
                        choices=hist_init[0],
                        value=hist_init[1],
                        label="Select to play",
                        allow_custom_value=False,
                    )
                    hist_text = gr.Textbox(
                        label="Text",
                        lines=2,
                        interactive=False,
                        value=hist_init[3],
                    )
                    hist_params_display = gr.Textbox(
                        label="Generation Settings",
                        lines=1,
                        interactive=False,
                        value=hist_init[4],
                    )
                    hist_audio = gr.Audio(
                        label="Playback",
                        type="filepath",
                        interactive=False,
                        value=hist_init[2],
                    )
                    with gr.Row(elem_classes=["mini-btn-row"]):
                        hist_refresh = gr.Button("Refresh", size="sm")
                        hist_apply = gr.Button("Apply Params", size="sm")
                        hist_favorite = gr.Button("★ Favorite", size="sm")
                        hist_delete = gr.Button("Delete", size="sm", variant="stop")
                        hist_export = gr.Button("Export ZIP", size="sm")
                    hist_delete_confirm = gr.State(False)
                    hist_export_file = gr.File(label="Download Export", visible=False)

        with gr.Column(
            scale=1, min_width=320, elem_classes=["params-col", "compact-params-panel"]
        ):
            gr.HTML(
                '<div class="params-title">Generation settings</div>'
                '<div class="params-sub">Shared by every generation tab</div>'
            )
            save_indicator = gr.HTML(
                value='<span class="save-indicator">Settings saved</span>',
                elem_classes=["save-indicator-wrap"],
            )

            gr.HTML('<div class="params-label">Quick presets</div>')
            with gr.Row(elem_classes=["preset-btn-group"]):
                preset_fast = gr.Button(
                    "Fast", size="sm", min_width=0, elem_classes=["preset-btn-lg"]
                )
                preset_balanced = gr.Button(
                    "Balanced", size="sm", min_width=0, elem_classes=["preset-btn-lg"]
                )
                preset_quality = gr.Button(
                    "Quality", size="sm", min_width=0, elem_classes=["preset-btn-lg"]
                )
            reset_btn = gr.Button(
                "Reset to defaults",
                size="sm",
                variant="secondary",
                elem_classes=["reset-btn"],
            )

            with gr.Accordion("Basic Parameters", open=True):
                param_temp = gr.Slider(
                    0.1,
                    1.5,
                    value=min(settings["temperature"], 1.5),
                    step=0.05,
                    label="Temperature",
                    info=PARAM_TOOLTIPS["temperature"],
                )

                with gr.Row(elem_classes=["compact-slider-row"]):
                    param_top_k = gr.Slider(
                        1,
                        100,
                        value=settings["top_k"],
                        step=1,
                        label="Top-K",
                        info=PARAM_TOOLTIPS["top_k"],
                    )
                    param_top_p = gr.Slider(
                        0.1,
                        1.0,
                        value=settings["top_p"],
                        step=0.05,
                        label="Top-P",
                        info=PARAM_TOOLTIPS["top_p"],
                    )

            with gr.Accordion("Advanced Parameters", open=False):
                with gr.Row(elem_classes=["compact-slider-row"]):
                    param_rep_pen = gr.Slider(
                        1.0,
                        2.0,
                        value=settings["repetition_penalty"],
                        step=0.01,
                        label="Repetition Penalty",
                        info=PARAM_TOOLTIPS["repetition_penalty"],
                    )
                    param_max_tokens = gr.Slider(
                        256,
                        8192,
                        value=settings["max_new_tokens"],
                        step=256,
                        label="Auto Max Tokens",
                        info=PARAM_TOOLTIPS["max_new_tokens"],
                        interactive=False,
                    )

                gr.HTML(
                    '<div class="params-note">Subtalker model - defaults recommended</div>'
                )

                with gr.Row(elem_classes=["compact-slider-row"]):
                    param_sub_temp = gr.Slider(
                        0.1,
                        1.5,
                        value=min(settings["subtalker_temperature"], 1.5),
                        step=0.05,
                        label="Sub Temperature",
                        info=PARAM_TOOLTIPS["subtalker_temperature"],
                    )
                    param_sub_top_k = gr.Slider(
                        1,
                        100,
                        value=settings["subtalker_top_k"],
                        step=1,
                        label="Sub Top-K",
                        info=PARAM_TOOLTIPS["subtalker_top_k"],
                    )

                param_sub_top_p = gr.Slider(
                    0.1,
                    1.0,
                    value=settings["subtalker_top_p"],
                    step=0.05,
                    label="Sub Top-P",
                    info=PARAM_TOOLTIPS["subtalker_top_p"],
                )

    all_param_sliders = [
        param_temp,
        param_top_k,
        param_top_p,
        param_rep_pen,
        param_max_tokens,
        param_sub_temp,
        param_sub_top_k,
        param_sub_top_p,
    ]

    cv_text.change(fn=update_char_count, inputs=[cv_text], outputs=[cv_char_count])
    vc_test_text.change(
        fn=update_char_count, inputs=[vc_test_text], outputs=[vc_test_char_count]
    )
    sv_text.change(fn=update_char_count, inputs=[sv_text], outputs=[sv_char_count])

    for slider in all_param_sliders:
        slider.change(
            fn=on_param_change, inputs=all_param_sliders, outputs=[save_indicator]
        )

    preset_fast.click(
        fn=lambda: apply_preset("fast"), outputs=all_param_sliders + [save_indicator]
    )
    preset_balanced.click(
        fn=lambda: apply_preset("balanced"),
        outputs=all_param_sliders + [save_indicator],
    )
    preset_quality.click(
        fn=lambda: apply_preset("quality"), outputs=all_param_sliders + [save_indicator]
    )

    reset_btn.click(fn=reset_params, outputs=all_param_sliders + [save_indicator])

    podcast_quality_preset.change(
        fn=lambda preset: apply_podcast_preset(preset)[:-1],
        inputs=[podcast_quality_preset],
        outputs=all_param_sliders + [podcast_num_segments],
    )

    cv_btn.click(
        fn=_disable_btn,
        outputs=[cv_btn],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=generate_custom_voice,
        inputs=[cv_text, cv_model, cv_speaker, cv_language, cv_instruct]
        + all_param_sliders,
        outputs=[cv_audio, cv_status],
        concurrency_limit=1,
        concurrency_id="generation",
    ).then(
        fn=_enable_btn,
        outputs=[cv_btn],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=_refresh_history_on_success,
        inputs=[cv_audio, hist_search, hist_favorites, hist_tab_filter],
        outputs=[hist_display, hist_dropdown, hist_audio, hist_text, hist_params_display],
        show_progress="hidden",
    )

    vd_text.change(fn=update_char_count, inputs=[vd_text], outputs=[vd_char_count])

    vd_btn.click(
        fn=_disable_btn,
        outputs=[vd_btn],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=generate_voice_design,
        inputs=[vd_text, vd_description, vd_language] + all_param_sliders,
        outputs=[vd_audio, vd_status],
        concurrency_limit=1,
        concurrency_id="generation",
    ).then(
        fn=_enable_btn,
        outputs=[vd_btn],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=_refresh_history_on_success,
        inputs=[vd_audio, hist_search, hist_favorites, hist_tab_filter],
        outputs=[hist_display, hist_dropdown, hist_audio, hist_text, hist_params_display],
        show_progress="hidden",
    )

    hist_dropdown.change(
        fn=play_history_item_with_details,
        inputs=[hist_dropdown],
        outputs=[hist_audio, hist_text, hist_params_display],
        concurrency_id="history",
        concurrency_limit=None,
        show_progress="hidden",
    ).then(
        fn=lambda: False,
        outputs=[hist_delete_confirm],
    )
    hist_refresh.click(
        fn=search_history_filtered,
        inputs=[hist_search, hist_favorites, hist_tab_filter],
        outputs=[hist_display, hist_dropdown, hist_audio, hist_text, hist_params_display],
        concurrency_id="history",
        concurrency_limit=None,
        show_progress="hidden",
    )
    hist_apply.click(
        fn=apply_history_params,
        inputs=[hist_dropdown],
        outputs=all_param_sliders + [save_indicator],
        concurrency_id="history",
        concurrency_limit=None,
    )
    hist_delete.click(
        fn=history_tab_delete,
        inputs=[hist_dropdown, hist_delete_confirm, hist_search, hist_favorites, hist_tab_filter],
        outputs=[
            hist_params_display,
            hist_dropdown,
            hist_audio,
            hist_delete_confirm,
            hist_display,
            hist_text,
        ],
        concurrency_id="history",
        concurrency_limit=None,
    )
    hist_favorite.click(
        fn=history_tab_favorite,
        inputs=[hist_dropdown, hist_search, hist_favorites, hist_tab_filter],
        outputs=[hist_params_display, hist_display, hist_dropdown, hist_audio, hist_text],
        concurrency_id="history",
        concurrency_limit=None,
    )
    hist_search.change(
        fn=search_history_filtered,
        inputs=[hist_search, hist_favorites, hist_tab_filter],
        outputs=[hist_display, hist_dropdown, hist_audio, hist_text, hist_params_display],
        concurrency_id="history",
        concurrency_limit=None,
        show_progress="hidden",
    )
    hist_favorites.change(
        fn=search_history_filtered,
        inputs=[hist_search, hist_favorites, hist_tab_filter],
        outputs=[hist_display, hist_dropdown, hist_audio, hist_text, hist_params_display],
        concurrency_id="history",
        concurrency_limit=None,
        show_progress="hidden",
    )
    hist_tab_filter.change(
        fn=search_history_filtered,
        inputs=[hist_search, hist_favorites, hist_tab_filter],
        outputs=[hist_display, hist_dropdown, hist_audio, hist_text, hist_params_display],
        concurrency_id="history",
        concurrency_limit=None,
        show_progress="hidden",
    )
    hist_export.click(
        fn=export_history_to_zip,
        outputs=[hist_export_file, hist_params_display],
    )

    def build_transcripts_json(files, t1, t2, t3):
        transcripts = {}
        if files:
            for i, f in enumerate(files[:3]):
                if f is None:
                    continue
                path = extract_file_path(f)
                if not path:
                    continue
                transcript = [t1, t2, t3][i] if i < 3 else ""
                transcripts[path] = transcript.strip() if transcript else ""
        return json.dumps(transcripts)

    def auto_transcribe_first_sample(files):
        if not files:
            return ""
        first_file = files[0] if isinstance(files, list) else files
        path = extract_file_path(first_file)
        if not path:
            return ""
        return auto_transcribe_audio(path)

    vc_ref_audio.change(
        fn=flush_transcripts_to_state,
        inputs=[
            vc_transcript_state,
            vc_current_file_paths,
            vc_transcript_1,
            vc_transcript_2,
            vc_transcript_3,
        ],
        outputs=[vc_transcript_state],
    ).then(
        fn=update_transcript_fields,
        inputs=[vc_ref_audio, vc_transcript_state],
        outputs=[
            vc_transcripts_info,
            vc_transcript_1,
            vc_transcript_2,
            vc_transcript_3,
            vc_auto_transcribe_btn,
            vc_transcript_state,
            vc_current_file_paths,
        ],
    ).then(
        fn=analyze_uploaded_samples,
        inputs=[vc_ref_audio],
        outputs=[vc_samples_summary, vc_samples_warnings],
    )

    for transcript_box in [vc_transcript_1, vc_transcript_2, vc_transcript_3]:
        transcript_box.change(
            fn=flush_transcripts_to_state,
            inputs=[
                vc_transcript_state,
                vc_current_file_paths,
                vc_transcript_1,
                vc_transcript_2,
                vc_transcript_3,
            ],
            outputs=[vc_transcript_state],
        )

    vc_clone_btn.click(
        fn=_disable_btn,
        outputs=[vc_clone_btn],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=flush_transcripts_to_state,
        inputs=[
            vc_transcript_state,
            vc_current_file_paths,
            vc_transcript_1,
            vc_transcript_2,
            vc_transcript_3,
        ],
        outputs=[vc_transcript_state],
    ).then(
        fn=lambda files, t1, t2, t3: build_transcripts_json(files, t1, t2, t3),
        inputs=[vc_ref_audio, vc_transcript_1, vc_transcript_2, vc_transcript_3],
        outputs=[vc_transcripts_json],
    ).then(
        fn=clone_voice_multi,
        inputs=[
            vc_ref_audio,
            vc_transcripts_json,
            vc_model,
            vc_test_text,
            vc_language,
            vc_ref_language,
            vc_crosslingual_opt,
            vc_combine_samples,
        ]
        + all_param_sliders,
        outputs=[
            vc_output,
            current_prompt_data,
            current_clone_model,
            vc_samples_meta_json,
            vc_status,
        ],
        concurrency_limit=1,
        concurrency_id="generation",
    ).then(
        fn=lambda: _enable_btn("Clone & Generate"),
        outputs=[vc_clone_btn],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=_refresh_history_on_success,
        inputs=[vc_output, hist_search, hist_favorites, hist_tab_filter],
        outputs=[hist_display, hist_dropdown, hist_audio, hist_text, hist_params_display],
        show_progress="hidden",
    )

    vc_auto_transcribe_btn.click(
        fn=auto_transcribe_first_sample,
        inputs=[vc_ref_audio],
        outputs=[vc_transcript_1],
    ).then(
        fn=flush_transcripts_to_state,
        inputs=[
            vc_transcript_state,
            vc_current_file_paths,
            vc_transcript_1,
            vc_transcript_2,
            vc_transcript_3,
        ],
        outputs=[vc_transcript_state],
    )

    vc_save_btn.click(
        fn=flush_transcripts_to_state,
        inputs=[
            vc_transcript_state,
            vc_current_file_paths,
            vc_transcript_1,
            vc_transcript_2,
            vc_transcript_3,
        ],
        outputs=[vc_transcript_state],
    ).then(
        fn=lambda files, t1, t2, t3: build_transcripts_json(files, t1, t2, t3),
        inputs=[vc_ref_audio, vc_transcript_1, vc_transcript_2, vc_transcript_3],
        outputs=[vc_transcripts_json],
    ).then(
        fn=save_cloned_voice_multi,
        inputs=[
            vc_name,
            vc_description,
            vc_style_note,
            vc_ref_language,
            vc_ref_audio,
            vc_transcripts_json,
            current_prompt_data,
            current_clone_model,
            vc_samples_meta_json,
        ],
        outputs=[vc_save_status, sv_voice_dropdown],
    ).then(
        fn=lambda: [gr.update(choices=_get_podcast_voice_choices()) for _ in range(4)],
        outputs=[slot[1] for slot in podcast_speaker_slots],
    ).then(
        fn=lambda: gr.update(choices=_get_persona_voice_choices(), value=None),
        outputs=[persona_voice_dropdown],
    )

    sv_refresh_btn.click(
        fn=lambda: gr.update(choices=get_saved_voice_choices()),
        outputs=[sv_voice_dropdown],
    )

    sv_voice_dropdown.change(
        fn=get_voice_details,
        inputs=[sv_voice_dropdown],
        outputs=[
            sv_description,
            sv_style_note,
            sv_ref_text,
            sv_ref_language_info,
            sv_model_info,
            sv_ref_audio,
        ],
    ).then(fn=lambda: False, outputs=[sv_delete_confirm])

    sv_delete_btn.click(
        fn=delete_saved_voice,
        inputs=[sv_voice_dropdown, sv_delete_confirm],
        outputs=[sv_delete_status, sv_voice_dropdown, sv_ref_audio, sv_delete_confirm],
    )

    sv_generate_btn.click(
        fn=_disable_btn,
        outputs=[sv_generate_btn],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=generate_with_saved_voice,
        inputs=[sv_text, sv_voice_dropdown, sv_language, sv_crosslingual_opt]
        + all_param_sliders,
        outputs=[sv_audio, sv_status],
        concurrency_limit=1,
        concurrency_id="generation",
    ).then(
        fn=_enable_btn,
        outputs=[sv_generate_btn],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=_refresh_history_on_success,
        inputs=[sv_audio, hist_search, hist_favorites, hist_tab_filter],
        outputs=[hist_display, hist_dropdown, hist_audio, hist_text, hist_params_display],
        show_progress="hidden",
    )

    def refresh_podcast_voice_dropdowns():
        choices = _get_podcast_voice_choices()
        return [gr.update(choices=choices) for _ in range(4)]

    podcast_refresh_voices_btn.click(
        fn=refresh_podcast_voice_dropdowns,
        outputs=[slot[1] for slot in podcast_speaker_slots],
    )

    # The Generate buttons sit at the bottom of the (long) input column while the
    # progress / status / error messages are at the top of the right column. Bring
    # them into view when a run starts so the click never looks like a no-op.
    for _generate_btn in (podcast_generate_btn, custom_generate_btn):
        _generate_btn.click(
            fn=None,
            js="() => { const el = document.querySelector('.progress-anchor'); "
            "if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' }); }",
        )

    # OpenAI API tab
    _api_settings_inputs = api_link_inputs + [
        api_host,
        api_port,
        api_key_box,
        api_autostart,
    ]
    for _api_save_event in (api_save_btn.click, api_autostart.input):
        _api_save_event(
            fn=api_save_settings,
            inputs=_api_settings_inputs + [api_test_voice],
            outputs=[api_message, api_usage, api_test_voice],
        )
    api_start_btn.click(
        fn=api_start_server,
        inputs=_api_settings_inputs + [api_test_voice],
        outputs=[api_status, api_message, api_usage, api_test_voice],
    )
    api_stop_btn.click(fn=api_stop_server, outputs=[api_status, api_message])
    api_refresh_btn.click(
        fn=api_refresh_voices, inputs=api_voice_dropdowns, outputs=api_voice_dropdowns
    )
    api_tab.select(
        fn=api_on_tab_select,
        inputs=api_voice_dropdowns,
        outputs=[api_status] + api_voice_dropdowns,
        show_progress="hidden",
    )
    api_test_btn.click(
        fn=_disable_btn,
        outputs=[api_test_btn],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=api_test_link,
        inputs=[api_test_voice, api_test_text, api_test_speed],
        outputs=[api_test_audio, api_test_status],
        concurrency_limit=1,
        concurrency_id="generation",
    ).then(
        fn=lambda: _enable_btn("Generate Test"),
        outputs=[api_test_btn],
        queue=False,
        show_progress="hidden",
    )

if __name__ == "__main__":
    print("Starting Qwen3-TTS Studio...")
    print("=" * 50)
    print("Features:")
    print("  • Visible parameters with auto-save")
    print("  • Quick presets: Fast, Balanced, Quality")
    print("  • Character count with warnings")
    print("  • Generation time tracking")
    print("  • History with search & favorites")
    print("  • Export history to ZIP")
    print("=" * 50)
    demo.queue(default_concurrency_limit=1)

    _startup_api_settings = load_api_settings()
    if _startup_api_settings.get("autostart"):
        try:
            api_server.start(
                _startup_api_settings["host"], int(_startup_api_settings["port"])
            )
        except (RuntimeError, ValueError) as e:
            print(f"[API] OpenAI-compatible API server not started: {e}")

    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port)
