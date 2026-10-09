"""Audio generation for podcast dialogue using Qwen3-TTS."""

import copy
import json
import pickle
import threading
from contextlib import contextmanager
from math import ceil
from pathlib import Path
from typing import Any, Callable, Generator

import numpy as np
import soundfile as sf
import torch

# Timeout for TTS generation (10 minutes per clip for MPS/MacBook)
TTS_TIMEOUT_SECONDS = 600


@contextmanager
def timeout_handler(
    seconds: int, error_context: str = ""
) -> Generator[None, None, None]:
    """Context manager for timeout protection.

    Uses threading.Timer for thread-safe timeout support across all platforms.
    Note: This implementation sets a flag on timeout but cannot interrupt
    blocking operations. The timeout is checked after the operation completes.
    For true interruption of long-running TTS, the model itself would need
    timeout support.

    Args:
        seconds: Timeout duration in seconds.
        error_context: Additional context for error message (e.g., clip index, speaker).

    Yields:
        None

    Raises:
        TimeoutError: If the operation exceeds the timeout duration.
    """
    # Thread-safe timeout using threading.Timer (works in any thread)
    timeout_occurred = threading.Event()

    def timeout_trigger() -> None:
        timeout_occurred.set()

    timer = threading.Timer(seconds, timeout_trigger)
    timer.start()
    try:
        yield
        # Check if timeout occurred during the operation
        if timeout_occurred.is_set():
            raise TimeoutError(
                f"TTS generation timed out after {seconds}s. {error_context}"
            )
    finally:
        timer.cancel()


from podcast.models import Dialogue, SpeakerProfile, Transcript
from podcast.presets import MODEL_SAMPLING_DEFAULTS

SAVED_VOICES_DIR = Path("saved_voices")


def _get_model_dtype_device(model: Any) -> tuple[torch.dtype, torch.device]:
    """Get model's dtype and device from talker module (most reliable for Qwen3-TTS)."""
    hf = getattr(model, "model", model)
    talker = getattr(hf, "talker", None)

    target = talker if talker is not None else hf

    try:
        param = next(target.parameters())
        return param.dtype, param.device
    except (StopIteration, AttributeError):
        pass

    if torch.backends.mps.is_available():
        return torch.float16, torch.device("mps")
    elif torch.cuda.is_available():
        return torch.bfloat16, torch.device("cuda")
    return torch.float32, torch.device("cpu")


def _prepare_voice_clone_prompt(voice_clone_prompt: Any, model: Any) -> Any:
    """Normalize voice clone prompt dtype/device to match model."""
    model_dtype, model_device = _get_model_dtype_device(model)

    def convert_item(item: Any) -> Any:
        if hasattr(item, "ref_spk_embedding") and isinstance(
            item.ref_spk_embedding, torch.Tensor
        ):
            if (
                item.ref_spk_embedding.dtype != model_dtype
                or item.ref_spk_embedding.device != model_device
            ):
                item.ref_spk_embedding = item.ref_spk_embedding.to(
                    dtype=model_dtype, device=model_device
                )
        if hasattr(item, "ref_code") and isinstance(item.ref_code, torch.Tensor):
            if item.ref_code.device != model_device:
                item.ref_code = item.ref_code.to(device=model_device)
        return item

    if isinstance(voice_clone_prompt, list):
        return [convert_item(copy.copy(item)) for item in voice_clone_prompt]
    elif isinstance(voice_clone_prompt, dict):
        result = voice_clone_prompt.copy()
        if "ref_spk_embedding" in result:
            emb = result["ref_spk_embedding"]
            if isinstance(emb, list):
                result["ref_spk_embedding"] = [
                    e.to(dtype=model_dtype, device=model_device)
                    if isinstance(e, torch.Tensor)
                    else e
                    for e in emb
                ]
            elif isinstance(emb, torch.Tensor):
                result["ref_spk_embedding"] = emb.to(
                    dtype=model_dtype, device=model_device
                )
        if "ref_code" in result:
            code = result["ref_code"]
            if isinstance(code, list):
                result["ref_code"] = [
                    c.to(device=model_device) if isinstance(c, torch.Tensor) else c
                    for c in code
                ]
            elif isinstance(code, torch.Tensor):
                result["ref_code"] = code.to(device=model_device)
        return result
    else:
        return convert_item(copy.copy(voice_clone_prompt))


LANGUAGE_MAP = {
    "en": "english",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "es": "spanish",
}


def _normalize_language(lang: str) -> str:
    if lang in LANGUAGE_MAP:
        return LANGUAGE_MAP[lang]
    return lang


CHUNK_TARGET = 120
CHUNK_MAX = 150
CHUNK_MIN = 50

# Chunk quality checks. Qwen3-TTS can get stuck emitting silence frames
# mid-chunk (dead air the trailing trim cannot see) or return a chunk with no
# speech at all. Levels are 50ms-window RMS on float audio in [-1, 1].
SILENCE_WINDOW_SEC = 0.05
# Silence inside stuck runs measured <= 0.012 (~-38 dBFS); speech windows sit
# far above it.
SILENCE_ABS_THRESHOLD = 0.012
# A chunk whose loudest 1% of windows is below this has no speech in it
# (measured: real speech >= 0.089, failed chunks <= 0.025).
NO_SPEECH_LEVEL = 0.04
# Pauses at least this long are dead air, not natural phrasing.
LONG_SILENCE_SEC = 1.5
# Length each dead-air run is shortened to when every attempt has some.
COMPRESSED_SILENCE_SEC = 0.5
# Attempts per chunk before giving up (no seed is set, so retries differ).
CHUNK_ATTEMPTS = 3

# Token budget: the 12Hz codec emits 12.5 frames per second of audio.
CODEC_FRAMES_PER_SEC = 12.5
MAX_TOKENS_HEADROOM = 2.5
MAX_TOKENS_PAD = 24


def _split_text_into_chunks(text: str) -> list[str]:
    """Split long text into sentence-based chunks for TTS generation."""
    text = text.strip()
    if len(text) <= CHUNK_MAX:
        return [text]

    import re

    sentences = re.split(r"(?<=[.!?。！？])\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(sentence) > CHUNK_MAX:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            words = sentence.split()
            temp = ""
            for word in words:
                if len(temp) + len(word) + 1 <= CHUNK_TARGET:
                    temp = f"{temp} {word}".strip()
                else:
                    if temp:
                        chunks.append(temp)
                    temp = word
            if temp:
                chunks.append(temp)
            continue

        if len(current_chunk) + len(sentence) + 1 <= CHUNK_TARGET:
            current_chunk = f"{current_chunk} {sentence}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    merged = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        while i + 1 < len(chunks) and len(chunk) < CHUNK_MIN:
            i += 1
            chunk = f"{chunk} {chunks[i]}"
        merged.append(chunk.strip())
        i += 1

    return merged if merged else [text]


def _window_rms(audio: np.ndarray, sr: int, window_sec: float) -> tuple[np.ndarray, int]:
    """Return per-window RMS levels and the window size in samples."""
    window = max(1, int(sr * window_sec))
    n = len(audio) // window
    if n == 0:
        return np.zeros(0, dtype=np.float32), window
    rms = np.sqrt(np.mean(audio[: n * window].reshape(n, window) ** 2, axis=1))
    return rms, window


def _speech_level(window_rms: np.ndarray) -> float:
    """Loudness of the speech in a chunk: the 99th-percentile window RMS."""
    if window_rms.size == 0:
        return 0.0
    return float(np.percentile(window_rms, 99))


def _silence_threshold(window_rms: np.ndarray) -> float:
    """Window RMS below which audio counts as silence.

    Absolute so that hissy "silence" from noisy voice references is caught,
    but scaled down for unusually quiet voices so their speech is not.
    """
    return min(SILENCE_ABS_THRESHOLD, 0.1 * _speech_level(window_rms))


def _find_long_silences(
    audio: np.ndarray,
    sr: int,
    min_sec: float = LONG_SILENCE_SEC,
) -> list[tuple[int, int]]:
    """
    Find leading or mid-chunk silences of at least ``min_sec``.

    Returns (start, end) sample ranges. A run that reaches the end of the
    audio is trailing silence, which _trim_trailing_silence handles.
    """
    rms, window = _window_rms(audio, sr, SILENCE_WINDOW_SEC)
    if rms.size < 2:
        return []

    silent = rms < _silence_threshold(rms)
    min_windows = int(ceil(min_sec / SILENCE_WINDOW_SEC))

    runs: list[tuple[int, int]] = []
    start = None
    for k, is_silent in enumerate(silent):
        if is_silent and start is None:
            start = k
        elif not is_silent and start is not None:
            if k - start >= min_windows:
                runs.append((start * window, k * window))
            start = None
    return runs


def _compress_silences(
    audio: np.ndarray,
    sr: int,
    silences: list[tuple[int, int]],
    keep_sec: float = COMPRESSED_SILENCE_SEC,
) -> np.ndarray:
    """Shorten each silence run to ``keep_sec`` (half kept from each side)."""
    half = int(sr * keep_sec / 2)
    pieces = []
    cursor = 0
    for start, end in silences:
        pieces.append(audio[cursor : start + half])
        cursor = max(start + half, end - half)
    pieces.append(audio[cursor:])
    return np.concatenate(pieces)


def _trim_trailing_silence(
    audio: np.ndarray,
    sr: int,
    max_keep_sec: float = 0.5,
) -> np.ndarray:
    """
    Trim trailing silence from a generated chunk, keeping up to
    ``max_keep_sec`` of natural tail padding.

    The fast engine can pad the end of a chunk with silence tokens (the
    model finishes the text but keeps decoding until EOS), so chunks often
    carry 1-20s of trailing silence. That padding is harmless audio - trim
    it instead of failing the chunk.

    Args:
        audio: Audio array (float32, normalized to [-1, 1])
        sr: Sample rate
        max_keep_sec: Silence kept after the last speech frame (natural pause)

    Returns:
        Trimmed audio array (or the original if nothing meaningful to trim).
    """
    if audio.size == 0:
        return audio

    seg_rms, window = _window_rms(audio, sr, 0.02)  # 20ms analysis windows
    if seg_rms.size < 2:
        return audio

    # Index of the last non-silent window (0 if all silent - then keep as-is).
    non_silent = np.nonzero(seg_rms >= _silence_threshold(seg_rms))[0]
    if non_silent.size == 0:
        return audio
    last_speech = int(non_silent[-1])

    keep_end = min(len(audio), (last_speech + 1) * window + int(sr * max_keep_sec))
    # Never trim below a quarter second of audio.
    if keep_end < int(sr * 0.25):
        return audio

    trimmed = audio[:keep_end]
    if len(trimmed) < len(audio):
        print(
            f"[TTS] Trimmed {len(audio) / sr - len(trimmed) / sr:.2f}s trailing silence "
            f"(kept {len(trimmed) / sr:.2f}s)",
            flush=True,
        )
    return trimmed


def _check_duration_truncation(
    audio: np.ndarray,
    sr: int,
    text: str,
    label: str,
    min_ratio: float = 0.6,
    chars_per_sec: float = 15.0,
) -> None:
    """
    Detect real premature-EOS truncation by comparing speech duration to the
    text length.

    A chunk is truncated only if the (silence-trimmed) audio is far shorter
    than the text could possibly be spoken: below ``min_ratio`` of
    ``speech_chars / chars_per_sec``. This catches early EOS that produces
    no silence tail (the model simply stops), which a trailing-silence check
    alone cannot see.

    Args:
        audio: Silence-trimmed audio array (float32)
        sr: Sample rate
        text: The chunk text that was synthesized
        label: Human-readable context for the error message
        min_ratio: Minimum acceptable fraction of the expected duration
        chars_per_sec: Assumed narration rate (15 chars/s ~ 150 wpm)

    Raises:
        RuntimeError: If the audio is too short for the text.
    """
    speech_chars = sum(c.isalnum() for c in text)
    if speech_chars == 0:
        return

    expected_sec = speech_chars / chars_per_sec
    actual_sec = len(audio) / sr
    if actual_sec < min_ratio * expected_sec:
        raise RuntimeError(
            f"Audio truncation detected for {label}. "
            f"Audio is {actual_sec:.2f}s but {speech_chars} characters need "
            f"~{expected_sec:.2f}s of speech (minimum "
            f"{min_ratio * expected_sec:.2f}s). "
            f"This indicates premature EOS token generation."
        )


def _crossfade_audio(
    audio1: np.ndarray, audio2: np.ndarray, sr: int, fade_ms: int = 30
) -> np.ndarray:
    """Concatenate two audio arrays with crossfade."""
    fade_samples = int(sr * fade_ms / 1000)

    if len(audio1) < fade_samples or len(audio2) < fade_samples:
        return np.concatenate([audio1, audio2])

    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)

    audio1_end = audio1[-fade_samples:] * fade_out
    audio2_start = audio2[:fade_samples] * fade_in
    crossfaded = audio1_end + audio2_start

    return np.concatenate([audio1[:-fade_samples], crossfaded, audio2[fade_samples:]])


def _is_syllabic_char(c: str) -> bool:
    """CJK ideographs, kana and Hangul: roughly one syllable per character."""
    code = ord(c)
    return (
        0x3040 <= code <= 0x30FF  # Hiragana, Katakana
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0xAC00 <= code <= 0xD7AF  # Hangul syllables
    )


def _estimate_speech_seconds(text: str) -> float:
    """Rough spoken duration of text, erring on the slow side."""
    seconds = 0.0
    for c in text:
        if _is_syllabic_char(c):
            seconds += 0.25
        elif c.isdigit():
            # Digits expand when read aloud ("1,200" -> "twelve hundred").
            seconds += 0.25
        elif c.isalnum():
            seconds += 1 / 15
    return seconds


def _calculate_dynamic_max_tokens(text: str) -> int:
    """
    Calculate max_new_tokens from the expected spoken length of the text.

    The budget is a ceiling, not a target: the model normally stops at EOS.
    It matters when the model gets stuck emitting silence, so keep it to a
    few times the expected length instead of a fixed 20s+ minimum.
    """
    MIN_TOKENS = 48
    MAX_TOKENS = 768

    expected_sec = _estimate_speech_seconds(text)
    dynamic_max = ceil(expected_sec * CODEC_FRAMES_PER_SEC * MAX_TOKENS_HEADROOM) + MAX_TOKENS_PAD

    max_new = max(MIN_TOKENS, min(dynamic_max, MAX_TOKENS))

    print(
        f"[TTS] max_tokens: chars={len(text)}, expected={expected_sec:.1f}s, final={max_new}",
        flush=True,
    )

    return max_new


def _validate_chunk_audio(
    audio_data: np.ndarray, sr: int, text: str, label: str
) -> np.ndarray:
    """
    Normalize a generated chunk, trim its trailing silence, and reject chunks
    that are empty, contain no speech, or are truncated.

    Raises:
        RuntimeError: If the chunk fails a quality check.
    """
    if audio_data.size == 0:
        raise RuntimeError(f"Empty audio for {label}.")

    audio_f = audio_data.astype(np.float32)
    if np.issubdtype(audio_data.dtype, np.integer):
        audio_f = audio_f / np.iinfo(audio_data.dtype).max
    audio_rms = float(np.sqrt(np.mean(audio_f * audio_f)))
    audio_peak = float(np.max(np.abs(audio_f)))
    speech_level = _speech_level(_window_rms(audio_f, sr, SILENCE_WINDOW_SEC)[0])

    print(
        f"[TTS] {label}: RMS={audio_rms:.4f}, peak={audio_peak:.4f}, "
        f"speech level={speech_level:.4f}",
        flush=True,
    )

    # Peak/RMS alone miss chunks of low-level room tone, which can still
    # contain clicks; judge by the loudest windows instead.
    if speech_level < NO_SPEECH_LEVEL:
        raise RuntimeError(
            f"Silent audio for {label}. RMS={audio_rms:.6f}, peak={audio_peak:.6f}, "
            f"speech level={speech_level:.6f}."
        )

    # The fast engine may pad the chunk end with silence tokens. Trim
    # the tail (keeping a natural pause) and only fail the chunk if the
    # remaining speech is too short for the text (real premature EOS).
    audio_f = _trim_trailing_silence(audio_f, sr)
    _check_duration_truncation(audio_f, sr, text, label)
    return audio_f


def _synthesize_chunk(
    text: str,
    generate: Callable[[str, int], tuple[Any, int]],
    label: str,
) -> tuple[np.ndarray, int]:
    """
    Generate one chunk, regenerating it when the output has no speech, is
    truncated, or contains long dead-air pauses.

    If every attempt has dead air, the attempt with the least of it is used
    with its pauses shortened, rather than dropping the line.

    Args:
        text: Chunk text.
        generate: Callable (text, max_new_tokens) -> (wavs, sample_rate).
        label: Human-readable context for logs and errors.

    Raises:
        RuntimeError: If no attempt produced usable speech.
    """
    max_new_tokens = _calculate_dynamic_max_tokens(text)
    error_context = f"{label}, Text length: {len(text)} chars"

    best: tuple[float, np.ndarray, list[tuple[int, int]]] | None = None
    last_error: RuntimeError | None = None
    sr = 0
    for attempt in range(1, CHUNK_ATTEMPTS + 1):
        with timeout_handler(TTS_TIMEOUT_SECONDS, error_context):
            wavs, chunk_sr = generate(text, max_new_tokens)
        sr = int(chunk_sr)

        try:
            audio = _validate_chunk_audio(wavs[0], sr, text, label)
        except RuntimeError as e:
            last_error = e
            print(f"[TTS] {e} (attempt {attempt}/{CHUNK_ATTEMPTS})", flush=True)
            continue

        silences = _find_long_silences(audio, sr)
        if not silences:
            return audio, sr

        silent_sec = sum(end - start for start, end in silences) / sr
        print(
            f"[TTS] {label}: {silent_sec:.1f}s of dead air in {len(silences)} "
            f"pause(s) (attempt {attempt}/{CHUNK_ATTEMPTS})",
            flush=True,
        )
        if best is None or silent_sec < best[0]:
            best = (silent_sec, audio, silences)

    if best is not None:
        silent_sec, audio, silences = best
        print(
            f"[TTS] {label}: every attempt had dead air; shortening "
            f"{len(silences)} pause(s) totalling {silent_sec:.1f}s",
            flush=True,
        )
        return _compress_silences(audio, sr, silences), sr

    raise last_error or RuntimeError(f"No audio generated for {label}.")


def _synthesize_text(
    text: str,
    generate: Callable[[str, int], tuple[Any, int]],
    label: str,
) -> tuple[list[np.ndarray], int]:
    """Split text into chunks, synthesize each, and crossfade them together."""
    chunks = _split_text_into_chunks(text)

    if len(chunks) > 1:
        print(f"[TTS] Splitting text into {len(chunks)} chunks for {label}", flush=True)

    all_audio: list[np.ndarray] = []
    sr = 0
    for i, chunk in enumerate(chunks):
        audio, sr = _synthesize_chunk(chunk, generate, f"{label}, chunk {i + 1}/{len(chunks)}")
        all_audio.append(audio)

    merged = all_audio[0]
    for audio in all_audio[1:]:
        merged = _crossfade_audio(merged, audio, sr)
    if len(all_audio) > 1:
        print(f"[TTS] Merged {len(all_audio)} chunks into single audio", flush=True)

    return [merged], sr


def generate_dialogue_audio(
    dialogue: Dialogue,
    speaker_profile: SpeakerProfile,
    params: dict[str, Any],
    output_path: str | Path,
) -> str:
    """
    Generate audio for a single dialogue line using Qwen3-TTS.

    Args:
        dialogue: Dialogue instance with speaker name and text.
        speaker_profile: SpeakerProfile containing speaker voice mappings.
        params: TTS parameters dict with keys:
            - model_name: str (e.g., "Qwen3-TTS-12Hz-1.7B-Base")
            - temperature: float
            - top_k: int
            - top_p: float
            - repetition_penalty: float
            - max_new_tokens: int
            - subtalker_temperature: float
            - subtalker_top_k: int
            - subtalker_top_p: float
            - language: str (e.g., "en")
            - instruct: str | None (optional instruction)
        output_path: Path where audio file will be saved.

    Returns:
        Path to the generated audio file.

    Raises:
        ValueError: If speaker not found in profile or voice type is invalid.
        RuntimeError: If TTS generation fails or device issues occur.
    """
    # Find speaker in profile
    speaker = None
    for s in speaker_profile.speakers:
        if s.name.lower() == dialogue.speaker.lower():
            speaker = s
            break

    if speaker is None:
        raise ValueError(
            f"Speaker '{dialogue.speaker}' not found in profile. "
            f"Available: {', '.join(s.name for s in speaker_profile.speakers)}"
        )

    # Validate voice type
    if speaker.type not in ("preset", "saved"):
        raise ValueError(
            f"Invalid voice type: {speaker.type}. Must be 'preset' or 'saved'."
        )

    model_name = _resolve_voice_model(
        speaker.type, speaker.voice_id, params.get("model_name", "1.7B-CustomVoice")
    )

    try:
        from audio.model_loader import get_model

        model = get_model(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}")

    try:
        if speaker.type == "preset":
            wavs, sr = _generate_preset_voice(
                model, dialogue.text, speaker.voice_id, params
            )
        else:
            wavs, sr = _generate_saved_voice(
                model, dialogue.text, speaker.voice_id, params
            )
    except Exception as e:
        raise RuntimeError(f"TTS generation failed for speaker '{speaker.name}': {e}")

    # Save audio to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        sf.write(str(output_path), wavs[0], sr)
    except Exception as e:
        raise RuntimeError(f"Failed to save audio to {output_path}: {e}")

    return str(output_path)


def _resolve_voice_model(voice_type: str, voice_id: str, base_model_name: str) -> str:
    """Model that renders a voice: saved voices use the model they were cloned with."""
    if voice_type != "saved":
        return base_model_name
    voice_meta_path = SAVED_VOICES_DIR / voice_id / "metadata.json"
    if voice_meta_path.exists():
        with open(voice_meta_path) as f:
            return json.load(f).get("model", "1.7B-Base")
    return base_model_name.replace("CustomVoice", "Base")


def synthesize_speech(
    text: str, voice_type: str, voice_id: str, params: dict[str, Any]
) -> tuple[np.ndarray, int]:
    """
    Synthesize text with a preset or saved voice.

    Uses the same chunking, retry, and dead-air handling as podcast clips.

    Args:
        text: Text to synthesize (any length; split into chunks internally).
        voice_type: "preset" or "saved".
        voice_id: Preset speaker name or saved voice directory name.
        params: TTS parameters (see generate_dialogue_audio).

    Returns:
        Tuple of (float32 mono audio, sample_rate).
    """
    if voice_type not in ("preset", "saved"):
        raise ValueError(f"Invalid voice type: {voice_type}. Must be 'preset' or 'saved'.")

    from audio.model_loader import get_model

    model_name = _resolve_voice_model(
        voice_type, voice_id, params.get("model_name", "1.7B-CustomVoice")
    )
    model = get_model(model_name)
    if voice_type == "preset":
        wavs, sr = _generate_preset_voice(model, text, voice_id, params)
    else:
        wavs, sr = _generate_saved_voice(model, text, voice_id, params)
    return wavs[0], sr


def _sampling_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """Sampling arguments shared by every generate call."""
    defaults = MODEL_SAMPLING_DEFAULTS
    return {
        "temperature": params.get("temperature", defaults["temperature"]),
        "top_k": int(params.get("top_k", defaults["top_k"])),
        "top_p": params.get("top_p", defaults["top_p"]),
        "repetition_penalty": params.get(
            "repetition_penalty", defaults["repetition_penalty"]
        ),
        "subtalker_temperature": params.get(
            "subtalker_temperature", defaults["subtalker_temperature"]
        ),
        "subtalker_top_k": int(params.get("subtalker_top_k", defaults["subtalker_top_k"])),
        "subtalker_top_p": params.get("subtalker_top_p", defaults["subtalker_top_p"]),
    }


def _generate_preset_voice(
    model: Any, text: str, speaker: str, params: dict[str, Any]
) -> tuple[Any, int]:
    """
    Generate audio using a preset voice.

    Args:
        model: Qwen3-TTS model instance.
        text: Text to synthesize.
        speaker: Preset speaker name.
        params: TTS parameters.

    Returns:
        Tuple of (wavs, sample_rate).
    """
    lang = _normalize_language(params.get("language", "english"))
    print(f"[LANG] TTS normalized: {lang}", flush=True)

    sampling = _sampling_kwargs(params)

    def generate(chunk: str, max_new_tokens: int) -> tuple[Any, int]:
        return model.generate_custom_voice(
            text=chunk,
            speaker=speaker,
            language=lang,
            instruct=params.get("instruct"),
            non_streaming_mode=True,
            max_new_tokens=max_new_tokens,
            **sampling,
        )

    return _synthesize_text(text, generate, f"preset voice {speaker}")


def _generate_saved_voice(
    model: Any, text: str, voice_id: str, params: dict[str, Any]
) -> tuple[Any, int]:
    """
    Generate audio using a saved voice clone.

    Args:
        model: Qwen3-TTS model instance.
        text: Text to synthesize.
        voice_id: Saved voice identifier.
        params: TTS parameters.

    Returns:
        Tuple of (wavs, sample_rate).

    Raises:
        FileNotFoundError: If saved voice not found.
    """
    voice_dir = SAVED_VOICES_DIR / voice_id
    prompt_path = voice_dir / "prompt.pkl"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Saved voice not found: {voice_id}")

    with open(prompt_path, "rb") as f:
        raw_prompt = pickle.load(f)

    voice_clone_prompt = _prepare_voice_clone_prompt(raw_prompt, model)
    print(
        f"[TTS] Prepared voice clone prompt for {voice_id} (dtype/device normalized)",
        flush=True,
    )

    lang = _normalize_language(params.get("language", "english"))
    print(f"[LANG] TTS normalized (voice clone): {lang}", flush=True)

    sampling = _sampling_kwargs(params)

    def generate(chunk: str, max_new_tokens: int) -> tuple[Any, int]:
        return model.generate_voice_clone(
            text=chunk,
            language=lang,
            voice_clone_prompt=voice_clone_prompt,
            non_streaming_mode=True,
            max_new_tokens=max_new_tokens,
            **sampling,
        )

    return _synthesize_text(text, generate, f"voice {voice_id}")


def generate_transcript_audio(
    transcript: Transcript,
    speaker_profile: SpeakerProfile,
    params: dict[str, Any],
    output_dir: str | Path,
) -> list[str]:
    """
    Generate audio for all dialogues in a transcript.

    Args:
        transcript: Transcript with dialogue list.
        speaker_profile: SpeakerProfile with voice mappings.
        params: TTS parameters.
        output_dir: Directory to save audio files.

    Returns:
        List of paths to generated audio files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = []
    for i, dialogue in enumerate(transcript.dialogues):
        # Sanitize speaker name for safe filename
        safe_speaker = (
            "".join(c for c in dialogue.speaker if c.isalnum() or c in "_-")[:30]
            or "unknown"
        )
        output_file = output_dir / f"dialogue_{i:03d}_{safe_speaker}.wav"
        try:
            path = generate_dialogue_audio(
                dialogue, speaker_profile, params, output_file
            )
            audio_paths.append(path)
        except Exception as e:
            print(f"Warning: Failed to generate audio for dialogue {i}: {e}")

    return audio_paths


if __name__ == "__main__":
    # Test with mock data
    from podcast_models import Dialogue, Speaker, SpeakerProfile, Transcript

    print("=== Audio Generator Test ===\n")

    # Create test speaker profile
    speakers = [
        Speaker(name="Alice", voice_id="male_1", role="Host", type="preset"),
        Speaker(name="Bob", voice_id="female_1", role="Guest", type="preset"),
    ]
    profile = SpeakerProfile(speakers=speakers)

    # Create test transcript
    dialogues = [
        Dialogue(speaker="Alice", text="Welcome to the podcast."),
        Dialogue(speaker="Bob", text="Thanks for having me."),
        Dialogue(speaker="Alice", text="Let's dive into the topic."),
    ]
    transcript = Transcript(dialogues=dialogues)

    # TTS parameters
    tts_params = {
        "model_name": "Qwen3-TTS-12Hz-1.7B-Base",
        "temperature": 0.3,
        "top_k": 50,
        "top_p": 0.85,
        "repetition_penalty": 1.0,
        "max_new_tokens": 1024,
        "subtalker_temperature": 0.3,
        "subtalker_top_k": 50,
        "subtalker_top_p": 0.85,
        "language": "en",
        "instruct": None,
    }

    # Test 1: Single dialogue generation
    print("Test 1: Generate single dialogue audio")
    try:
        output_file = Path("test_output") / "test_dialogue.wav"
        path = generate_dialogue_audio(dialogues[0], profile, tts_params, output_file)
        print(f"✓ Generated: {path}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 2: Missing speaker error handling
    print("\nTest 2: Missing speaker error handling")
    try:
        bad_dialogue = Dialogue(speaker="Unknown", text="This should fail.")
        path = generate_dialogue_audio(bad_dialogue, profile, tts_params, "test.wav")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")

    # Test 3: Invalid voice type error handling
    print("\nTest 3: Invalid voice type error handling")
    try:
        bad_speaker = Speaker(
            name="Charlie", voice_id="v1", role="Guest", type="invalid"
        )
        bad_profile = SpeakerProfile(speakers=[bad_speaker])
        dialogue = Dialogue(speaker="Charlie", text="Test")
        path = generate_dialogue_audio(dialogue, bad_profile, tts_params, "test.wav")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")

    print("\n=== Tests completed ===")
