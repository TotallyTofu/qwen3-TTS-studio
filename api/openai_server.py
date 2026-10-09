"""OpenAI-compatible text-to-speech server for studio voices.

Serves ``POST /v1/audio/speech`` with the request shape of OpenAI's Speech API
(https://developers.openai.com/api/docs/guides/text-to-speech), so any app or
chat client that speaks that API can use studio voices. Each studio voice is
linked to an OpenAI voice name (alloy, echo, sage, ...) in the "OpenAI API"
tab; a request for that name is rendered with the linked voice.

The server runs on its own port in a background thread next to the Gradio UI.
Links and the API key are re-read from disk on every request, so saving them
in the UI takes effect without a restart.
"""

from __future__ import annotations

import copy
import io
import json
import os
import secrets
import shutil
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict

from audio.generator import synthesize_speech
from podcast.presets import MODEL_SAMPLING_DEFAULTS
from storage.persona import load_persona

SETTINGS_FILE = Path("openai_api_settings.json")
SAVED_VOICES_DIR = Path("saved_voices")

# Built-in voice names of OpenAI's Speech API.
OPENAI_VOICES = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
)
OPENAI_MODELS = ("tts-1", "tts-1-hd", "gpt-4o-mini-tts")
MAX_LINKS = 4
# OpenAI's limit on the request's ``input`` field.
MAX_INPUT_CHARS = 4096
MIN_SPEED, MAX_SPEED = 0.25, 4.0

CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}
# OpenAI's "pcm" format: raw 24kHz 16-bit signed little-endian mono samples.
PCM_SAMPLE_RATE = 24000
# Opus only encodes at these rates.
OPUS_SAMPLE_RATES = (8000, 12000, 16000, 24000, 48000)

DEFAULT_SETTINGS: dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 7880,
    "api_key": "",
    "autostart": False,
    # [{"openai_voice": "alloy", "voice": "saved:<id>" | "preset:<id>", "language": "auto"}]
    "links": [],
}


# ---------------------------------------------------------------------------
# Settings and voice links
# ---------------------------------------------------------------------------


def load_api_settings() -> dict[str, Any]:
    """Load API settings, falling back to defaults for missing keys."""
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as f:
                settings.update(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[API] Ignoring unreadable {SETTINGS_FILE}: {e}")
    return settings


def save_api_settings(settings: dict[str, Any]) -> None:
    """Save API settings atomically (the server may be reading them)."""
    tmp_path = SETTINGS_FILE.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, SETTINGS_FILE)


def parse_voice_value(value: str) -> tuple[str, str]:
    """Split a "saved:<id>" / "preset:<id>" voice value into (type, id)."""
    voice_type, _, voice_id = (value or "").partition(":")
    if voice_type not in ("preset", "saved") or not voice_id:
        raise ValueError(f"Invalid studio voice: {value!r}")
    return voice_type, voice_id


def voice_display_name(value: str) -> str:
    """Human-readable name of a studio voice: its persona name if it has one."""
    try:
        voice_type, voice_id = parse_voice_value(value)
    except ValueError:
        return value
    try:
        persona = load_persona(voice_id, voice_type)
    except Exception:
        persona = None
    if persona is not None:
        return persona.character_name
    if voice_type == "saved":
        meta_path = SAVED_VOICES_DIR / voice_id / "metadata.json"
        try:
            with open(meta_path) as f:
                return json.load(f).get("name") or voice_id
        except (OSError, json.JSONDecodeError):
            return voice_id
    return voice_id.replace("_", " ").title()


def find_link(settings: dict[str, Any], openai_voice: str) -> dict[str, Any] | None:
    """Return the link for an OpenAI voice name (case-insensitive), if any."""
    name = (openai_voice or "").strip().lower()
    for link in settings.get("links", []):
        if link.get("openai_voice", "").lower() == name and link.get("voice"):
            return link
    return None


def client_base_url(host: str, port: int) -> str:
    """Base URL an app on this PC should use to reach the server."""
    if host in ("", "0.0.0.0", "::"):
        host = "127.0.0.1"
    elif ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{port}/v1"


# ---------------------------------------------------------------------------
# Synthesis and encoding
# ---------------------------------------------------------------------------

# Renders one request at a time, so concurrent requests for voices on
# different models don't thrash model loading between their chunks.
_synth_lock = threading.Lock()


def synthesize_link(
    link: dict[str, Any], text: str, instructions: str | None = None
) -> tuple[np.ndarray, int]:
    """Render text with a linked studio voice. Returns (float32 audio, sample_rate)."""
    voice_type, voice_id = parse_voice_value(link["voice"])
    params = {
        **MODEL_SAMPLING_DEFAULTS,
        "model_name": "1.7B-CustomVoice",
        "language": link.get("language") or "auto",
        # Only preset voices (CustomVoice model) take style instructions.
        "instruct": (instructions or None) if voice_type == "preset" else None,
    }
    with _synth_lock:
        return synthesize_speech(text, voice_type, voice_id, params)


def _ffmpeg_exe() -> str | None:
    """ffmpeg bundled with imageio-ffmpeg (a moviepy dependency), else from PATH."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _run_ffmpeg(audio: np.ndarray, sr: int, output_args: list[str]) -> bytes:
    """Pipe mono float32 audio through ffmpeg and return its stdout."""
    exe = _ffmpeg_exe()
    if exe is None:
        raise RuntimeError("ffmpeg was not found (install moviepy or add ffmpeg to PATH).")
    cmd = [
        exe, "-hide_banner", "-loglevel", "error",
        "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
        *output_args, "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        input=audio.astype("<f4").tobytes(),
        capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed: {detail}")
    return proc.stdout


def change_speed(audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Time-stretch audio without changing pitch (ffmpeg atempo)."""
    if abs(speed - 1.0) < 1e-3:
        return audio
    if _ffmpeg_exe() is None:
        print(f"[API] ffmpeg not found; ignoring speed={speed}")
        return audio
    # One atempo stage covers 0.5-2.0; chain stages for the rest of 0.25-4.0.
    factors = []
    remaining = speed
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    factors.append(remaining)
    atempo = ",".join(f"atempo={f:.6f}" for f in factors)
    out = _run_ffmpeg(audio, sr, ["-filter:a", atempo, "-f", "f32le"])
    return np.frombuffer(out, dtype="<f4").copy()


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr or audio.size == 0:
        return audio
    n_out = max(1, round(len(audio) * target_sr / sr))
    positions = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


def encode_audio(audio: np.ndarray, sr: int, fmt: str) -> bytes:
    """Encode mono float audio in one of OpenAI's response formats."""
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    if fmt == "pcm":
        audio = _resample(audio, sr, PCM_SAMPLE_RATE)
        return (audio * 32767).astype("<i2").tobytes()
    if fmt == "aac":
        return _run_ffmpeg(audio, sr, ["-c:a", "aac", "-b:a", "128k", "-f", "adts"])

    buf = io.BytesIO()
    if fmt == "wav":
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    elif fmt == "flac":
        sf.write(buf, audio, sr, format="FLAC")
    elif fmt == "mp3":
        sf.write(buf, audio, sr, format="MP3")
    elif fmt == "opus":
        if sr not in OPUS_SAMPLE_RATES:
            audio, sr = _resample(audio, sr, 48000), 48000
        sf.write(buf, audio, sr, format="OGG", subtype="OPUS")
    else:
        raise ValueError(f"Unsupported response_format: {fmt}")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# HTTP app
# ---------------------------------------------------------------------------


class OpenAIError(Exception):
    """An error returned in OpenAI's JSON error shape."""

    def __init__(
        self,
        status_code: int,
        message: str,
        param: str | None = None,
        error_type: str = "invalid_request_error",
        code: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.param = param
        self.error_type = error_type
        self.code = code


def _error_response(exc: OpenAIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.error_type,
                "param": exc.param,
                "code": exc.code,
            }
        },
    )


class SpeechRequest(BaseModel):
    """Body of POST /v1/audio/speech (unknown fields are ignored)."""

    model_config = ConfigDict(extra="ignore")

    model: str = "tts-1"
    input: str
    voice: str | dict[str, Any]
    instructions: str | None = None
    response_format: str = "mp3"
    speed: float = 1.0
    stream_format: str | None = None


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    """Check the bearer token when an API key is configured."""
    api_key = load_api_settings().get("api_key") or ""
    if not api_key:
        return
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(
        token.strip().encode(), api_key.encode()
    ):
        raise OpenAIError(
            401,
            "Incorrect API key provided. Use the key set in the studio's OpenAI API tab.",
            error_type="invalid_request_error",
            code="invalid_api_key",
        )


router = APIRouter()


@router.post("/audio/speech", dependencies=[Depends(require_api_key)])
def create_speech(req: SpeechRequest) -> Response:
    text = req.input.strip()
    if not text:
        raise OpenAIError(400, "'input' must not be empty.", "input")
    if len(req.input) > MAX_INPUT_CHARS:
        raise OpenAIError(
            400,
            f"'input' is {len(req.input)} characters; the maximum is {MAX_INPUT_CHARS}.",
            "input",
        )

    fmt = req.response_format.strip().lower()
    if fmt not in CONTENT_TYPES:
        raise OpenAIError(
            400,
            f"Unsupported response_format '{req.response_format}'. "
            f"Use one of: {', '.join(CONTENT_TYPES)}.",
            "response_format",
        )
    if not MIN_SPEED <= req.speed <= MAX_SPEED:
        raise OpenAIError(
            400, f"'speed' must be between {MIN_SPEED} and {MAX_SPEED}.", "speed"
        )
    if (req.stream_format or "audio").lower() != "audio":
        raise OpenAIError(
            400, "Only stream_format 'audio' is supported.", "stream_format"
        )

    voice = req.voice.get("id", "") if isinstance(req.voice, dict) else req.voice
    settings = load_api_settings()
    link = find_link(settings, voice)
    if link is None:
        linked = [l["openai_voice"] for l in settings.get("links", []) if l.get("voice")]
        raise OpenAIError(
            400,
            f"Voice '{voice}' is not linked to a studio voice. "
            f"Linked voices: {', '.join(linked) or 'none'} "
            "(link voices in the studio's OpenAI API tab).",
            "voice",
        )

    start = time.time()
    try:
        audio, sr = synthesize_link(link, text, req.instructions)
        audio = change_speed(audio, sr, req.speed)
        data = encode_audio(audio, sr, fmt)
    except Exception as e:
        traceback.print_exc()
        raise OpenAIError(500, f"Speech generation failed: {e}", error_type="server_error")

    print(
        f"[API] {voice} -> {voice_display_name(link['voice'])} | {len(text)} chars | "
        f"{fmt} | {len(audio) / sr:.1f}s audio in {time.time() - start:.1f}s",
        flush=True,
    )
    return Response(content=data, media_type=CONTENT_TYPES[fmt])


@router.get("/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "qwen3-tts-studio"}
            for m in OPENAI_MODELS
        ],
    }


# The two listings below follow the convention of OpenAI-compatible TTS
# servers, which apps such as Open WebUI query to fill their voice pickers.
@router.get("/audio/models")
def list_audio_models() -> dict[str, Any]:
    return {"models": [{"id": m} for m in OPENAI_MODELS]}


@router.get("/audio/voices")
def list_audio_voices() -> dict[str, Any]:
    links = [l for l in load_api_settings().get("links", []) if l.get("voice")]
    return {
        "voices": [
            {
                "id": l["openai_voice"],
                "name": f"{l['openai_voice']} ({voice_display_name(l['voice'])})",
            }
            for l in links
        ]
    }


def create_app() -> FastAPI:
    app = FastAPI(title="Qwen3-TTS Studio - OpenAI-compatible TTS API")

    @app.exception_handler(OpenAIError)
    async def _openai_error(_request: Request, exc: OpenAIError) -> JSONResponse:
        return _error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        problems = []
        for err in exc.errors():
            field = ".".join(str(p) for p in err.get("loc", ())[1:])
            problems.append(f"{field}: {err.get('msg')}" if field else str(err.get("msg")))
        return _error_response(OpenAIError(400, "; ".join(problems) or "Invalid request."))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # Apps differ on whether their base URL includes /v1; serve both.
    app.include_router(router, prefix="/v1")
    app.include_router(router)
    return app


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


class APIServer:
    """Runs the API app with uvicorn on a background thread."""

    def __init__(self) -> None:
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.host = ""
        self.port = 0

    @property
    def running(self) -> bool:
        return (
            self._server is not None
            and self._server.started
            and self._thread is not None
            and self._thread.is_alive()
        )

    @property
    def base_url(self) -> str:
        return client_base_url(self.host, self.port)

    def start(self, host: str, port: int) -> None:
        """Start serving; raises RuntimeError if the server cannot bind."""
        with self._lock:
            if self.running:
                raise RuntimeError(f"The API server is already running at {self.base_url}.")
            config = uvicorn.Config(
                create_app(),
                host=host,
                port=port,
                log_level="warning",
                timeout_graceful_shutdown=3,
            )
            server = uvicorn.Server(config)
            thread = threading.Thread(target=server.run, name="openai-tts-api", daemon=True)
            thread.start()

            deadline = time.time() + 10
            while not server.started and thread.is_alive() and time.time() < deadline:
                time.sleep(0.05)
            if not server.started:
                server.should_exit = True
                raise RuntimeError(
                    f"Could not start the API server on {host}:{port}. "
                    "Is the port already in use? See the console for details."
                )

            self._server, self._thread = server, thread
            self.host, self.port = host, port
            print(f"[API] OpenAI-compatible TTS API running at {self.base_url}", flush=True)

    def stop(self) -> None:
        with self._lock:
            if self._server is None:
                return
            self._server.should_exit = True
            if self._thread is not None:
                self._thread.join(timeout=10)
            self._server = None
            self._thread = None
            print("[API] OpenAI-compatible TTS API stopped", flush=True)


api_server = APIServer()
