"""Model loading utilities for Qwen3-TTS - isolated from Gradio UI.

Engine: **faster-qwen3-tts** (CUDA-graph inference, 6-10x faster than the
reference qwen-tts loop). A thin adapter (:class:`FasterQwen3TTSAdapter`)
exposes the studio's historical call signatures on top of the fast engine,
so no call site in the UI or audio pipeline needs to change.
"""

import os
import gc
import functools
import threading
import warnings
from pathlib import Path
from collections import OrderedDict

import torch

# Minimum required qwen-tts version for critical tokenizer bugfixes
# (padding bugs in 12Hz tokenizer decode: commits 5f8581d0, 6cafe558)
QWEN_TTS_MIN_VERSION = "0.1.1"

# Model registry: studio name -> HuggingFace repo ID.
# When a repo is present in the local HF hub cache, the loader resolves it
# to the cached snapshot directory so loading is 100% local (see
# _find_cached_snapshot / _resolve_model_path).
MODEL_PATHS = {
    "1.7B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "0.6B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "1.7B-Base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B-Base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B-VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

# Optional: point the studio at bare local model directories instead of HF
# repo IDs. When set, "1.7B-CustomVoice" resolves to
# <QWEN_TTS_MODEL_DIR>/Qwen3-TTS-12Hz-1.7B-CustomVoice (upstream layout)
# and the directory must exist.
MODEL_DIR_ENV = "QWEN_TTS_MODEL_DIR"

# LRU cache of loaded models. Bump (e.g. 2) to keep two models resident
# at the cost of ~2x VRAM.
MAX_LOADED_MODELS = int(os.environ.get("QWEN_TTS_MAX_LOADED_MODELS", "1"))

# The studio previously patched model.model.generate to enforce this
# minimum (anti-truncation). The fast engine's CUDA-graph decode loop does
# not go through that method, so the adapter injects it explicitly per call.
#
# Keep this SMALL: the fast engine implements min_new_tokens by suppressing
# the EOS token until the floor is reached, so every token beyond the natural
# end of the text is generated as SILENCE. A floor of 60 (5s at 12Hz) forced
# multi-second silent tails on short chunks, which the podcast pipeline's
# trailing-silence check then flagged as truncation and retried to failure.
# 12 tokens (1s) still guards against pathologically short outputs without
# padding normal chunks. Raise via QWEN_TTS_MIN_NEW_TOKENS if you see
# premature-EOS truncation on longer chunks.
MIN_NEW_TOKENS_DEFAULT = int(os.environ.get("QWEN_TTS_MIN_NEW_TOKENS", "12"))

# Static talker cache size for the CUDA graphs. Must cover the largest
# max_new_tokens the studio can request (estimate_max_tokens caps at 4096).
MAX_SEQ_LEN = 4096


@functools.lru_cache(maxsize=1)
def _check_qwen_tts_version() -> None:
    """Verify qwen-tts package meets minimum version requirement.

    Raises RuntimeError if the installed version is too old (known tokenizer
    decode bugs that corrupt 12Hz audio output). Can be bypassed with
    QWEN_TTS_ALLOW_OLD=1 environment variable.

    Cached so the check only runs once per process (env-based bypass is
    locked in at first call).
    """
    if os.environ.get("QWEN_TTS_ALLOW_OLD", "").strip() == "1":
        return

    try:
        from importlib.metadata import version as pkg_version
        from packaging.version import Version

        installed = pkg_version("qwen-tts")
        if Version(installed) < Version(QWEN_TTS_MIN_VERSION):
            raise RuntimeError(
                f"qwen-tts {installed} is installed, but >={QWEN_TTS_MIN_VERSION} is required. "
                f"Older versions have known tokenizer decode bugs that corrupt 12Hz audio. "
                f"Run: pip install -U 'qwen-tts>={QWEN_TTS_MIN_VERSION}' "
                f"(set QWEN_TTS_ALLOW_OLD=1 to bypass this check)"
            )
    except ImportError:
        # packaging not available — skip version check but warn
        warnings.warn(
            "Cannot verify qwen-tts version (missing 'packaging' library). "
            f"Ensure qwen-tts >= {QWEN_TTS_MIN_VERSION} is installed.",
            RuntimeWarning,
            stacklevel=2,
        )


def _detect_device() -> str:
    """Resolve the CUDA device for the fast engine.

    Priority: QWEN_TTS_DEVICE env var > cuda:0.

    faster-qwen3-tts requires CUDA (CUDA-graph inference); any non-CUDA
    device is a hard error with actionable guidance.
    """
    override = os.environ.get("QWEN_TTS_DEVICE", "").strip()
    device = override if override else "cuda:0"

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            "faster-qwen3-tts requires a CUDA GPU (CUDA-graph inference). "
            f"Resolved device: {device!r}, torch.cuda.is_available()="
            f"{torch.cuda.is_available()}. "
            "Set QWEN_TTS_DEVICE=cuda:N to pick a GPU, and make sure a "
            "CUDA-enabled PyTorch build is installed, e.g. "
            "pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )
    return device


def _find_cached_snapshot(repo_id: str):
    """Return the local HF hub cache snapshot dir for a repo, or None.

    Reads only the local cache (``refs`` + ``snapshots``) via
    ``try_to_load_from_cache`` — never touches the network, whether or not
    ``HF_HUB_OFFLINE`` is set.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        cached_file = try_to_load_from_cache(repo_id, "config.json")
    except Exception:
        return None
    if not cached_file:
        return None
    snapshot_dir = Path(cached_file).parent
    # Require the weights to be present as well (single-file or sharded).
    has_weights = (snapshot_dir / "model.safetensors").exists() or any(
        snapshot_dir.glob("model-*.safetensors")
    ) or (snapshot_dir / "pytorch_model.bin").exists()
    return str(snapshot_dir) if has_weights else None


def _resolve_model_path(model_name: str) -> str:
    """Resolve a studio model name to a local model directory or HF repo ID.

    Resolution order:
      1. ``QWEN_TTS_MODEL_DIR`` override — bare local directories in the
         upstream layout (``<dir>/Qwen3-TTS-12Hz-<name>``).
      2. Local HuggingFace hub cache — the snapshot directory of the repo's
         ``main`` revision, when the repo is already cached. Loading from a
         local directory is fully offline: it skips every huggingface_hub
         network probe. (Important: transformers' "mistral regex" check
         calls the hub API for repo IDs even with ``HF_HUB_OFFLINE=1``,
         which breaks offline loads of Qwen3-TTS.)
      3. HuggingFace repo ID — downloads on first use (requires network).
    """
    repo_id = MODEL_PATHS.get(model_name)
    if not repo_id:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_PATHS.keys())}"
        )

    model_dir = os.environ.get(MODEL_DIR_ENV, "").strip()
    if model_dir:
        local_path = Path(model_dir) / f"Qwen3-TTS-12Hz-{model_name}"
        if not local_path.exists():
            raise ValueError(
                f"{MODEL_DIR_ENV} is set but the model directory was not found: "
                f"{local_path}"
            )
        return str(local_path)

    cached = _find_cached_snapshot(repo_id)
    if cached is not None:
        return cached
    return repo_id


class FasterQwen3TTSAdapter:
    """Adapts FasterQwen3TTS to the studio's historical call signatures.

    - Drops kwargs the fast engine does not accept: ``subtalker_*``
      everywhere (its predictor uses internal sampling defaults), and
      ``non_streaming_mode`` for custom voice / voice design.
    - Forwards ``non_streaming_mode`` to ``generate_voice_clone`` (the fast
      engine accepts it there; the studio always passes True).
    - Injects ``min_new_tokens`` (the studio's previous anti-truncation
      patch, now applied explicitly per call).
    - Defaults a missing/empty ``language`` to ``"auto"`` (qwen-tts
      semantics; the fast engine requires an explicit language string).
    """

    _DROP_ALWAYS = ("subtalker_temperature", "subtalker_top_k", "subtalker_top_p")

    def __init__(self, fast, min_new_tokens: int = MIN_NEW_TOKENS_DEFAULT):
        self._fast = fast
        self._min_new_tokens = min_new_tokens

    def _call(self, method: str, **kwargs):
        for key in self._DROP_ALWAYS:
            kwargs.pop(key, None)
        kwargs.setdefault("min_new_tokens", self._min_new_tokens)
        return getattr(self._fast, method)(**kwargs)

    @staticmethod
    def _auto_language(language):
        return language if language else "auto"

    def generate_custom_voice(self, text, speaker, language=None, instruct=None, **kwargs):
        kwargs.pop("non_streaming_mode", None)
        return self._call(
            "generate_custom_voice",
            text=text,
            speaker=speaker,
            language=self._auto_language(language),
            instruct=instruct,
            **kwargs,
        )

    def generate_voice_clone(
        self,
        text,
        language=None,
        voice_clone_prompt=None,
        ref_audio=None,
        ref_text="",
        **kwargs,
    ):
        # non_streaming_mode is accepted by the fast engine on this path;
        # keep the studio's value (it always passes True).
        #
        # ref_text bridge: upstream qwen-tts reads ref_text from each
        # VoiceClonePromptItem when a precomputed prompt is passed, but the
        # fast engine requires it as a parameter in ICL mode. The studio's
        # call sites never pass it, so pull it from the prompt item.
        if not ref_text and voice_clone_prompt is not None:
            ref_text = self._ref_text_from_prompt(voice_clone_prompt)
        return self._call(
            "generate_voice_clone",
            text=text,
            language=self._auto_language(language),
            voice_clone_prompt=voice_clone_prompt,
            ref_audio=ref_audio,
            ref_text=ref_text,
            **kwargs,
        )

    @staticmethod
    def _ref_text_from_prompt(voice_clone_prompt) -> str:
        """Extract ref_text from a precomputed ICL prompt (upstream semantics)."""
        items = (
            voice_clone_prompt
            if isinstance(voice_clone_prompt, list)
            else [voice_clone_prompt]
        )
        for it in items:
            if it is None:
                continue
            icl = getattr(it, "icl_mode", None)
            xvec = getattr(it, "x_vector_only_mode", None)
            is_icl = bool(icl) if icl is not None else (not bool(xvec))
            if not is_icl:
                continue
            rt = getattr(it, "ref_text", None)
            if rt and str(rt).strip():
                return str(rt)
        return ""

    def generate_voice_design(self, **kwargs):
        kwargs.pop("non_streaming_mode", None)
        kwargs["language"] = self._auto_language(kwargs.get("language"))
        return self._call("generate_voice_design", **kwargs)

    def create_voice_clone_prompt(self, *args, **kwargs):
        """Proxy to the underlying qwen-tts model (the wrapper has no such method)."""
        return self._fast.model.create_voice_clone_prompt(*args, **kwargs)

    @property
    def model(self):
        """Underlying qwen-tts Qwen3TTSModel (for internals)."""
        return self._fast.model

    @property
    def sample_rate(self):
        return self._fast.sample_rate

    @property
    def speech_tokenizer(self):
        return self._fast.speech_tokenizer

    def __getattr__(self, name):
        # Passthrough: prefer the fast wrapper, then the underlying qwen-tts
        # model (get_supported_speakers, get_supported_languages, ...).
        fast = self.__dict__.get("_fast")
        if fast is not None and hasattr(fast, name):
            return getattr(fast, name)
        base = self.__dict__.get("_base")
        if base is not None and hasattr(base, name):
            return getattr(base, name)
        raise AttributeError(
            f"{type(self).__name__} has no attribute {name!r} "
            f"(nor do the wrapped engines)"
        )


loaded_models: OrderedDict = OrderedDict()
_model_lock = threading.Lock()


def _gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unload_model(model_name: str) -> None:
    if model_name in loaded_models:
        del loaded_models[model_name]
        _gpu_cleanup()
        print(f"Unloaded {model_name}")


def get_model(model_name: str):
    """
    Get or load a Qwen3-TTS model via the faster-qwen3-tts engine.

    Args:
        model_name: Name of the model to load (e.g., "1.7B-CustomVoice")

    Returns:
        A FasterQwen3TTSAdapter wrapping the fast engine.

    Raises:
        ValueError: If the model is unknown or a local dir override is missing
        RuntimeError: If faster-qwen3-tts is missing, CUDA is unavailable,
            or model loading fails
    """
    with _model_lock:
        if model_name in loaded_models:
            loaded_models.move_to_end(model_name)
            return loaded_models[model_name]

        while len(loaded_models) >= MAX_LOADED_MODELS:
            old_name, _ = next(iter(loaded_models.items()))
            _unload_model(old_name)

        _check_qwen_tts_version()

        try:
            from faster_qwen3_tts import FasterQwen3TTS
        except ImportError as e:
            raise RuntimeError(
                "faster-qwen3-tts is not installed. "
                "Run: pip install faster-qwen3-tts"
            ) from e

        model_path = _resolve_model_path(model_name)
        device = _detect_device()

        print(f"Loading {model_name} ({model_path}) on {device} via faster-qwen3-tts...")

        try:
            m = FasterQwen3TTS.from_pretrained(
                model_path,
                device=device,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                max_seq_len=MAX_SEQ_LEN,
            )
        except Exception as e:
            _gpu_cleanup()
            raise RuntimeError(f"Failed to load {model_name}: {e}") from e

        adapter = FasterQwen3TTSAdapter(m, MIN_NEW_TOKENS_DEFAULT)
        # Keep a handle to the underlying qwen-tts model for the passthrough
        # fallback in __getattr__ (avoids re-traversing on every miss).
        adapter._base = m.model

        loaded_models[model_name] = adapter
        print(
            f"{model_name} loaded on {device} (bf16, CUDA graphs, "
            f"min_new_tokens={MIN_NEW_TOKENS_DEFAULT})!"
        )
        return adapter


if __name__ == "__main__":
    # Test model loading
    print("Testing model loader...")
    try:
        model = get_model("1.7B-CustomVoice")
        print("Model loaded successfully!")
        print(f"Supported speakers: {model.get_supported_speakers()}")
    except Exception as e:
        print(f"Error: {e}")