@echo off
rem ============================================================
rem  Qwen3-TTS Studio launcher (faster-qwen3-tts engine)
rem  Double-click to run, or run from a terminal.
rem ============================================================
cd /d "%~dp0"

rem All Qwen3-TTS models are already in the local HF cache, so skip
rem network probes. Remove this line if you want to download a new model.
set HF_HUB_OFFLINE=1

rem Select the GPU explicitly if needed (GPU0 = cuda:0 by default):
rem set QWEN_TTS_DEVICE=cuda:1

rem Load two models at once (doubles VRAM usage):
rem set QWEN_TTS_MAX_LOADED_MODELS=2

python qwen_tts_ui.py

echo.
echo Studio exited (code %errorlevel%).
pause >nul