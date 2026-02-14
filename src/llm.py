"""
LLM Module — Dual Backend Support
- Backend 1: llama-cpp-python (direct GGUF loading, CPU or GPU)
- Backend 2: Ollama (manages CUDA automatically, easiest setup)
- Supports streaming token generation on both
"""

import json
from pathlib import Path
from collections.abc import Generator

from config import (
    LLM_MODEL_PATH,
    LLM_N_CTX,
    LLM_N_GPU_LAYERS,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_REPEAT_PENALTY,
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SYSTEM_PROMPT,
    format_prompt,
)


class LLMGenerator:
    """Manages LLM inference with automatic backend selection."""

    def __init__(
        self,
        backend: str | None = None,
        model_path: str | Path | None = None,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
    ):
        self.backend = backend or LLM_BACKEND
        self.model_path = str(model_path or LLM_MODEL_PATH)
        self.n_ctx = n_ctx or LLM_N_CTX
        self.n_gpu_layers = n_gpu_layers or LLM_N_GPU_LAYERS
        self._model = None  # llama-cpp model
        self._loaded = False

    def load(self) -> dict:
        """
        Load the model. Call once at startup.

        Returns:
            Dict with backend info and status.
        """
        if self._loaded:
            return {"status": "already_loaded", "backend": self.backend}

        if self.backend == "ollama":
            return self._load_ollama()
        else:
            return self._load_llama_cpp()

    def _load_ollama(self) -> dict:
        """Verify Ollama is running and model is available."""
        import urllib.request

        try:
            req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())

            available = [m["name"] for m in data.get("models", [])]
            model_found = any(OLLAMA_MODEL in m for m in available)

            self._loaded = True
            return {
                "status": "ready",
                "backend": "ollama",
                "model": OLLAMA_MODEL,
                "model_found": model_found,
                "available_models": available,
                "note": (
                    "Model ready." if model_found
                    else f"Model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}"
                ),
            }
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                f"Start it with 'ollama serve' or install from https://ollama.ai\n"
                f"Error: {e}"
            )

    def _load_llama_cpp(self) -> dict:
        """Load GGUF model via llama-cpp-python."""
        from llama_cpp import Llama

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"The model should auto-download on app startup. Check app initialization."
            )

        # Try GPU first, fall back to CPU
        gpu_layers = self.n_gpu_layers
        gpu_used = gpu_layers != 0

        try:
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=gpu_layers,
                verbose=False,
            )
        except Exception:
            if gpu_layers != 0:
                gpu_used = False
                self._model = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=0,
                    verbose=False,
                )

        self._loaded = True
        return {
            "status": "ready",
            "backend": "llama_cpp",
            "model": Path(self.model_path).name,
            "gpu_offload": gpu_used,
            "n_ctx": self.n_ctx,
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Generation: Non-Streaming ──

    def generate(
        self,
        context: str,
        question: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a complete answer (non-streaming)."""
        if not self.is_loaded:
            self.load()

        if self.backend == "ollama":
            return self._generate_ollama(context, question, max_tokens, temperature)
        else:
            return self._generate_llama_cpp(context, question, max_tokens, temperature)

    def _generate_llama_cpp(
        self, context: str, question: str,
        max_tokens: int | None, temperature: float | None,
    ) -> str:
        prompt = format_prompt(context, question)
        response = self._model(
            prompt,
            max_tokens=max_tokens or LLM_MAX_TOKENS,
            temperature=temperature or LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            repeat_penalty=LLM_REPEAT_PENALTY,
            stop=["<|end|>", "<|user|>", "<|system|>"],
            echo=False,
        )
        return response["choices"][0]["text"].strip()

    def _generate_ollama(
        self, context: str, question: str,
        max_tokens: int | None, temperature: float | None,
    ) -> str:
        import urllib.request

        user_msg = f"Context:\n{context}\n\nQuestion: {question}"
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {
                "temperature": temperature or LLM_TEMPERATURE,
                "top_p": LLM_TOP_P,
                "repeat_penalty": LLM_REPEAT_PENALTY,
                "num_predict": max_tokens or LLM_MAX_TOKENS,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())

        return data["message"]["content"].strip()

    # ── Generation: Streaming ──

    def generate_stream(
        self,
        context: str,
        question: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Generator[str, None, None]:
        """Stream tokens one at a time (for real-time UI)."""
        if not self.is_loaded:
            self.load()

        if self.backend == "ollama":
            yield from self._stream_ollama(context, question, max_tokens, temperature)
        else:
            yield from self._stream_llama_cpp(context, question, max_tokens, temperature)

    def _stream_llama_cpp(
        self, context: str, question: str,
        max_tokens: int | None, temperature: float | None,
    ) -> Generator[str, None, None]:
        prompt = format_prompt(context, question)
        stream = self._model(
            prompt,
            max_tokens=max_tokens or LLM_MAX_TOKENS,
            temperature=temperature or LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            repeat_penalty=LLM_REPEAT_PENALTY,
            stop=["<|end|>", "<|user|>", "<|system|>"],
            echo=False,
            stream=True,
        )
        for chunk in stream:
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    def _stream_ollama(
        self, context: str, question: str,
        max_tokens: int | None, temperature: float | None,
    ) -> Generator[str, None, None]:
        import urllib.request

        user_msg = f"Context:\n{context}\n\nQuestion: {question}"
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "stream": True,
            "options": {
                "temperature": temperature or LLM_TEMPERATURE,
                "top_p": LLM_TOP_P,
                "repeat_penalty": LLM_REPEAT_PENALTY,
                "num_predict": max_tokens or LLM_MAX_TOKENS,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue  # Skip malformed lines
        except Exception as e:
            yield f"Error during generation: {str(e)}"

    def unload(self) -> None:
        """Free the model from memory/VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
