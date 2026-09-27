"""Pluggable LLM providers for generation and automatic evaluation.

Providers (config `llm.provider`):
  anthropic          Claude via the official SDK (key from ANTHROPIC_API_KEY in .env / env)
  openai_compatible  any OpenAI-compatible endpoint: OpenAI, Ollama, LM Studio, vLLM...
  huggingface_local  a local transformers chat model (downloaded on first use)
  none               no LLM; callers fall back to extractive answers

Every provider exposes `.available`, `.name` and `generate(system, user, ...)`. Responses are
cached on disk (keyed by provider, model and prompt) so reruns are reproducible and free.
"""
import hashlib
import json
import os
from pathlib import Path


class LLMUnavailable(RuntimeError):
    pass


class _Cache:
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.data = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec["value"]

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}) + "\n")


class BaseProvider:
    name = "base"
    model = ""

    def __init__(self, log, cache_path=None):
        self.log = log
        self.available = False
        self.reason = ""
        self.cache = _Cache(cache_path)
        self.calls = 0

    def _key(self, system, user, schema, max_tokens):
        return hashlib.sha256(json.dumps([self.name, self.model, system, user, schema, max_tokens],
                                         sort_keys=True).encode()).hexdigest()

    def generate(self, system, user, json_schema=None, max_tokens=None, use_cache=True):
        """Return {"text", "json", "model", "cached"}. Raises LLMUnavailable on failure."""
        if not self.available:
            raise LLMUnavailable(self.reason or f"{self.name} unavailable")
        key = self._key(system, user, json_schema, max_tokens)
        if use_cache and (hit := self.cache.get(key)) is not None:
            return {**hit, "cached": True}
        self.calls += 1
        out = self._generate(system, user, json_schema, max_tokens)
        if json_schema is not None and out.get("json") is None:
            out["json"] = _parse_json(out["text"])
        self.cache.put(key, out)
        return {**out, "cached": False}

    def _generate(self, system, user, json_schema, max_tokens):  # pragma: no cover - interface
        raise NotImplementedError


def _parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise LLMUnavailable("model did not return JSON")
    return json.loads(text[start:end + 1])


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, cfg, log, cache_path=None):
        super().__init__(log, cache_path)
        self.model = cfg.get("model", "claude-opus-5")
        self.effort = cfg.get("effort", "low")
        self.max_tokens = cfg.get("max_tokens", 2000)
        self.use_fallbacks = cfg.get("use_server_fallbacks", True)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            self.reason = "ANTHROPIC_API_KEY is not set (add it to .env)"
            return
        try:
            import anthropic
        except ImportError:
            self.reason = "the 'anthropic' package is not installed"
            return
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=4)
        self.available = True

    def _generate(self, system, user, json_schema, max_tokens):
        output_config = {"effort": self.effort}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        params = {"model": self.model, "max_tokens": max_tokens or self.max_tokens, "system": system,
                  "messages": [{"role": "user", "content": user}], "output_config": output_config}
        a = self.anthropic
        try:
            if self.use_fallbacks:
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params)
            else:
                resp = self.client.messages.create(**params)
        except a.BadRequestError as e:
            raise LLMUnavailable(f"bad request: {e.message}") from e
        except a.AuthenticationError as e:
            self.available, self.reason = False, "invalid ANTHROPIC_API_KEY"
            raise LLMUnavailable(self.reason) from e
        except (a.RateLimitError, a.APIConnectionError, a.APIStatusError) as e:
            raise LLMUnavailable(f"API error after retries: {e}") from e
        if resp.stop_reason == "refusal":
            raise LLMUnavailable("request declined by the model")
        if resp.stop_reason == "max_tokens":
            raise LLMUnavailable("response truncated at max_tokens")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return {"text": text, "json": json.loads(text) if json_schema is not None else None, "model": resp.model}


class OpenAICompatibleProvider(BaseProvider):
    name = "openai_compatible"

    def __init__(self, cfg, log, cache_path=None):
        super().__init__(log, cache_path)
        self.model = os.environ.get(cfg.get("model_env", "OPENAI_MODEL"), "")
        base_url = os.environ.get(cfg.get("base_url_env", "OPENAI_BASE_URL")) or None
        api_key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY")) or ("local" if base_url else None)
        self.max_tokens = cfg.get("max_tokens", 1500)
        if not self.model:
            self.reason = "OPENAI_MODEL is not set (add it to .env)"
            return
        if not api_key:
            self.reason = "OPENAI_API_KEY (or OPENAI_BASE_URL for a local server) is not set"
            return
        try:
            from openai import OpenAI
        except ImportError:
            self.reason = "the 'openai' package is not installed"
            return
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.available = True

    def _generate(self, system, user, json_schema, max_tokens):
        if json_schema is not None:
            user = f"{user}\n\nRespond with JSON only, matching this schema:\n{json.dumps(json_schema)}"
        try:
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=max_tokens or self.max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        except Exception as e:  # network/auth/model errors from any compatible server
            raise LLMUnavailable(f"openai-compatible call failed: {e}") from e
        return {"text": resp.choices[0].message.content or "", "json": None, "model": self.model}


class HuggingFaceLocalProvider(BaseProvider):
    name = "huggingface_local"

    def __init__(self, cfg, log, cache_path=None):
        super().__init__(log, cache_path)
        self.model = cfg.get("model", "Qwen/Qwen2.5-0.5B-Instruct")
        self.max_new_tokens = cfg.get("max_new_tokens", 600)
        self._pipe = None
        try:
            import transformers  # noqa: F401
            self.available = True
        except ImportError:
            self.reason = "the 'transformers' package is not installed"

    def _load(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            self.log.info("Loading local LLM %s (%s)", self.model, "cuda" if device == 0 else "cpu")
            try:
                self._pipe = pipeline("text-generation", model=self.model, device=device)
            except Exception as e:
                self.available, self.reason = False, f"could not load {self.model}: {e}"
                raise LLMUnavailable(self.reason) from e
        return self._pipe

    def _generate(self, system, user, json_schema, max_tokens):
        if json_schema is not None:
            user = f"{user}\n\nRespond with JSON only, matching this schema:\n{json.dumps(json_schema)}"
        pipe = self._load()
        out = pipe([{"role": "system", "content": system}, {"role": "user", "content": user}],
                   max_new_tokens=max_tokens or self.max_new_tokens, do_sample=False)
        text = out[0]["generated_text"][-1]["content"]
        return {"text": text, "json": None, "model": self.model}


class NoneProvider(BaseProvider):
    name = "none"

    def __init__(self, cfg=None, log=None, cache_path=None):
        super().__init__(log, None)
        self.reason = "LLM disabled (provider: none)"


PROVIDERS = {"anthropic": AnthropicProvider, "openai_compatible": OpenAICompatibleProvider,
             "huggingface_local": HuggingFaceLocalProvider, "none": NoneProvider}


def get_llm(llm_cfg, log, cache_path=None, provider=None):
    """Build the configured provider, falling back to `fallback_provider` if unavailable."""
    names = [provider or llm_cfg.get("provider", "none"), llm_cfg.get("fallback_provider", "none")]
    last = None
    for name in dict.fromkeys(names):
        cls = PROVIDERS.get(name)
        if cls is None:
            log.error("Unknown LLM provider '%s'", name)
            continue
        p = cls(llm_cfg.get(name, {}), log, cache_path) if name != "none" else NoneProvider()
        if p.available:
            return p
        log.warning("LLM provider '%s' unavailable: %s", name, p.reason)
        last = p
    return last or NoneProvider()
