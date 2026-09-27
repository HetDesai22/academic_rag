"""Stage 5: LLM-assisted validation of uncertain classifications (optional, configurable).

Only questions in the configured confidence bands (default: medium) are sent, up to
llm.max_calls. The LLM must choose from the subject's taxonomy (JSON-schema constrained), and
its answer is never accepted blindly:

  LLM agrees with the classifier                          -> accept  (llm_confirmed)
  LLM disagrees, is confident, and its topic is in the
  classifier's top-3 candidates                           -> adopt   (llm_adjudicated)
  anything else (low LLM confidence, outside top-3,
  NONE_OF_THESE, API error)                               -> human review

Responses are cached on disk (keyed by model + prompt) so reruns are reproducible and free.
The API key is read from the environment variable named in config; it is never stored.
"""
import hashlib
import json
import os
from pathlib import Path

NONE_LABEL = "NONE_OF_THESE"

SYSTEM_PROMPT = """You label university exam questions with syllabus topics for trend analysis.
You are given one question, its subject, and that subject's syllabus taxonomy (topics with their
subtopics and short descriptions). Choose the topic and subtopic the question primarily examines.
Use only names that appear in the taxonomy, spelled exactly. Add secondary topics only when the
question explicitly asks about a second, different topic. If no topic in this syllabus fits the
question, answer NONE_OF_THESE. Keep the reason to one sentence."""


def _taxonomy_text(rows):
    lines, current = [], None
    for r in rows:
        if r["topic"] != current:
            current = r["topic"]
            lines.append(f"TOPIC: {current}")
        lines.append(f"  - {r['subtopic']}: {r['description']}")
    return "\n".join(lines)


def _schema(topics):
    return {
        "type": "object",
        "properties": {
            "primary_topic": {"type": "string", "enum": topics + [NONE_LABEL]},
            "subtopic": {"type": "string"},
            "secondary_topics": {"type": "array", "items": {"type": "string", "enum": topics}},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "reason": {"type": "string"},
        },
        "required": ["primary_topic", "subtopic", "secondary_topics", "confidence", "reason"],
        "additionalProperties": False,
    }


class LLMValidator:
    def __init__(self, cfg, log):
        self.cfg = cfg["topic_llm_check"]
        self.log = log
        self.enabled = bool(self.cfg.get("enabled"))
        self.calls = 0
        self.cache_path = Path(cfg["paths"]["cache_dir"]) / "llm_cache.jsonl"
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["response"]
        self.client = None
        if self.enabled:
            key_env = self.cfg.get("api_key_env", "ANTHROPIC_API_KEY")
            if not os.environ.get(key_env):
                log.warning("LLM validation enabled but %s is not set; uncertain questions go to human review", key_env)
                self.enabled = False
            else:
                import anthropic
                self.anthropic = anthropic
                self.client = anthropic.Anthropic(api_key=os.environ[key_env], max_retries=4)

    def _request(self, subject, question, rows, candidate):
        topics = list(dict.fromkeys(r["topic"] for r in rows))
        user = (f"SUBJECT: {subject}\n\nTAXONOMY:\n{_taxonomy_text(rows)}\n\n"
                f"QUESTION: {question}\n\n"
                f"An automatic classifier suggested: {candidate['primary_topic']} / {candidate['subtopic']}. "
                f"Judge independently; the suggestion may be wrong.")
        params = {
            "model": self.cfg["model"],
            "max_tokens": self.cfg.get("max_tokens", 2000),
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": self.cfg.get("effort", "low"),
                              "format": {"type": "json_schema", "schema": _schema(topics)}},
        }
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        return key, params

    def _call(self, params):
        if self.cfg.get("use_server_fallbacks", True):
            resp = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params)
        else:
            resp = self.client.messages.create(**params)
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"refused: {getattr(resp.stop_details, 'category', None)}")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("hit max_tokens before finishing the JSON answer")
        text = next(b.text for b in resp.content if b.type == "text")
        return {"answer": json.loads(text), "model": resp.model,
                "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}}

    def validate(self, subject, question, rows, candidate):
        """Return a dict with the decision; never raises for API problems."""
        key, params = self._request(subject, question, rows, candidate)
        cached = key in self.cache
        if not cached:
            if self.calls >= self.cfg.get("max_calls", 0):
                return {"status": "skipped_budget"}
            try:
                self.calls += 1
                response = self._call(params)
            except self.anthropic.BadRequestError as e:
                self.log.error("LLM bad request (not retried): %s", e.message)
                return {"status": "error", "error": str(e.message)}
            except (self.anthropic.RateLimitError, self.anthropic.APIConnectionError,
                    self.anthropic.APIStatusError) as e:
                self.log.error("LLM API error after SDK retries: %s", e)
                return {"status": "error", "error": str(e)}
            except (RuntimeError, json.JSONDecodeError, StopIteration) as e:
                self.log.error("LLM response unusable: %s", e)
                return {"status": "error", "error": str(e)}
            self.cache[key] = response
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "response": response}) + "\n")
        response = self.cache[key]
        return self._decide(response, rows, candidate, cached)

    def _decide(self, response, rows, candidate, cached):
        ans = response["answer"]
        out = {"status": "ok", "cached": cached, "llm_model": response["model"],
               "llm_topic": ans["primary_topic"], "llm_subtopic": ans["subtopic"],
               "llm_secondary": [t for t in ans["secondary_topics"] if t != ans["primary_topic"]],
               "llm_confidence": ans["confidence"], "llm_reason": ans["reason"]}
        valid_subs = {r["subtopic"] for r in rows if r["topic"] == ans["primary_topic"]}
        if ans["primary_topic"] == NONE_LABEL:
            out["decision"] = "review_off_syllabus"
        elif ans["subtopic"] not in valid_subs:
            # Topic is valid but the subtopic is not under it: keep the topic signal, but review.
            out["decision"] = "review_invalid_subtopic"
        elif ans["primary_topic"] == candidate["primary_topic"]:
            out["decision"] = "confirmed"
        elif ans["confidence"] == "high" and ans["primary_topic"] in candidate["top_topics"]:
            out["decision"] = "adjudicated"
        else:
            out["decision"] = "review_disagreement"
        return out
