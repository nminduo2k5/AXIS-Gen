"""
AXIS-Gen prototype -- generators.py

Three request-synthesis strategies compared in the experiment:

  1. RandomGenerator        - uniform random sampling over attribute domains.
  2. PairwiseGenerator      - greedy 2-way combinatorial (covering-array) baseline,
                              representative of classical XACML test generation
                              (Bertolino et al. X-CREATE; Cotroneo et al.).
  3. LLMGuidedGenerator     - policy-aware synthesis guided by a natural-language
                              rendering of the policy. It is built around an
                              LLMClient interface so a real Claude call
                              (AnthropicLLMClient) is a drop-in replacement for
                              the deterministic HeuristicSurrogateLLMClient used
                              to run this offline prototype without network
                              access / API credentials.

IMPORTANT (research-integrity note): the numbers produced by this prototype
when configured with HeuristicSurrogateLLMClient are NOT "real LLM" numbers.
The surrogate deterministically implements the *prompting strategy* described
in the accompanying research document (rule-activation targeting, boundary
value probing, single-predicate negation for MC/DC-style coverage, and
cross-rule interaction probes) so that the evaluation harness, metrics, and
comparison methodology can be validated end-to-end without an API key. Swap
in AnthropicLLMClient (or any LLMClient) to obtain results for the actual
LLM-guided approach.
"""

from __future__ import annotations

import abc
import itertools
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

from policy_model import (
    AttrRef, ICS_ATTRIBUTE_DOMAINS, Policy, Predicate, Request,
    all_attr_refs, domain_values,
)


# --------------------------------------------------------------------------
# 1. Random baseline
# --------------------------------------------------------------------------

def random_request(rng: random.Random, domains=ICS_ATTRIBUTE_DOMAINS) -> Request:
    values: Dict[Tuple[str, str], object] = {}
    for ref in all_attr_refs(domains):
        values[(ref.category, ref.attribute)] = rng.choice(domain_values(ref, domains))
    return Request(values)


def random_generator(n: int, seed: int) -> List[Request]:
    rng = random.Random(seed)
    return [random_request(rng) for _ in range(n)]


# --------------------------------------------------------------------------
# 2. Pairwise (2-way) combinatorial baseline -- greedy candidate-based
# --------------------------------------------------------------------------

def _all_pairs(domains=ICS_ATTRIBUTE_DOMAINS):
    refs = all_attr_refs(domains)
    pairs = []
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            for vi in domain_values(refs[i], domains):
                for vj in domain_values(refs[j], domains):
                    pairs.append(((refs[i], vi), (refs[j], vj)))
    return refs, pairs


def _request_covers(values: Dict[Tuple[str, str], object], pair) -> bool:
    (r1, v1), (r2, v2) = pair
    return values[(r1.category, r1.attribute)] == v1 and values[(r2.category, r2.attribute)] == v2


def pairwise_generator(seed: int, candidate_pool: int = 400, max_tests: int = 120) -> List[Request]:
    rng = random.Random(seed)
    refs, pairs = _all_pairs()
    uncovered = set(range(len(pairs)))
    suite: List[Request] = []

    def gen_candidate():
        return {(r.category, r.attribute): rng.choice(domain_values(r)) for r in refs}

    while uncovered and len(suite) < max_tests:
        best_candidate, best_gain, best_covered_idx = None, -1, set()
        for _ in range(candidate_pool):
            cand = gen_candidate()
            covered_idx = {idx for idx in uncovered if _request_covers(cand, pairs[idx])}
            if len(covered_idx) > best_gain:
                best_gain, best_candidate, best_covered_idx = len(covered_idx), cand, covered_idx
        if best_gain <= 0:
            break
        suite.append(Request(dict(best_candidate)))
        uncovered -= best_covered_idx

    return suite


# --------------------------------------------------------------------------
# 3. LLM-guided synthesis
# --------------------------------------------------------------------------

class LLMClient(abc.ABC):
    """Pluggable text-generation backend used by LLMGuidedGenerator."""

    @abc.abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, seed: Optional[int] = None) -> str:
        """Return raw model output (expected to be a JSON array of request dicts).

        `seed` is forwarded to providers that support seeded sampling
        (Gemini, OpenAI) to make runs as reproducible as the API allows.
        """


# Shared sampling settings so all real backends are compared fairly.
LLM_TEMPERATURE = 0.7
LLM_MAX_OUTPUT_TOKENS = 16384   # 40 requests x ~130 tokens = ~5k; big headroom
FALLBACK_EMPTY = "[]"


class AnthropicLLMClient(LLMClient):
    """Real backend: calls the Claude API via the `anthropic` Python SDK.

    Requires `pip install anthropic` and an `ANTHROPIC_API_KEY` environment
    variable.
    """

    def __init__(self, model: str = "claude-sonnet-5-5"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy: no hard dependency for other backends
            headers = {}
            workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
            if workspace_id:
                headers["anthropic-workspace-id"] = workspace_id
            self._client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                default_headers=headers or None,
                max_retries=6,      # SDK-level backoff for 429/5xx/overloaded
                timeout=300.0,
            )
        return self._client

    def generate(self, system_prompt: str, user_prompt: str, seed: Optional[int] = None) -> str:
        import anthropic
        client = self._get_client()
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=LLM_MAX_OUTPUT_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                anthropic.NotFoundError, anthropic.BadRequestError) as e:
            # configuration errors (bad key / model name / workspace): retrying
            # or silently scoring 0% would hide the real problem -> fail loudly.
            raise RuntimeError(f"[anthropic] configuration error: {e}") from e
        except Exception as e:
            print(f"[anthropic] call failed after retries: {str(e)[:200]}")
            return FALLBACK_EMPTY
        if getattr(resp, "stop_reason", None) == "max_tokens":
            print("[anthropic] WARNING — response truncated (stop_reason=max_tokens).")
        return "".join(block.text for block in resp.content if block.type == "text")


class GeminiLLMClient(LLMClient):
    """Robust Gemini backend for multi-seed experiment pipelines.

    Design goals:
      1. NO health-check request at startup — model is accepted optimistically,
         validated on first real call only. Saves quota and avoids false negatives.
      2. 503 (overloaded) → retry same model with exponential backoff + jitter.
      3. 429 (quota exceeded) → long backoff (60–300s); after MAX_QUOTA_RETRIES
         give up and return a FALLBACK_EMPTY string so the seed is skipped
         gracefully rather than crashing the whole experiment.
      4. 404/403 (model permanently unavailable) → try next model in FALLBACK_MODELS
         exactly ONCE, then graceful failure — never crash the pipeline.
      5. Cached client — one Client object per instance, not per call.

    Requires:  pip install -U google-genai
    Env var:   GEMINI_API_KEY
    """

    # Conservative list — experiment uses ONE model for all seeds.
    # Second entry is only used if the first is permanently unavailable.
    FALLBACK_MODELS = [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
    ]

    # Graceful-failure sentinel returned when quota is exhausted.
    # The harness treats an empty JSON array as zero requests generated.
    FALLBACK_EMPTY = "[]"

    # Delays for 503 transient errors
    MAX_RETRIES_503  = 6
    BASE_DELAY_503   = 2.0   # seconds
    MAX_DELAY_503    = 60.0

    # Delays for 429 errors — per-minute rate limits clear within ~1 min, so a few
    # long waits rescue a seed; a daily-quota exhaustion fails after the retries.
    MAX_QUOTA_RETRIES = 3
    BASE_DELAY_429    = 20.0
    MAX_DELAY_429     = 90.0

    def __init__(self, model: str = "gemini-3.5-flash"):
        self._client = None
        # Accept model optimistically — no health-check request.
        self.model = model
        # gemini-3.x / 2.5+ models reason internally by default, and that
        # reasoning is billed against max_output_tokens unless a separate
        # thinking budget is set. None = not yet probed; True/False = whether
        # thinking_budget=0 is accepted by the current model (cached after
        # the first successful call so we don't retry a rejected param
        # every single request).
        self._thinking_budget_zero_supported = None
        print(f"[gemini] Configured model: {self.model} (will validate on first call)")

    # ── Client cache ──────────────────────────────────────────────────────
    def _get_client(self):
        if self._client is None:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. "
                    "Set it in .env.gemini and run again."
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    # ── Error classification ──────────────────────────────────────────────
    # Prefer the structured HTTP status (google.genai.errors.APIError.code /
    # .status) over substring matching on the message: a bare "500" or
    # "503" substring also matches unrelated numbers such as "8500 tokens".
    @staticmethod
    def _code(error: Exception):
        code = getattr(error, "code", None)
        return code if isinstance(code, int) else None

    @classmethod
    def _is_auth(cls, error: Exception) -> bool:
        """Invalid / missing API key — retrying or falling back cannot help."""
        msg = str(error).upper()
        return (cls._code(error) == 401
                or "API_KEY_INVALID" in msg or "API KEY NOT VALID" in msg
                or "API KEY EXPIRED" in msg)

    @classmethod
    def _is_quota(cls, error: Exception) -> bool:
        """429 — daily/minute quota exhausted."""
        code = cls._code(error)
        if code is not None:
            return code == 429
        msg = str(error).upper()
        return "RESOURCE_EXHAUSTED" in msg or "429" in msg

    @classmethod
    def _is_transient(cls, error: Exception) -> bool:
        """500/502/503/504 — server-side temporary failure."""
        code = cls._code(error)
        if code is not None:
            return code in (500, 502, 503, 504)
        msg = str(error).upper()
        return "UNAVAILABLE" in msg or "OVERLOADED" in msg

    @classmethod
    def _is_permanent(cls, error: Exception) -> bool:
        """404/403 — model gone or project denied."""
        code = cls._code(error)
        if code is not None:
            return code in (403, 404)
        msg = str(error).upper()
        return "NOT_FOUND" in msg or "PERMISSION_DENIED" in msg

    # Total token budget for a call. gemini-2.5+/3.x models reason
    # internally and that reasoning is billed against max_output_tokens
    # unless a separate (smaller) thinking budget carves out room for the
    # actual visible answer — without it, long-reasoning prompts (this one
    # asks the model to work through 6 rules x negation/boundary/pair
    # probes) can burn the whole budget on internal thought, leaving only a
    # truncated mid-thought fragment as the visible response.
    MAX_OUTPUT_TOKENS = LLM_MAX_OUTPUT_TOKENS
    THINKING_BUDGET    = 0  # 0 = disable extended thinking where supported

    def _build_config(self, system_prompt: str, disable_thinking: bool,
                      seed: Optional[int] = None):
        from google.genai import types

        kwargs = dict(
            system_instruction=system_prompt,
            max_output_tokens=self.MAX_OUTPUT_TOKENS,
            temperature=LLM_TEMPERATURE,
            # Forces syntactically valid JSON output (no fences / prose).
            response_mime_type="application/json",
        )
        if seed is not None:
            kwargs["seed"] = int(seed)
        if disable_thinking:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.THINKING_BUDGET,
            )
        return types.GenerateContentConfig(**kwargs)

    # ── Core call with full retry logic ───────────────────────────────────
    def _call(self, model: str, system_prompt: str, user_prompt: str,
              seed: Optional[int] = None) -> str:
        """Send one generation request; handle 503 retries internally."""
        import time
        import warnings

        client = self._get_client()

        for attempt in range(self.MAX_RETRIES_503 + 1):
            try:
                # Try with thinking disabled first (cheapest, most budget
                # left for the actual JSON answer); if this model/SDK
                # combination rejects thinking_config, fall back to the
                # plain config and remember that for future calls.
                try_thinking_off = self._thinking_budget_zero_supported is not False
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        resp = client.models.generate_content(
                            model=model,
                            contents=user_prompt,
                            config=self._build_config(system_prompt, disable_thinking=try_thinking_off, seed=seed),
                        )
                        if try_thinking_off:
                            self._thinking_budget_zero_supported = True
                    except Exception as cfg_err:
                        # Only treat this as "thinking_config rejected" (and
                        # retry without it) when the error actually looks
                        # like a bad-request/schema complaint mentioning the
                        # field — anything else (quota, transient, auth) must
                        # propagate untouched so the outer handler classifies
                        # it correctly instead of burning an extra call.
                        msg = str(cfg_err)
                        looks_like_config_rejection = (
                            try_thinking_off
                            and self._is_quota(cfg_err) is False
                            and self._is_transient(cfg_err) is False
                            and "thinking" in msg.lower()
                        )
                        if not looks_like_config_rejection:
                            raise
                        print(f"[gemini] thinking_budget=0 not accepted by {model} "
                              f"({msg[:100]}); retrying without it.")
                        self._thinking_budget_zero_supported = False
                        resp = client.models.generate_content(
                            model=model,
                            contents=user_prompt,
                            config=self._build_config(system_prompt, disable_thinking=False, seed=seed),
                        )

                finish_reason = None
                if getattr(resp, "candidates", None):
                    finish_reason = getattr(resp.candidates[0], "finish_reason", None)
                if finish_reason is not None and str(finish_reason).upper().find("MAX_TOKENS") != -1:
                    print(f"[gemini] WARNING — response truncated (finish_reason={finish_reason}). "
                          "The model ran out of its token budget before finishing; "
                          "output is likely incomplete/invalid JSON.")

                text = (resp.text or "").strip()
                if not text:
                    raise RuntimeError("[gemini] Empty response from model.")
                print(f"[gemini] Raw response received ({len(text)} chars)")
                return text

            except Exception as e:
                # 429 quota — bubble up to generate() for long backoff
                if self._is_quota(e):
                    raise

                # 503 transient — retry same model
                if self._is_transient(e):
                    if attempt >= self.MAX_RETRIES_503:
                        raise
                    delay = min(
                        self.BASE_DELAY_503 * (2 ** attempt),
                        self.MAX_DELAY_503,
                    ) + random.uniform(0, 2)
                    print(f"[gemini] 503 transient (attempt {attempt+1}/{self.MAX_RETRIES_503}): "
                          f"retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue

                # Anything else (404, 403, auth, bad request) — propagate
                raise

        raise RuntimeError("[gemini] Exhausted 503 retries.")

    # ── Public generate() ─────────────────────────────────────────────────
    def generate(self, system_prompt: str, user_prompt: str, seed: Optional[int] = None) -> str:
        """Generate with graceful handling of quota and permanent failures.

        Returns FALLBACK_EMPTY ("[]") instead of crashing when:
          - 429 quota is exhausted after MAX_QUOTA_RETRIES long waits.
          - Model is permanently unavailable (404/403) and all fallbacks fail.
        This allows the experiment to continue with remaining seeds/methods.
        """
        import time

        models_to_try = [self.model] + [
            m for m in self.FALLBACK_MODELS if m != self.model
        ]

        permanent_errors: list = []
        for model in models_to_try:
            # ── 429 quota retry loop ──────────────────────────────────────
            for quota_attempt in range(self.MAX_QUOTA_RETRIES + 1):
                try:
                    result = self._call(model, system_prompt, user_prompt, seed=seed)
                    # On success, lock in this model for future calls
                    if model != self.model:
                        print(f"[gemini] Switched to fallback model: {model}")
                        self.model = model
                    return result

                except Exception as e:
                    if self._is_auth(e):
                        raise RuntimeError(
                            f"[gemini] Invalid/missing API key: {str(e)[:160]}") from e
                    if self._is_quota(e):
                        if quota_attempt >= self.MAX_QUOTA_RETRIES:
                            print(
                                f"[gemini] Quota exhausted on {model} after "
                                f"{self.MAX_QUOTA_RETRIES} waits. "
                                "Returning empty result for this seed — "
                                "experiment continues."
                            )
                            return self.FALLBACK_EMPTY
                        delay = min(
                            self.BASE_DELAY_429 * (2 ** quota_attempt),
                            self.MAX_DELAY_429,
                        ) + random.uniform(0, 5)
                        print(
                            f"[gemini] 429 quota exceeded "
                            f"(attempt {quota_attempt+1}/{self.MAX_QUOTA_RETRIES}): "
                            f"waiting {delay:.0f}s before retry..."
                        )
                        time.sleep(delay)
                        continue  # retry same model after wait

                    if self._is_permanent(e):
                        print(f"[gemini] {model} permanently unavailable: {str(e)[:120]}")
                        permanent_errors.append(e)
                        break  # try next model in outer loop

                    # 503 exhausted or unknown — try next model
                    print(f"[gemini] {model} failed: {str(e)[:120]}")
                    break

        # Every model failed with 403/404: this is an account/config problem
        # (project denied, model not enabled), not a transient one. Continuing
        # would just write meaningless 0% results, so stop here.
        if len(permanent_errors) == len(models_to_try):
            raise RuntimeError(
                "[gemini] All models rejected with 403/404 (API key / Google project "
                f"has no access): {str(permanent_errors[0])[:200]}")

        # All models failed — graceful failure, do not crash experiment
        print(
            "[gemini] All models failed. Returning empty result for this seed — "
            "experiment continues."
        )
        return self.FALLBACK_EMPTY


class OpenAILLMClient(LLMClient):
    """Real backend: calls OpenAI API via the `openai` Python SDK.

    Requires `pip install openai` and an `OPENAI_API_KEY` environment
    variable (https://platform.openai.com/api-keys).

    Notes:
      * Uses `max_completion_tokens` (accepted by all current chat models;
        the legacy `max_tokens` is rejected by newer/reasoning models).
      * Reasoning models (o-series, gpt-5*) reject a custom `temperature`;
        on that specific 400 error the call is retried once without it.
      * Retries 429/5xx/timeouts via the SDK (`max_retries`); auth /
        bad-model errors fail loudly instead of silently scoring 0%.
    """

    # Overridable by OpenAI-compatible providers (see GroqLLMClient).
    LABEL = "openai"
    API_KEY_ENV = "OPENAI_API_KEY"
    ENV_FILE_HINT = ".env.openai"
    BASE_URL: Optional[str] = None
    MAX_TOKENS = LLM_MAX_OUTPUT_TOKENS

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None
        # Optional sampling params some models reject (e.g. temperature on
        # reasoning models). A rejected one is dropped once and remembered.
        self._dropped_params: set = set()

    def _extra_params(self) -> Dict[str, object]:
        """Provider/model specific optional params (overridden by subclasses)."""
        return {}

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy import
            api_key = os.environ.get(self.API_KEY_ENV)
            if not api_key:
                raise RuntimeError(f"{self.API_KEY_ENV} is not set. Set it in "
                                   f"{self.ENV_FILE_HINT} and run again.")
            self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL,
                                  max_retries=3, timeout=300.0)
        return self._client

    def _create(self, system_prompt: str, user_prompt: str, seed: Optional[int]):
        kwargs = dict(
            model=self.model,
            max_completion_tokens=self.MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        optional = {"temperature": LLM_TEMPERATURE, **self._extra_params()}
        if seed is not None:
            optional["seed"] = int(seed)
        kwargs.update({k: v for k, v in optional.items() if k not in self._dropped_params})
        return self._get_client().chat.completions.create(**kwargs)

    def generate(self, system_prompt: str, user_prompt: str, seed: Optional[int] = None) -> str:
        import openai
        try:
            while True:
                try:
                    resp = self._create(system_prompt, user_prompt, seed)
                    break
                except openai.BadRequestError as e:
                    # If the error names an optional param we sent, drop it and retry.
                    msg = str(e).lower()
                    bad = [p for p in ("temperature", "seed", "reasoning_effort")
                           if p in msg and p not in self._dropped_params]
                    if not bad:
                        raise
                    print(f"[{self.LABEL}] {self.model} rejects {bad}; retrying without.")
                    self._dropped_params.update(bad)
        except (openai.AuthenticationError, openai.PermissionDeniedError,
                openai.NotFoundError, openai.BadRequestError) as e:
            raise RuntimeError(f"[{self.LABEL}] configuration error: {e}") from e
        except openai.RateLimitError as e:
            # 429 has two meanings: a transient rate limit (already retried by
            # the SDK) vs. "insufficient_quota" = no billing credit, which no
            # retry can fix -> fail loudly instead of scoring every seed 0%.
            if "insufficient_quota" in str(e) or "no credits" in str(e).lower():
                raise RuntimeError(f"[{self.LABEL}] out of credits: {e}") from e
            if "request too large" in str(e).lower():
                # Per-minute token cap smaller than one answer: never succeeds.
                raise RuntimeError(f"[{self.LABEL}] request exceeds this model's "
                                   f"per-minute token limit (cannot be fixed by retrying): {e}") from e
            print(f"[{self.LABEL}] rate-limited after retries: {str(e)[:500]}")
            return FALLBACK_EMPTY
        except Exception as e:
            print(f"[{self.LABEL}] call failed after retries: {str(e)[:200]}")
            return FALLBACK_EMPTY

        choice = resp.choices[0]
        if choice.finish_reason == "length":
            print(f"[{self.LABEL}] WARNING — response truncated (finish_reason=length).")
        return choice.message.content or ""


class GroqLLMClient(OpenAILLMClient):
    """Groq Cloud (https://console.groq.com) via its OpenAI-compatible API.

    Requires `pip install openai` and a `GROQ_API_KEY` environment variable.
    Reuses all OpenAI-client behaviour (seed, retries, fail-fast on bad
    key/model, truncation warnings) with a different base URL.
    Output cap is 8192 tokens: enough for ~40 requests, and small enough to
    stay under the per-request token limits of Groq's free tier.
    """
    LABEL = "groq"
    API_KEY_ENV = "GROQ_API_KEY"
    ENV_FILE_HINT = ".env.groq"
    BASE_URL = "https://api.groq.com/openai/v1"
    MAX_TOKENS = 8192

    def __init__(self, model: str = "openai/gpt-oss-120b"):
        super().__init__(model=model)

    def _extra_params(self) -> Dict[str, object]:
        # gpt-oss models spend output tokens on hidden reasoning; with the
        # default effort that can exhaust the cap and truncate the JSON.
        # Low effort keeps the answer complete (fair across seeds).
        if "gpt-oss" in self.model:
            return {"reasoning_effort": "low"}
        return {}


class HeuristicSurrogateLLMClient(LLMClient):
    """Deterministic stand-in for an LLM, implementing prompting strategy
    items (1)-(3) (rule activation, single-predicate negation, boundary
    probing) from PROMPT_TEMPLATE_SYSTEM below, so the pipeline can be
    exercised offline. It receives the same (system_prompt, user_prompt)
    pair a real LLM would, but "answers" by running the strategy
    programmatically instead of via a language model.

    NOTE: deliberately does NOT implement prompting item (4) (rule-pair
    conflict requests). This is intentional: our experiment showed that
    even a well-specified natural-language instruction to "probe rule
    interactions" is easy for a purely example-driven generator to satisfy
    only partially, because finding a genuine multi-rule conflict witness
    is a constraint-satisfaction problem, not a sampling problem. See
    llm_guided_hybrid_generator() / rule_pair_conflict_probes() for the
    fix, and the research document's ablation study for the measured
    effect of adding it back via a symbolic pass instead of prompting
    alone.
    """

    def generate(self, system_prompt: str, user_prompt: str, seed: Optional[int] = None) -> str:
        # The policy object is threaded in via closure by build_llm_prompt();
        # for the surrogate we recover it from a side-channel attribute set
        # by LLMGuidedGenerator right before calling generate().
        policy: Policy = self._policy  # type: ignore[attr-defined]
        n: int = self._n  # type: ignore[attr-defined]
        rng = random.Random(self._seed)  # type: ignore[attr-defined]

        out: List[Dict[str, object]] = []

        def base_from_rule(rule) -> Dict[Tuple[str, str], object]:
            """Start from a fully-random request, then pin every predicate in
            the rule's target/condition to the value that satisfies it, so
            the rule is *activated* (rule-coverage targeting)."""
            vals = {(r.category, r.attribute): rng.choice(domain_values(r)) for r in all_attr_refs()}
            for part in (rule.target, rule.condition):
                for pred in part.leaves():
                    vals[(pred.ref.category, pred.ref.attribute)] = _satisfying_value(pred)
            return vals

        for rule in policy.rules:
            base = base_from_rule(rule)
            # (a) rule-activation request
            out.append(dict(base))

            leaves = list(rule.target.leaves()) + list(rule.condition.leaves())
            for pred in leaves:
                # (b) single-predicate negation -> MC/DC-style "flip one
                # condition, hold the rest fixed" probe
                neg = dict(base)
                neg[(pred.ref.category, pred.ref.attribute)] = _violating_value(pred, rng)
                out.append(neg)

                # (c) boundary probe for numeric thresholds (n-1, n, n+1)
                if isinstance(pred.value, int):
                    for delta in (-1, 0, 1):
                        b = dict(base)
                        b[(pred.ref.category, pred.ref.attribute)] = pred.value + delta
                        out.append(b)

        # (d) a handful of fully-random exploration requests for diversity /
        # unanticipated-interaction coverage
        for _ in range(max(0, n - len(out))):
            out.append({(r.category, r.attribute): rng.choice(domain_values(r)) for r in all_attr_refs()})

        rng.shuffle(out)
        out = out[:n]
        as_json = [{f"{c}.{a}": v for (c, a), v in req.items()} for req in out]
        return json.dumps(as_json)


def _satisfying_value(pred: Predicate):
    if pred.op == "eq":
        return pred.value
    if pred.op == "neq":
        opts = [v for v in domain_values(pred.ref) if v != pred.value]
        return opts[0] if opts else pred.value
    if pred.op in ("gte", "gt"):
        return pred.value
    if pred.op in ("lte", "lt"):
        return pred.value
    if pred.op == "in":
        return sorted(pred.value)[0]
    if pred.op == "not_in":
        opts = [v for v in domain_values(pred.ref) if v not in pred.value]
        return opts[0] if opts else pred.value
    return pred.value


def _violating_value(pred: Predicate, rng: random.Random):
    domain = domain_values(pred.ref)
    if pred.op == "eq":
        opts = [v for v in domain if v != pred.value]
    elif pred.op == "neq":
        opts = [pred.value]
    elif pred.op in ("gte", "gt"):
        thr = pred.value if pred.op == "gte" else pred.value + 1
        opts = [v for v in domain if isinstance(v, int) and v < thr] or [domain[0]]
    elif pred.op in ("lte", "lt"):
        thr = pred.value if pred.op == "lte" else pred.value - 1
        opts = [v for v in domain if isinstance(v, int) and v > thr] or [domain[-1]]
    elif pred.op == "in":
        opts = [v for v in domain if v not in pred.value] or [domain[0]]
    elif pred.op == "not_in":
        opts = sorted(pred.value)
    else:
        opts = domain
    return rng.choice(opts) if opts else rng.choice(domain)


# --------------------------------------------------------------------------
# 3b. Lightweight symbolic "rule-pair conflict witness" search.
#
# Motivation (an empirical finding from this prototype, not a hypothesis):
# generating a rule-activation request per rule individually -- as (a)/(b)/(c)
# above do -- does NOT guarantee that combining-algorithm faults (RCM
# mutants) are killed. Killing those requires a request where TWO rules'
# Target+Condition are BOTH satisfied simultaneously (a "rule-pair
# coverage" witness, cf. Xu et al.'s rule-pair coverage criterion and
# Bertolino et al.'s Z3-based strong-mutation constraint solving). Because
# our attribute domains are small and finite, we obtain such witnesses by
# exhaustive grid search rather than a full SMT solver -- the point being
# generalised in the research document to "any lightweight/CSP or SMT
# witness-finder wired in behind the LLM-guided front end".
# --------------------------------------------------------------------------

_PROBE_CACHE: Dict[int, List[Request]] = {}


def rule_pair_conflict_probes(policy: Policy) -> List[Request]:
    """Deterministic (seed-independent), so cached per policy object: the
    full 92,160-point grid scan would otherwise repeat for every seed."""
    key = id(policy)
    if key not in _PROBE_CACHE:
        _PROBE_CACHE[key] = _compute_rule_pair_conflict_probes(policy)
    return list(_PROBE_CACHE[key])


def _compute_rule_pair_conflict_probes(policy: Policy) -> List[Request]:
    from policy_model import Decision

    refs = all_attr_refs()
    domains = [domain_values(r) for r in refs]
    grid = itertools.product(*domains)

    # decisions[i] -> list of (assignment_tuple) that make rule i PERMIT/DENY
    witnesses: Dict[int, List[Tuple]] = {i: [] for i in range(len(policy.rules))}
    found_pairs = set()
    probes: List[Request] = []
    needed_pairs = {
        (i, j) for i in range(len(policy.rules)) for j in range(i + 1, len(policy.rules))
    }

    for assignment in grid:
        values = {(r.category, r.attribute): v for r, v in zip(refs, assignment)}
        req = Request(values)
        applicable = [
            i for i, rule in enumerate(policy.rules)
            if rule.evaluate(req) != Decision.NOT_APPLICABLE
        ]
        if len(applicable) >= 2:
            for i, j in itertools.combinations(applicable, 2):
                if (i, j) in needed_pairs and (i, j) not in found_pairs:
                    found_pairs.add((i, j))
                    probes.append(req)
        if found_pairs == needed_pairs:
            break

    return probes


PROMPT_TEMPLATE_SYSTEM = """You are a security test engineer generating XACML \
access-control test REQUESTS (not policies) for an industrial control system. \
You will be given (1) a natural-language rendering of every rule in an ABAC \
policy, including its Target and Condition, and (2) the finite attribute \
domains available. Your job is to output a JSON array of requests -- each a \
flat object mapping "<category>.<attribute>" to a value drawn ONLY from the \
listed domain -- that are likely to reveal faults in the policy. \
Prioritise, in order: (1) at least one request that activates each rule \
exactly as written; (2) for every atomic condition in a rule, a paired \
request that flips ONLY that one condition (all else fixed) to probe \
condition/decision coverage (MC/DC-style); (3) boundary values (n-1, n, n+1) \
for every numeric threshold; (4) for every pair of rules that COULD plausibly \
be applicable to the same request (i.e. their Targets do not obviously \
exclude one another), at least one request where BOTH rules' Target and \
Condition are satisfied simultaneously, so the rule-combining algorithm is \
actually exercised, not just each rule in isolation.

Output format rules (strict):
- Return ONLY a raw JSON array. Do not wrap it in Markdown code fences \
(no ```json, no ```). Do not add any explanation, heading, numbering, \
bullet list, or commentary before or after the array.
- The response body must start with '[' and end with ']' and contain \
nothing else.
- Each array element is a flat JSON object mapping "<category>.<attribute>" \
to a value drawn only from the listed domain. An optional "comment" field \
may describe the intent of the request.
- If no valid request can be generated, return exactly []."""


def build_user_prompt(policy: Policy, n: Optional[int] = None) -> str:
    lines = []
    if n is not None:
        lines.append(f"Generate exactly {n} requests (no more, no fewer). Use JSON "
                     "integers (not strings) for numeric attributes.\n")
    lines += ["Policy rules (combining algorithm: %s):" % policy.combining_algorithm]
    for r in policy.rules:
        lines.append(f"- {r.rule_id} [{r.effect.name}] {r.description}")
    lines.append("\nAttribute domains:")
    for cat, attrs in ICS_ATTRIBUTE_DOMAINS.items():
        for attr, dom in attrs.items():
            lines.append(f"- {cat}.{attr}: {dom}")
    return "\n".join(lines)


def _extract_json(text: str) -> Optional[object]:
    """Best-effort recovery of a JSON value from raw LLM text output.

    Real LLM backends (Gemini in particular) routinely wrap the JSON payload
    in a Markdown code fence (```json ... ```) or prepend/append prose even
    when explicitly told not to. Strict json.loads() on the raw string fails
    in both cases even though the JSON itself is well-formed once isolated,
    which previously caused every such response to be silently treated as
    "0 requests generated". This tries, in order: (1) the raw text as-is,
    (2) the contents of the first fenced code block, (3) the first balanced
    '[' ... ']' span, (4) the first balanced '{' ... '}' span.
    """
    if not text:
        return None

    # Reasoning models (Qwen3, DeepSeek-R1...) may inline <think>...</think>;
    # drop it so brackets inside the reasoning are not mistaken for the answer.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    candidates: List[str] = [text.strip()]

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())

    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start:end + 1].strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Last resort: the output was truncated mid-array (token limit). Salvage
    # every COMPLETE top-level object that precedes the cut.
    start = text.find("[")
    if start != -1:
        dec = json.JSONDecoder()
        pos, salvaged = start + 1, []
        while True:
            nxt = text.find("{", pos)
            if nxt == -1:
                break
            try:
                obj, end = dec.raw_decode(text, nxt)
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict):
                salvaged.append(obj)
            pos = end
        if salvaged:
            print(f"[json] WARNING — output truncated; salvaged {len(salvaged)} complete objects.")
            return salvaged
    return None


# Per-call diagnostics for real-LLM runs (parse failures, repairs, budget
# adherence). main_experiment.py dumps this into aggregate_stats.json so the
# paper can report LLM validity honestly.
LLM_DIAGNOSTICS: List[Dict[str, object]] = []


def _coerce_to_domain(val, dom: List[object]):
    """Map a raw LLM value onto the attribute domain. Returns (value, ok).
    Accepts harmless format drift ("3" for 3, " Onsite " for "onsite")
    without counting it as an invalid value; anything else is not ok."""
    if val in dom and not isinstance(val, bool):
        return val, True
    if dom and isinstance(dom[0], int):
        try:
            iv = int(float(str(val).strip()))
            if iv in dom:
                return iv, True
        except (ValueError, TypeError):
            pass
        return None, False
    if isinstance(val, str):
        norm = val.strip().lower().replace("-", "_").replace(" ", "_")
        for d in dom:
            if str(d).lower() == norm:
                return d, True
    return None, False


def llm_guided_generator(policy: Policy, n: int, seed: int, client: LLMClient = None) -> List[Request]:
    client = client or HeuristicSurrogateLLMClient()
    is_real = not isinstance(client, HeuristicSurrogateLLMClient)
    backend = type(client).__name__
    system_prompt = PROMPT_TEMPLATE_SYSTEM
    user_prompt = build_user_prompt(policy, n)

    # side-channel the policy/n/seed into the surrogate so its generate()
    # signature matches the real LLMClient interface exactly.
    if not is_real:
        client._policy, client._n, client._seed = policy, n, seed  # type: ignore[attr-defined]

    diag = {"backend": backend, "seed": seed, "requested": n, "returned": 0,
            "used": 0, "invalid_values": 0, "total_values": 0, "failed": False,
            "attempts": 0}

    # A real LLM occasionally answers "[]" / prose / broken JSON even though
    # the same prompt works on the next try. Retry a couple of times (new
    # sampling seed) before declaring the seed failed, so a one-off glitch
    # does not turn into a 0% data point.
    max_attempts = 3 if is_real else 1
    valid: List[Dict[str, object]] = []
    for attempt in range(max_attempts):
        diag["attempts"] = attempt + 1
        call_seed = seed if attempt == 0 else seed + 1000 * attempt
        raw = client.generate(system_prompt, user_prompt, seed=call_seed)
        if is_real:
            print(f"[{backend}] Raw response (first 300 chars): {raw[:300]}")

        parsed = _extract_json(raw)
        if parsed is None:
            print(f"[{backend}] WARNING — Could not extract valid JSON. Raw: {raw[:200]}")
            parsed = []
        elif isinstance(parsed, dict):
            parsed = [parsed]
        elif not isinstance(parsed, list):
            print(f"[{backend}] WARNING — Parsed JSON was not a list/object "
                  f"(got {type(parsed).__name__}). Raw: {raw[:200]}")
            parsed = []

        valid = [item for item in parsed if isinstance(item, dict)]
        if len(valid) != len(parsed):
            print(f"[{backend}] WARNING — dropped {len(parsed) - len(valid)} "
                  f"non-object entries from parsed response.")
        if valid:
            break
        if attempt + 1 < max_attempts:
            print(f"[{backend}] Empty/invalid answer for seed={seed} — retrying "
                  f"({attempt + 2}/{max_attempts}).")

    if is_real and not valid:
        print(f"[{backend}] FAILED — no usable output for seed={seed} after "
              f"{max_attempts} attempts. This seed is flagged failed and excluded "
              "from LLM aggregates.")
        diag["failed"] = True
        LLM_DIAGNOSTICS.append(diag)
        return []

    diag["returned"] = len(valid)
    # Enforce the equal request budget: an LLM may over-generate.
    if len(valid) > n:
        print(f"[{backend}] Truncating {len(valid)} -> {n} requests to keep the equal budget.")
        valid = valid[:n]
    elif len(valid) < n:
        print(f"[{backend}] NOTE — model returned {len(valid)} < requested {n} requests.")

    print(f"[{backend}] Parsed request count: {len(valid)}")

    requests: List[Request] = []
    rng = random.Random(seed)
    for item in valid:
        values: Dict[Tuple[str, str], object] = {}
        for ref in all_attr_refs():
            key = f"{ref.category}.{ref.attribute}"
            dom = domain_values(ref)
            val, ok = _coerce_to_domain(item.get(key), dom)
            diag["total_values"] += 1
            if not ok:
                # --- schema/domain repair step (validity-hardening) ---
                diag["invalid_values"] += 1
                val = rng.choice(dom)
            values[(ref.category, ref.attribute)] = val
        requests.append(Request(values))

    diag["used"] = len(requests)
    diag["failed"] = len(requests) == 0
    if is_real:
        LLM_DIAGNOSTICS.append(diag)
        tv = max(1, diag["total_values"])
        print(f"[{backend}] domain-repair: {diag['invalid_values']}/{diag['total_values']} "
              f"values invalid ({100.0 * diag['invalid_values'] / tv:.1f}%)")
    print(f"[llm_guided] seed={seed} generated {len(requests)} requests")
    return requests


def llm_guided_hybrid_generator(policy: Policy, n: int, seed: int, client: LLMClient = None,
                                llm_reqs: List[Request] = None) -> List[Request]:
    """LLM-guided synthesis + the symbolic rule-pair conflict-witness pass.
    This is the "AXIS-Gen" configuration proposed in the research document:
    an LLM front end for semantic/boundary/negation reasoning, backed by a
    lightweight symbolic pass that closes the specific gap the pure
    LLM-guided prototype was empirically found to miss (see
    rule_pair_conflict_probes docstring above).

    If `llm_reqs` (the plain LLM-Guided suite for the same seed) is given, it
    is REUSED instead of calling the LLM again. This makes the comparison
    LLM-Guided vs LLM-Guided+Symbolic a paired ablation: the only difference
    is the symbolic probes (which replace the last suite entries to keep the
    budget), not an independent stochastic LLM sample."""
    probes = rule_pair_conflict_probes(policy)
    remaining = max(0, n - len(probes))
    if llm_reqs is not None:
        return probes + list(llm_reqs[:remaining])
    llm_reqs = llm_guided_generator(policy, remaining, seed, client=client) if remaining else []
    return probes + llm_reqs
