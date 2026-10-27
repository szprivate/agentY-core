"""Alibaba Model Studio (DashScope) endpoints: which URL belongs to which kind of key.

Alibaba sells three kinds of access, each with its OWN key and its OWN base URL; a key
sent to another plan's (or another region's) URL answers 401. One table here, so the
installer, the Settings page, the LLM layer, the embedder and the generation tool all
agree and no caller hardcodes a host.

Sources (checked 2026-10):
  https://www.alibabacloud.com/help/en/model-studio/base-url
  https://help.aliyun.com/en/model-studio/token-plan-team-quickstart   ("Token Plan API Keys start with sk-sp-")
  https://www.alibabacloud.com/help/en/model-studio/coding-plan

* pay-as-you-go: a workspace key, per region. OpenAI-compatible ``<host>/compatible-mode/v1``;
  the native API (image/video generation, tasks) is the same host with ``/api/v1``.
* Token Plan / Coding Plan: a subscription key (``sk-sp-...``) and a dedicated URL. Only the
  OpenAI-compatible and Anthropic-compatible chat APIs are documented for them - no native API,
  so generation and (presumably) embeddings need a pay-as-you-go key. Alibaba restricts both
  plans to interactive AI coding tools; "Custom applications: automated scripts, application
  backends calling the API directly" are listed as unsupported.
"""
from __future__ import annotations

import os
import re
import urllib.error
import urllib.request

PAYG, TOKEN_PLAN, CODING_PLAN = "payg", "token_plan", "coding_plan"
PLAN_LABELS = {PAYG: "Pay-as-you-go", TOKEN_PLAN: "Token Plan", CODING_PLAN: "Coding Plan"}

#: the subscription-plan key prefix (Token Plan; the Coding Plan FAQ says the same for its keys)
PLAN_KEY_PREFIX = "sk-sp-"

# (plan, region) -> OpenAI-compatible base URL.  Order = the order probes and menus use.
ENDPOINTS: dict[tuple[str, str], str] = {
    (PAYG, "singapore"): "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    (PAYG, "beijing"): "https://dashscope.aliyuncs.com/compatible-mode/v1",
    (PAYG, "us"): "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    (PAYG, "hongkong"): "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1",
    (TOKEN_PLAN, "singapore"): "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
    (TOKEN_PLAN, "beijing"): "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    (CODING_PLAN, "singapore"): "https://coding-intl.dashscope.aliyuncs.com/v1",
    (CODING_PLAN, "beijing"): "https://coding.dashscope.aliyuncs.com/v1",
}
REGION_LABELS = {"singapore": "Singapore (international)", "beijing": "China (Beijing)",
                 "us": "US (Virginia)", "hongkong": "China (Hong Kong)"}

#: the one default, used wherever nothing is configured
DEFAULT_BASE_URL = ENDPOINTS[(PAYG, "singapore")]

TERMS_WARNING = (
    "Alibaba restricts Token Plan and Coding Plan keys to interactive AI coding tools "
    "(Claude Code, Codex...). Its docs list 'automated scripts, application backends calling "
    "the API directly' as unsupported and say using a plan key outside the permitted scope is a "
    "violation that can suspend the subscription or ban the key. agentY is such a backend, so "
    "using a plan key here may break Alibaba's terms."
)
PAYG_NEEDED = (
    "Image/video generation (the native /api/v1 API) and embeddings (text-embedding-v4) are not "
    "offered on Token Plan / Coding Plan keys. Set a pay-as-you-go key: DASHSCOPE_PAYG_API_KEY "
    "(and DASHSCOPE_PAYG_BASE_URL for its region)."
)


def host_of(url: str) -> str:
    m = re.match(r"(https?://[^/]+)", (url or "").strip())
    return m.group(1).lower() if m else ""


def is_plan_key(key: str) -> bool:
    return (key or "").strip().startswith(PLAN_KEY_PREFIX)


def plan_of_url(url: str) -> str | None:
    """Which plan a base URL belongs to, or None for an unknown/custom host."""
    h = host_of(url)
    if not h:
        return None
    if h.startswith("https://token-plan."):
        return TOKEN_PLAN
    if h.startswith("https://coding"):
        return CODING_PLAN
    if h.endswith(".aliyuncs.com"):
        return PAYG
    return None


def endpoint_for(plan: str, region: str) -> str:
    """The OpenAI-compatible URL for a plan and region (ValueError when none exists)."""
    try:
        return ENDPOINTS[(plan, region)]
    except KeyError:
        raise ValueError(f"no {PLAN_LABELS.get(plan, plan)} endpoint for region '{region}'") from None


def regions_for(plan: str) -> list[str]:
    return [r for (p, r) in ENDPOINTS if p == plan]


def candidates_for_key(key: str) -> list[tuple[str, str, str]]:
    """(plan, region, url) to try for a key, most likely first. A plan key (sk-sp-) can only be
    a Token or Coding Plan key; any other key is pay-as-you-go."""
    plans = (TOKEN_PLAN, CODING_PLAN) if is_plan_key(key) else (PAYG,)
    return [(p, r, u) for (p, r), u in ENDPOINTS.items() if p in plans]


def native_url(base_url: str) -> str | None:
    """``<host>/api/v1`` for a pay-as-you-go base URL; None for a plan URL (no native API)
    or an empty one."""
    if not (base_url or "").strip():
        return None
    if plan_of_url(base_url) in (TOKEN_PLAN, CODING_PLAN):
        return None
    h = host_of(base_url)
    return f"{h}/api/v1" if h else None


def _env(env, name: str) -> str:
    return ((env if env is not None else os.environ).get(name) or "").strip()


def chat_credentials(env=None) -> tuple[str, str]:
    """(api_key, base_url) the chat models use."""
    key = _env(env, "DASHSCOPE_API_KEY") or _env(env, "ALIBABA_API_KEY")
    return key, _env(env, "DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL


def payg_credentials(env=None) -> tuple[str, str, str]:
    """(api_key, base_url, error) for what needs a pay-as-you-go key (generation, embeddings).

    The dedicated DASHSCOPE_PAYG_API_KEY/_BASE_URL win; otherwise the main key and URL, unless
    those are a Token/Coding Plan (then error says what to set). A main key without an URL uses
    the default region."""
    pkey = _env(env, "DASHSCOPE_PAYG_API_KEY")
    if pkey:
        return pkey, _env(env, "DASHSCOPE_PAYG_BASE_URL") or DEFAULT_BASE_URL, ""
    key, base = chat_credentials(env)
    if not key:
        return "", base, "No DashScope key set (DASHSCOPE_PAYG_API_KEY or DASHSCOPE_API_KEY)."
    if is_plan_key(key) or plan_of_url(base) in (TOKEN_PLAN, CODING_PLAN):
        return "", base, "The DashScope key in use is a Token Plan / Coding Plan key. " + PAYG_NEEDED
    return key, base, ""


def probe(base_url: str, key: str, timeout: float = 6.0) -> bool:
    """One free GET <base>/models; True when it answers 200 with this key."""
    req = urllib.request.Request(base_url.rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - https URLs from the table
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def detect(key: str, probe_fn=probe) -> tuple[str, str, str] | None:
    """(plan, region, url) of the first candidate endpoint that accepts the key, else None."""
    for plan, region, url in candidates_for_key(key):
        if probe_fn(url, key):
            return plan, region, url
    return None
