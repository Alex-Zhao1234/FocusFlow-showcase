
"""Batch LLM judges for pipeline stages S5 (quality) and S6 (mode fit).

Two judges share one implementation, invoked by the orchestrator:

- ``judge_quality`` (S5): mode-independent review of three low-quality
  categories (clickbait, divisive, empty content).
- ``judge_mode_fit`` (S6): decides whether a video fits the current mode and
  generates the user-facing recommendation reason in the same LLM call.

Design points
-------------
Judges never touch the database. Reads and writes belong to the caller, for
three reasons: quality verdicts are content-level while filter results are
per-user/per-refresh, so mixing them would force user context into what should
be a pure LLM-judgement function; a single writer avoids two write paths into
the verdict cache; and a db-free judge can be unit-tested offline at zero cost.
Crash-safety ("already-judged batches are not lost") is preserved by exposing
each judge as an async generator that yields batch by batch, so the caller can
persist each batch immediately — which also fits progressive rendering.
Channel names for ``compress`` are supplied by the caller (``channel_titles``).
Degraded verdicts must not be cached; they are identified solely by
``degraded=True`` (``confidence == 0`` is a legitimate LLM answer, not a flag).

Three layers of protection per batch (identical for both judges):

1. Schema validation: PydanticAI ``output_type=list[...]``; the framework
   retries ``LLM_MAX_SCHEMA_RETRIES`` times on schema failure.
2. Batch validation, injected per judge: S5 checks ID mismatches and missing
   grades (``grade=None`` is schema-valid, so code must catch it); S6 checks ID
   mismatches only. ``check_ids -> bool`` is the core protocol; the validators
   wrap it. Both failure kinds share one whole-batch retry with a
   failure-specific retry prompt. If it still fails, the batch is escalated
   once to the stronger model (when allowed), and then degraded.
3. Differentiated degradation: ``wind_down`` (strict mode) fails closed;
   other modes fail open — quality falls back to the statistical score against
   ``DEGRADE_PASS_THRESHOLD``, fit gives a neutral ``fit_score=0.5`` with a
   "not mode-judged" reason.

Deterministic in-batch shuffling: before building the prompt, the batch is
canonically sorted and shuffled with a content-derived seed, so prompt order
is a pure function of batch *content*. This removes position bias caused by
retrieval order (which systematically favours subscribed content). Results are
reordered back to input order, so downstream code is unaffected.

Grade normalisation: for quality batches ``quality_grade`` is the single
source of truth; ``passed`` is derived in code as
``grade >= QUALITY_GRADE_PASS_MIN`` right after the LLM returns.
"""
import asyncio
import hashlib
import json
import logging
import random
from collections.abc import AsyncIterator, Callable, Iterator

from pydantic_ai import Agent

import config
from agent import _model_id, _model_settings
from agent.prompts import PROMPTS
from core.viewing_plan import PlanRuling   # S6 plan-ruling prompt segment (pure model, no import cycle)
from config import (
    COMPRESS_DESC_CHARS, COMPRESS_TAGS_MAX, DEGRADE_PASS_THRESHOLD,
    ESCALATE_CONF_MEAN_MIN, LLM_BATCH_SIZE, LLM_MAX_SCHEMA_RETRIES, LLM_MODEL,
    LLM_MODEL_FALLBACK, LLM_TIMEOUT_SEC, MODE_LLM_OVERRIDES,
    QUALITY_GRADE_PASS_MIN, YT_CATEGORY_NAMES,
)
from core import llmmeter                    # token-usage metering for the judge stage
from core.logsafe import safe_repr           # redact secrets from exceptions before logging
from core.models import Mode, ModeFitVerdict, QualityVerdict, Video

logger = logging.getLogger(__name__)

# Whole-batch retries after a batch-validation failure
_ID_RETRY_MAX = 1

# Retry prompts, keyed by failure type
_RETRY_PROMPTS = {
    "ID_MISMATCH": ("上一次输出的 video_id 集合与输入不符(缺失/多出/重复)。"
                    "请重新判定,严格逐条对应以下输入:\n"),
    "GRADE_MISSING": ("上一次输出中存在 quality_grade 缺失的条目。"
                      "每条判定必须给出 0/1/2/3 四档之一的 quality_grade。"
                      "请重新判定以下输入:\n"),
}


# ============================================================================
# Pure functions (no LLM, no db; unit-testable offline)
# ============================================================================

def compress(v: Video, channel_titles: dict[str, str] | None = None) -> dict:
    """Compress a Video into the summary dict fed to both judges.

    Balances giving the LLM enough signal against token cost; truncation
    limits live in config. The channel name comes from the caller (judges are
    db-free) and falls back to the channel_id. ``category_id`` is mapped to
    the official English category name; unmapped categories are omitted.
    """
    out = {
        "id": v.video_id,
        "title": v.title,
        "desc": v.description[:COMPRESS_DESC_CHARS],
        "tags": v.tags[:COMPRESS_TAGS_MAX],
        "channel": (channel_titles or {}).get(v.channel_id, v.channel_id),
        "duration_sec": v.duration_sec,
        "view": v.view_count,
        "like": v.like_count,
        "comment": v.comment_count,
        "like_hidden": v.like_hidden,
    }
    category = YT_CATEGORY_NAMES.get(v.category_id) if v.category_id is not None else None
    if category:
        out["category"] = category
    return out


def check_ids(batch: list[Video], verdicts: list) -> bool:
    """True iff the verdict IDs match the batch IDs one-to-one.

    Guards against the classic batch-LLM failure: attaching a verdict to the
    wrong video, or dropping/inventing IDs. Schema validation cannot catch
    this, so it is the second protection layer.
    """
    out_ids = [vd.video_id for vd in verdicts]
    return len(out_ids) == len(batch) and set(out_ids) == {v.video_id for v in batch}


def validate_quality_batch(batch: list[Video], verdicts: list) -> str | None:
    """S5 batch validator: ID mismatch first, then missing quality grades."""
    if not check_ids(batch, verdicts):
        return "ID_MISMATCH"
    if any(vd.quality_grade is None for vd in verdicts):
        return "GRADE_MISSING"
    return None


def validate_fit_batch(batch: list[Video], verdicts: list) -> str | None:
    """S6 batch validator: ID mismatch only."""
    return None if check_ids(batch, verdicts) else "ID_MISMATCH"


def shuffled_copy(batch: list[Video]) -> list[Video]:
    """Return a deterministically shuffled copy of the batch for the prompt.

    LLMs judging a list are biased by item position, and retrieval order is
    not random (it favours subscribed content). Two steps are both required:
    (1) sort by video_id to erase input order, (2) shuffle with a seed derived
    from the sorted IDs. Prompt order is then a reproducible function of batch
    content, and differs between batches.
    """
    key = "".join(sorted(v.video_id for v in batch)).encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")  # 64-bit content seed

    out = sorted(batch, key=lambda v: v.video_id)   # canonical order; original list untouched
    random.Random(seed).shuffle(out)
    return out


def should_escalate(*, failure: str | None, exception: bool, confidences: list,
                    already_escalated: bool) -> bool:
    """Decide whether to re-judge a batch with the stronger fallback model.

    - Exceptions never escalate; a batch escalates at most once.
    - A batch that still fails validation after retry escalates (stronger
      models follow output format more reliably).
    - Otherwise escalate when mean confidence < ``ESCALATE_CONF_MEAN_MIN``
      (0.0 disables this trigger, the current default).

    ``confidences`` must contain only real LLM judgements; mixing in degraded
    outputs would trigger escalation every time.
    """
    if already_escalated or exception:
        return False
    if failure is not None:
        return True
    if not confidences:
        return False
    return sum(confidences) / len(confidences) < ESCALATE_CONF_MEAN_MIN


def _chunked(items: list, size: int) -> Iterator[list]:
    """Yield consecutive slices of ``size`` items."""
    for i in range(0, len(items), size):
        yield items[i: i + size]


def _degrade_quality(batch: list[Video], mode: Mode,
                     stat_scores: dict[str, float]) -> list[QualityVerdict]:
    """Quality verdicts without the LLM (timeout, error, or failed validation).

    Strict mode fails closed (reject everything). Other modes fail open,
    passing videos whose S4 statistical score reaches
    ``DEGRADE_PASS_THRESHOLD`` — better to let something through than show an
    empty screen. A missing score counts as 0.0 (rejected), the safe direction.

    ``degraded=True`` is the only degradation marker; the caller uses it to
    skip the verdict cache. ``quality_grade`` is always None and
    ``confidence=0.0`` carries no meaning.
    """
    if mode.strict_mode:
        return [QualityVerdict(video_id=v.video_id, passed=False, degraded=True,
                               categories=[], confidence=0.0,
                               reason="LLM不可用,睡前模式宁缺毋滥")
                for v in batch]
    return [QualityVerdict(video_id=v.video_id,
                           passed=stat_scores.get(v.video_id, 0.0) >= DEGRADE_PASS_THRESHOLD,
                           degraded=True, categories=[], confidence=0.0,
                           reason="LLM不可用,按统计分降级")
            for v in batch]


def _degrade_fit(batch: list[Video], mode: Mode) -> list[ModeFitVerdict]:
    """Mode-fit verdicts without the LLM.

    Strict mode fails closed. Other modes fail open with a neutral
    ``fit_score=0.5``: unlike quality, mode fit cannot be estimated from
    engagement statistics, so there is no score to fall back on. Downstream
    threshold exemptions and rejection attribution key off ``degraded=True``.
    """
    if mode.strict_mode:
        return [ModeFitVerdict(video_id=v.video_id, fit=False, fit_score=0.0,
                               degraded=True, reason="LLM不可用,睡前模式宁缺毋滥")
                for v in batch]
    return [ModeFitVerdict(video_id=v.video_id, fit=True, fit_score=0.5,
                           degraded=True, reason="未经模式判断") for v in batch]


# ============================================================================
# LLM calls (agents built lazily, cached per (prompt, model)).
# MODE_LLM_OVERRIDES applies here: a mode that misjudges too often can be
# switched to a stronger model with one config line; both judges honour it.
# ============================================================================
_AGENTS: dict[tuple[str, str], Agent] = {}


def _get_agent(prompt_name: str, mode: Mode, output_type,
               model_override: str | None = None) -> Agent:
    """Lazily build and cache an Agent keyed by (prompt_name, model_name).

    ``model_override`` (the fallback model on escalation) takes precedence
    over the static per-mode table ``MODE_LLM_OVERRIDES``, which in turn
    overrides the default ``LLM_MODEL``. A missing API key raises on first
    construction and is caught by the batch degradation path.
    """
    model_name = model_override or MODE_LLM_OVERRIDES.get(mode.name, LLM_MODEL)
    key = (prompt_name, model_name)
    if key not in _AGENTS:
        _AGENTS[key] = Agent(
            _model_id(model_name),
            output_type=output_type,                        # framework retries on schema failure
            system_prompt=PROMPTS[prompt_name].system,      # versioned prompt registry
            retries=LLM_MAX_SCHEMA_RETRIES,
            model_settings=_model_settings(model_name, timeout=LLM_TIMEOUT_SEC),
        )
    return _AGENTS[key]


async def _call_agent(agent: Agent, prompt: str) -> list:
    """One batch LLM call — the monkeypatch seam for tests.

    Replacing this function drives all batch logic offline. Token usage is
    recorded in ``core.llmmeter``; metering errors are swallowed so they never
    interrupt judging.
    """
    result = await agent.run(prompt)
    try:
        llmmeter.add(result.usage())
    except Exception:
        pass
    return result.output


_LLM_SEM: asyncio.Semaphore | None = None
_LLM_SEM_LIMIT: int | None = None             # limit the current semaphore was built with


def _llm_semaphore() -> asyncio.Semaphore:
    """Global concurrency gate for judge LLM calls.

    Built lazily from ``config.LLM_MAX_CONCURRENCY`` and rebuilt when the
    limit changes. Both judges share one instance, so the concurrency budget
    is global.
    """
    global _LLM_SEM, _LLM_SEM_LIMIT
    limit = max(1, int(config.LLM_MAX_CONCURRENCY))
    if _LLM_SEM is None or _LLM_SEM_LIMIT != limit:
        _LLM_SEM = asyncio.Semaphore(limit)
        _LLM_SEM_LIMIT = limit
    return _LLM_SEM


async def _call_agent_guarded(agent: Agent, prompt: str) -> list:
    """Gated LLM call — the single exit for every judge call site.

    The gate wraps ``_call_agent`` rather than living inside it: since
    ``_call_agent`` is the test seam, a gate inside it would be bypassed by
    stubs and the "peak concurrency never exceeds the limit" property could
    not be tested.
    """
    async with _llm_semaphore():
        return await _call_agent(agent, prompt)


async def _judge_batch(get_agent: Callable[[str | None], Agent], batch: list[Video],
                       channel_titles: dict[str, str] | None,
                       degrade: Callable[[list[Video]], list],
                       validate: Callable[[list[Video], list], str | None], *,
                       model_name: str, prompt_name: str,
                       allow_escalation: bool = True, lang: str = "zh",
                       prompt_prefix: str = "") -> list:
    """Run the layered protection for one batch (shared by both judges).

    The agent factory, validator and degradation function are injected per
    judge. Flow: call → on validation failure retry the whole batch once →
    escalate once if allowed → otherwise degrade. The degradation path also
    catches every LLM exception. The prompt uses a shuffled copy; results are
    returned in input order with ``degraded=False`` forced on success.
    ``prompt_prefix`` is an optional per-batch instruction segment (used only
    by the fit judge).
    """
    prompt_ver = PROMPTS[prompt_name].version if prompt_name in PROMPTS else "?"  # for logs
    prompt = prompt_prefix + json.dumps(
        [compress(v, channel_titles) for v in shuffled_copy(batch)],
        ensure_ascii=False)                    # keep CJK as-is: readable and fewer tokens
    if lang == "en":
        prompt += "\n\n[Output requirement] Write every `reason` field in English."
    try:
        agent = get_agent(None)
        verdicts = await _call_agent_guarded(agent, prompt)
        retries = 0
        failure = validate(batch, verdicts)    # None = pass, str = failure type
        # ID mismatch and missing grade share a single retry budget.
        while failure is not None and retries < _ID_RETRY_MAX:
            retries += 1
            logger.warning("judges[%s@%s/%s]: 批校验失败(%s),整批重试 %d/%d",
                           prompt_name, prompt_ver, model_name, failure,
                           retries, _ID_RETRY_MAX)
            verdicts = await _call_agent_guarded(agent, _RETRY_PROMPTS[failure] + prompt)
            failure = validate(batch, verdicts)

        # Escalation sits before degradation; the exception path never reaches
        # it. Confidences come only from validated LLM verdicts (fit verdicts
        # have no confidence field, so the mean-confidence trigger is inert).
        confidences = ([c for vd in verdicts
                        if (c := getattr(vd, "confidence", None)) is not None]
                       if failure is None else [])
        if allow_escalation and should_escalate(failure=failure, exception=False,
                                                confidences=confidences,
                                                already_escalated=False):
            if failure is not None:
                logger.warning("judges[%s@%s/%s]: 批校验重试后仍失败(%s),升档 %s",
                               prompt_name, prompt_ver, model_name, failure,
                               LLM_MODEL_FALLBACK)

            else:   # valid batch but low mean confidence: ask the stronger model
                logger.warning("judges[%s@%s/%s]: 批置信均值 %.2f < %.2f,升档 %s",
                               prompt_name, prompt_ver, model_name,
                               sum(confidences) / len(confidences),
                               ESCALATE_CONF_MEAN_MIN, LLM_MODEL_FALLBACK)
            verdicts2 = await _call_agent_guarded(get_agent(LLM_MODEL_FALLBACK), prompt)
            failure2 = validate(batch, verdicts2)
            if failure2 is None:
                # Adopt the escalated result only if it validates; otherwise
                # keep the original failure and fall through to degradation.
                verdicts, failure = verdicts2, None
                logger.info("judges[%s@%s]: 升档批通过校验(%s)",
                            prompt_name, prompt_ver, LLM_MODEL_FALLBACK)

        if failure is not None:
            logger.error("judges[%s@%s/%s]: 批校验(%s)重试后仍不符,该批降级",
                         prompt_name, prompt_ver, model_name, failure)
            return degrade(batch)
    except Exception as e:                           # any LLM error (missing key, timeout...) degrades; never escalates
        logger.error("judges[%s@%s/%s]: LLM 异常(%s),该批降级",
                     prompt_name, prompt_ver, model_name, safe_repr(e))
        return degrade(batch)

    for vd in verdicts:
        vd.degraded = False                          # set by code path only, never by the LLM
    by_id = {vd.video_id: vd for vd in verdicts}
    return [by_id[v.video_id] for v in batch]        # restore input order


# ============================================================================
# S5 · judge_quality / S6 · judge_mode_fit
# ============================================================================

async def judge_quality(
    review: list[Video], mode: Mode, *,
    stat_scores: dict[str, float],
    channel_titles: dict[str, str] | None = None,
) -> AsyncIterator[list[QualityVerdict]]:
    """S5 low-quality review.

    ``review`` holds only the videos S4 routed to LLM review (pass-through and
    cache hits never arrive here). Videos are split into batches of
    ``LLM_BATCH_SIZE``; all batches run concurrently and are yielded in input
    order, so the caller can persist each one as it arrives. Empty input
    yields nothing. ``passed`` is derived here from
    ``grade >= QUALITY_GRADE_PASS_MIN``; ``stat_scores`` is used only for
    degradation.
    """
    model_name = MODE_LLM_OVERRIDES.get(mode.name, LLM_MODEL)
    batches = list(_chunked(review, LLM_BATCH_SIZE))

    async def _one(b: list[Video]) -> list[QualityVerdict]:
        vds = await _judge_batch(
            lambda m=None: _get_agent("quality", mode, list[QualityVerdict],
                                      model_override=m),
            b, channel_titles,
            lambda x: _degrade_quality(x, mode, stat_scores),
            validate_quality_batch,
            model_name=model_name, prompt_name="quality")
        for vd in vds:
            if not vd.degraded and vd.quality_grade is not None:
                vd.passed = vd.quality_grade >= QUALITY_GRADE_PASS_MIN
        return vds

    # Schedule every batch up front, then await in index order: later batches
    # keep running in the background, so concurrency is preserved and the
    # yield order matches the input order.
    tasks = [asyncio.create_task(_one(b)) for b in batches]
    try:
        for t in tasks:
            yield await t
    finally:
        # Consumer stopped early or errored: cancel unfinished tasks and wait
        # for all of them to settle (exceptions collected, not raised).
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _profile_prefix(profile_lines: list[str]) -> str:
    """Build the user-profile segment prepended to the S6 user prompt.

    The user's own declared preferences for this mode are listed by priority,
    and the judge must also attribute each card to the single best-matching
    entry via ``profile_entry`` (or null). Attribution is decided on content
    topic so broad base categories cannot impersonate specific entries. Only
    the user prompt changes; the system prompt stays fixed. Not used for
    intent batches (the caller guarantees mutual exclusion).
    """
    joined = "\n".join(f"{i}. {t}" for i, t in enumerate(profile_lines, 1))
    return (
        "【用户亲笔画像】用户为本模式亲笔声明的偏好(按重要性降序编号):\n"
        f"{joined}\n命中上述偏好的内容应给更高 fit_score 并在 reason 点明"
        "命中哪条;其余仍按本模式常规判据判定——画像不豁免模式排除条款,"
        "质量底线照旧。另外,每张卡都要在 profile_entry 字段输出它的画像归属:"
        "内容明确属于上面第 N 条偏好则输出整数 N(只选最贴合的一条),"
        "都不属于则输出 null——按内容主题从严判,标题沾边但主题不符的不算命中。\n\n")


async def judge_mode_fit(
    candidates: list[Video], mode: Mode, *,
    channel_titles: dict[str, str] | None = None,
    allow_escalation: bool = True,
    lang: str = "zh",
    intent_text: str = "",
    profile_lines: list[str] | None = None,
    neg_pref_lines: list[str] | None = None,
    plan_ruling: "PlanRuling | None" = None,
) -> AsyncIterator[list[ModeFitVerdict]]:
    """S6 mode fit plus recommendation reason, generated in one call.

    The last LLM gate of the pipeline. The prompt is looked up as
    ``fit_{mode.name}``; for ``fit=True`` the ``reason`` becomes the card's
    recommendation text. Batching, protection, degradation and streaming
    mirror ``judge_quality``.

    The user-prompt prefix is chosen by priority (the system prompt never
    changes): intent ruling > viewing-plan ruling > profile entries. The
    negative-preference segment is appended only on the profile branch.
    ``allow_escalation=False`` is used for the first wave, where getting
    cards on screen fast matters more than escalating.
    """
    if intent_text and plan_ruling is not None:
        raise ValueError("judge_mode_fit: intent_text 与 plan_ruling 互斥(spec §3.3)")
    model_name = MODE_LLM_OVERRIDES.get(mode.name, LLM_MODEL)
    batches = list(_chunked(candidates, LLM_BATCH_SIZE))
    prefix = ""
    if intent_text:
        # Explicit user intent: judge relevance to the request instead of the
        # mode's exclusion rules; the quality floor still applies.
        prefix = (
            f"【意图裁定】用户本次显式声明:「{intent_text}」。本批视频均为响应"
            "该声明召回。判定标准改为:内容与该声明贴合即 fit=true、按贴合度给 "
            "fit_score,【不再按本模式常规判据拒绝】——模式排除条款对本批不适用。"
            "仍然照常:content_type 判类型;哗众取宠/煽动对立/空洞内容照拒"
            "(质量底线与模式无关)。fit=true 的 reason 用「应你的要求」口吻点明"
            "它贴合声明的哪一点。\n\n")

    elif plan_ruling is not None:
        prefix = plan_ruling.prompt_prefix()
    elif profile_lines:
        prefix = _profile_prefix(profile_lines)

    # Negative preferences: a separate segment that coexists with the profile
    # segment; never injected for intent or plan batches.
    if not intent_text and plan_ruling is None and neg_pref_lines:
        joined_neg = "\n".join(f"- {t}" for t in neg_pref_lines)
        hit_prefix = config.NEG_PREF_HIT_PREFIX.get(lang, config.NEG_PREF_HIT_PREFIX["zh"])
        prefix += (
            "【用户明确表示不喜欢】用户在本模式下申诉过不喜欢以下内容(强降权):\n"
            f"{joined_neg}\n明确命中者判 fit=false,其 reason 必须以「{hit_prefix}」"
            "开头并指明命中哪一条;疑似命中但另有明显价值者可保留 fit=true,"
            "fit_score 显著压低。宁精勿滥:与上述条目无关的内容照常判定,"
            "不得扩大解释。\n\n")

    async def _one(b: list[Video]) -> list[ModeFitVerdict]:
        return await _judge_batch(
            lambda m=None: _get_agent(f"fit_{mode.name}", mode, list[ModeFitVerdict],
                                      model_override=m),
            b, channel_titles,
            lambda x: _degrade_fit(x, mode),
            validate_fit_batch,
            model_name=model_name, prompt_name=f"fit_{mode.name}",
            allow_escalation=allow_escalation, lang=lang, prompt_prefix=prefix)

    # Same concurrent-but-ordered streaming as judge_quality.
    tasks = [asyncio.create_task(_one(b)) for b in batches]
    try:
        for t in tasks:
            yield await t
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
