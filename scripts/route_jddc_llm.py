from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "parsed" / "parsed_user_turns.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "routed" / "routed_jddc.jsonl"
DEFAULT_STATS = BASE_DIR / "routed" / "route_stats.json"

API_URL = "https://api.deepseek.com/chat/completions"
ALLOWED_MODELS = {"deepseek-chat", "deepseek-reasoner"}

ROUTING_LABELS = {
    "IN_SCOPE",
    "OUT_OF_SCOPE_ACCOUNT_SECURITY",
    "OUT_OF_SCOPE_PRODUCT_INFO",
    "OUT_OF_SCOPE_MERCHANT_SIDE",
}

SCENES = {
    "RETURN_REFUND",
    "QUALITY_ISSUE",
    "WRONG_OR_MISSING_ITEM",
    "LOGISTICS_EXCEPTION",
    "LOGISTICS_QUERY",
    "ORDER_CANCEL_MODIFY",
    "PROMOTION_COUPON",
    "PRICE_PROTECTION",
    "REFUND_PROGRESS",
    "INVOICE_REQUEST",
}

EMOTIONS = {"neutral", "anxious", "angry", "urgent"}
ORDER_STATUSES = {
    "pending_payment", "paid_unshipped", "packed", "shipped", "delivered",
    "signed", "completed", "cancelled", "refund_requested", "refund_processing", "refund_completed",
}
GOODS_TYPES = {"general", "food", "fresh", "customized", "virtual", "service", "medicine"}
ISSUE_TYPES = {"damaged", "malfunction", "expired", "wrong_item", "missing_item", "package_broken", "other"}
DELIVERY_TYPES = {"standard", "same_day", "instant"}
LOGISTICS_STATUSES = {"waiting_pickup", "in_transit", "delayed", "signed", "lost", "exception"}
LOGISTICS_QUERY_TYPES = {"ETA_QUERY", "STATION_STATUS_QUERY", "SELF_PICKUP_QUERY", "EARLY_DELIVERY_QUERY", "CONTACT_COURIER_QUERY"}
INVOICE_QUERY_TYPES = {"INVOICE_STATUS_QUERY", "INVOICE_REISSUE", "INVOICE_TYPE_CHANGE", "INVOICE_MAILING_CHANGE", "SHOPPING_LIST_QUERY"}
PRICE_QUERY_TYPES = {"PRICE_PROTECTION_ELIGIBILITY", "PRICE_PROTECTION_AMOUNT_QUERY", "PRICE_PROTECTION_RETURN_CHANNEL"}

SYSTEM_PROMPT = """
你是电商客服数据治理助手。你的任务不是回答用户，而是把一条 JDDC 用户咨询样本整理成后续规则引擎可消费的中间结构。

请严格遵守：
1. 只做分类与槽位抽取，不要生成客服回复。
2. 只能从“当前用户问题 + 给定历史上下文”中提取明确出现或高度直接可得的信息；不要臆造订单状态、商品类型、签收天数、是否符合价保等信息。
3. 如果问题明显超出客服规则权限，routing_label 必须打成以下之一：
   - OUT_OF_SCOPE_ACCOUNT_SECURITY
   - OUT_OF_SCOPE_PRODUCT_INFO
   - OUT_OF_SCOPE_MERCHANT_SIDE
   这时 scene 必须为 null。
4. 如果属于客服规则范围，routing_label=IN_SCOPE，并从以下 scene 中选一个最合适的主场景：
   - RETURN_REFUND
   - QUALITY_ISSUE
   - WRONG_OR_MISSING_ITEM
   - LOGISTICS_EXCEPTION
   - LOGISTICS_QUERY
   - ORDER_CANCEL_MODIFY
   - PROMOTION_COUPON
   - PRICE_PROTECTION
   - REFUND_PROGRESS
   - INVOICE_REQUEST
5. 允许 needs_context=true；当当前轮本身无法脱离上下文理解时，应设置为 true。
6. 输出必须是合法 JSON，不要附加解释文字。
7. structured_input 里所有未知信息都设为 null；以下字段使用默认值：emotion=neutral, abusive_language=false, repeated_contact_count 直接使用输入里给出的 prior_user_turn_count + 1, manual_review_required=false。
8. 不要输出 schema 里不存在的字段。
""".strip()

USER_PROMPT_TEMPLATE = """
请对下面这条 JDDC 用户咨询样本进行分类与槽位抽取。

【当前用户问题】
{current_user_query}

【最近历史上下文】
{dialog_history}

【上一句客服回复】
{previous_assistant_reply}

【弱提示（仅供参考，不是最终标签）】
{hint_tags}

【先验信息】
prior_user_turn_count = {prior_user_turn_count}
coarse_bucket = {coarse_bucket}

请按下面 JSON 结构输出，所有未知字段都填 null；不要省略字段：
{{
  "routing_label": "IN_SCOPE | OUT_OF_SCOPE_ACCOUNT_SECURITY | OUT_OF_SCOPE_PRODUCT_INFO | OUT_OF_SCOPE_MERCHANT_SIDE",
  "scene": "RETURN_REFUND | QUALITY_ISSUE | WRONG_OR_MISSING_ITEM | LOGISTICS_EXCEPTION | LOGISTICS_QUERY | ORDER_CANCEL_MODIFY | PROMOTION_COUPON | PRICE_PROTECTION | REFUND_PROGRESS | INVOICE_REQUEST | null",
  "needs_context": true,
  "confidence": 0.0,
  "meta_updates": {{
    "noise_tags": [],
    "style_tags": []
  }},
  "structured_input": {{
    "emotion": "neutral | anxious | angry | urgent",
    "abusive_language": false,
    "repeated_contact_count": 1,
    "manual_review_required": false,
    "order_status": null,
    "goods_type": null,
    "is_opened": null,
    "is_used": null,
    "is_redeemed": null,
    "signed_days": null,
    "quality_issue": null,
    "issue_type": null,
    "evidence_provided": null,
    "wrong_item": null,
    "missing_item": null,
    "damaged_package": null,
    "delivery_type": null,
    "delivery_delay_minutes": null,
    "logistics_status": null,
    "coupon_stackable": null,
    "price_protect_eligible": null,
    "logistics_query_type": null,
    "invoice_query_type": null,
    "price_query_type": null
  }}
}}
""".strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def history_to_text(history: list[dict[str, str]], max_turns: int = 8) -> str:
    if not history:
        return "无"
    turns = history[-max_turns:]
    parts = []
    for i, t in enumerate(turns, start=1):
        role = t.get("role", "user")
        content = t.get("content", "")
        parts.append(f"{i}. {role}: {content}")
    return "\n".join(parts)


def default_structured_input(prior_user_turn_count: int) -> dict[str, Any]:
    return {
        "emotion": "neutral",
        "abusive_language": False,
        "repeated_contact_count": prior_user_turn_count + 1,
        "manual_review_required": False,
        "order_status": None,
        "goods_type": None,
        "is_opened": None,
        "is_used": None,
        "is_redeemed": None,
        "signed_days": None,
        "quality_issue": None,
        "issue_type": None,
        "evidence_provided": None,
        "wrong_item": None,
        "missing_item": None,
        "damaged_package": None,
        "delivery_type": None,
        "delivery_delay_minutes": None,
        "logistics_status": None,
        "coupon_stackable": None,
        "price_protect_eligible": None,
        "logistics_query_type": None,
        "invoice_query_type": None,
        "price_query_type": None,
    }


def safe_json_loads(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        # try extract outermost json block
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None


def normalize_routing_label(value: Any) -> str:
    if isinstance(value, str) and value in ROUTING_LABELS:
        return value
    return "IN_SCOPE"


def normalize_scene(value: Any, routing_label: str) -> str | None:
    if routing_label != "IN_SCOPE":
        return None
    if isinstance(value, str) and value in SCENES:
        return value
    return None


def normalize_enum(value: Any, allowed: set[str]) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value in allowed:
        return value
    return None


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.lower().strip()
        if lower in {"true", "yes", "1"}:
            return True
        if lower in {"false", "no", "0"}:
            return False
    return None


def normalize_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if re.fullmatch(r"-?\d+", value):
            return int(value)
    return None


def normalize_structured_input(data: dict[str, Any] | None, prior_user_turn_count: int) -> dict[str, Any]:
    out = default_structured_input(prior_user_turn_count)
    if not isinstance(data, dict):
        return out

    emotion = data.get("emotion")
    if emotion in EMOTIONS:
        out["emotion"] = emotion

    for key in ["abusive_language", "manual_review_required", "is_opened", "is_used", "is_redeemed", "quality_issue", "evidence_provided", "wrong_item", "missing_item", "damaged_package", "coupon_stackable", "price_protect_eligible"]:
        out[key] = normalize_bool(data.get(key))

    repeated = normalize_int(data.get("repeated_contact_count"))
    out["repeated_contact_count"] = repeated if repeated is not None else prior_user_turn_count + 1

    out["signed_days"] = normalize_int(data.get("signed_days"))
    out["delivery_delay_minutes"] = normalize_int(data.get("delivery_delay_minutes"))

    out["order_status"] = normalize_enum(data.get("order_status"), ORDER_STATUSES)
    out["goods_type"] = normalize_enum(data.get("goods_type"), GOODS_TYPES)
    out["issue_type"] = normalize_enum(data.get("issue_type"), ISSUE_TYPES)
    out["delivery_type"] = normalize_enum(data.get("delivery_type"), DELIVERY_TYPES)
    out["logistics_status"] = normalize_enum(data.get("logistics_status"), LOGISTICS_STATUSES)
    out["logistics_query_type"] = normalize_enum(data.get("logistics_query_type"), LOGISTICS_QUERY_TYPES)
    out["invoice_query_type"] = normalize_enum(data.get("invoice_query_type"), INVOICE_QUERY_TYPES)
    out["price_query_type"] = normalize_enum(data.get("price_query_type"), PRICE_QUERY_TYPES)

    return out


def build_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    history_text = history_to_text(sample.get("dialog_history", []))
    previous_assistant_reply = sample.get("previous_assistant_reply") or "无"
    hint_tags = ", ".join(sample.get("hint_tags", [])) if sample.get("hint_tags") else "无"
    user_prompt = USER_PROMPT_TEMPLATE.format(
        current_user_query=sample.get("current_user_query", ""),
        dialog_history=history_text,
        previous_assistant_reply=previous_assistant_reply,
        hint_tags=hint_tags,
        prior_user_turn_count=sample.get("prior_user_turn_count", 0),
        coarse_bucket=sample.get("coarse_bucket", "ready_for_routing"),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def call_deepseek(api_key: str, model: str, messages: list[dict[str, str]], timeout: int = 120) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def route_one_sample(sample: dict[str, Any], api_key: str, model: str, max_retries: int = 5) -> dict[str, Any]:
    prior_user_turn_count = int(sample.get("prior_user_turn_count", 0))
    last_error = None
    raw_text = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_text = call_deepseek(api_key=api_key, model=model, messages=build_messages(sample))
            parsed = safe_json_loads(raw_text)
            if parsed is None:
                raise ValueError("LLM output is not valid JSON")

            routing_label = normalize_routing_label(parsed.get("routing_label"))
            scene = normalize_scene(parsed.get("scene"), routing_label)
            confidence = parsed.get("confidence", 0.0)
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            meta_updates = parsed.get("meta_updates") if isinstance(parsed.get("meta_updates"), dict) else {}
            noise_tags = meta_updates.get("noise_tags") if isinstance(meta_updates.get("noise_tags"), list) else []
            style_tags = meta_updates.get("style_tags") if isinstance(meta_updates.get("style_tags"), list) else []
            structured_input = normalize_structured_input(parsed.get("structured_input"), prior_user_turn_count)
            needs_context = bool(parsed.get("needs_context", False))

            # If out of scope, nullify scene and keep structured input defaults only.
            if routing_label != "IN_SCOPE":
                scene = None
                structured_input = default_structured_input(prior_user_turn_count)

            return {
                "sample_id": sample["sample_id"],
                "session_id": sample.get("session_id"),
                "turn_index": sample.get("turn_index"),
                "user_query": sample.get("current_user_query", ""),
                "dialog_history": sample.get("dialog_history", []),
                "structured_input": structured_input,
                "route_result": {
                    "routing_label": routing_label,
                    "scene": scene,
                    "needs_context": needs_context,
                    "confidence": confidence,
                },
                "meta": {
                    "origin": "human_seed",
                    "source_ref": sample.get("meta", {}).get("source_ref", sample.get("sample_id", "")),
                    "noise_tags": list(dict.fromkeys((sample.get("meta", {}).get("noise_tags", []) or []) + noise_tags)),
                    "style_tags": list(dict.fromkeys((sample.get("meta", {}).get("style_tags", []) or []) + style_tags)),
                    "routing_label": routing_label,
                },
                "debug": {
                    "coarse_bucket": sample.get("coarse_bucket"),
                    "hint_tags": sample.get("hint_tags", []),
                },
                "llm_ok": True,
            }
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            last_error = f"HTTPError(status={status}): {e}"
            if status in {429, 500, 502, 503, 504}:
                time.sleep(min(2 ** attempt, 20))
                continue
            break
        except Exception as e:
            last_error = str(e)
            time.sleep(min(2 ** attempt, 20))
            continue

    return {
        "sample_id": sample["sample_id"],
        "session_id": sample.get("session_id"),
        "turn_index": sample.get("turn_index"),
        "user_query": sample.get("current_user_query", ""),
        "dialog_history": sample.get("dialog_history", []),
        "structured_input": default_structured_input(int(sample.get("prior_user_turn_count", 0))),
        "route_result": {
            "routing_label": "IN_SCOPE",
            "scene": None,
            "needs_context": sample.get("coarse_bucket") == "needs_context",
            "confidence": 0.0,
        },
        "meta": {
            "origin": "human_seed",
            "source_ref": sample.get("meta", {}).get("source_ref", sample.get("sample_id", "")),
            "noise_tags": sample.get("meta", {}).get("noise_tags", []),
            "style_tags": sample.get("meta", {}).get("style_tags", []),
            "routing_label": "IN_SCOPE",
        },
        "debug": {
            "coarse_bucket": sample.get("coarse_bucket"),
            "hint_tags": sample.get("hint_tags", []),
            "error": last_error,
        },
        "llm_ok": False,
    }


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("sample_id")
                if sid:
                    done.add(sid)
            except Exception:
                continue
    return done


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Route JDDC intermediate samples with DeepSeek API.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Input JSONL from parse_jddc.py")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output routed JSONL path")
    parser.add_argument("--stats", type=str, default=str(DEFAULT_STATS), help="Stats JSON path")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="DeepSeek model name")
    parser.add_argument("--max-samples", type=int, default=0, help="Process only first N rows after filtering; 0 means all")
    parser.add_argument("--include-buckets", type=str, default="ready_for_routing,needs_context", help="Comma-separated coarse buckets to include")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep between calls")
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("Please set DEEPSEEK_API_KEY in environment before running.")

    if args.model not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported model: {args.model}. Use one of {sorted(ALLOWED_MODELS)}")

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_path = Path(args.stats)

    rows = read_jsonl(input_path)
    include_buckets = {x.strip() for x in args.include_buckets.split(",") if x.strip()}
    rows = [r for r in rows if r.get("coarse_bucket") in include_buckets]

    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    done_ids = load_done_ids(output_path) if args.resume else set()
    todo_rows = [r for r in rows if r.get("sample_id") not in done_ids]

    routing_counter: dict[str, int] = {}
    scene_counter: dict[str, int] = {}
    llm_fail = 0
    processed = 0

    for row in todo_rows:
        result = route_one_sample(sample=row, api_key=api_key, model=args.model)
        append_jsonl(output_path, result)
        processed += 1

        routing_label = result["route_result"]["routing_label"]
        routing_counter[routing_label] = routing_counter.get(routing_label, 0) + 1
        scene = result["route_result"].get("scene")
        if scene:
            scene_counter[scene] = scene_counter.get(scene, 0) + 1
        if not result.get("llm_ok", False):
            llm_fail += 1

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "model": args.model,
        "include_buckets": sorted(include_buckets),
        "requested_rows": len(rows),
        "already_done": len(done_ids),
        "processed_this_run": processed,
        "routing_distribution_this_run": routing_counter,
        "scene_distribution_this_run": scene_counter,
        "llm_fail_this_run": llm_fail,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
