from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# =========================================================
# JDDC parse script (high-recall / weak-filter version)
#
# Purpose:
# 1) Parse raw JDDC chat txt into session-level user turns
# 2) Score in-scope business scenes with high recall
# 3) Keep usable but ambiguous data in review_pool instead of hard-dropping
# 4) Split into four buckets:
#       - strong_in_scope
#       - multiturn_in_scope
#       - review_pool
#       - true_out_of_scope
#
# Input (recommended first pass):
#   raw/chat_0.1per.txt
#
# Why not use dev_question/dev_answer here?
#   They are evaluation-oriented files, not the best source for full-session restoration.
#   This script is designed for chat_*.txt style raw dialog logs.
# =========================================================

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "raw" / "chat_0.1per.txt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "parsed"

# ----------------------------
# Text patterns
# ----------------------------
END_CHAT_PATTERNS = [
    r"请问还有其他.*帮到您",
    r"感谢您对京东的支持",
    r"祝您生活愉快",
    r"欢迎再次光临",
    r"如有问题.*联系",
    r"点击.*评价",
    r"再见",
]

WEAK_NOISE_PATTERNS = [
    r"^你好+$",
    r"^您好+$",
    r"^在吗$",
    r"^嗯+$",
    r"^哦+$",
    r"^好的+$",
    r"^好吧$",
    r"^谢谢+$",
    r"^谢了$",
    r"^没有了$",
    r"^没了$",
    r"^拜拜$",
]

# Strong out-of-scope categories. These should be truly outside the policy engine scope.
OUT_OF_SCOPE_PATTERNS = {
    "OUT_OF_SCOPE_ACCOUNT_SECURITY": [
        r"密码",
        r"验证码",
        r"账号",
        r"登录不了",
        r"银行卡",
        r"身份证",
        r"绑定手机号",
        r"找回.*账号",
        r"修改密码",
        r"冻结",
    ],
    "OUT_OF_SCOPE_PRODUCT_INFO": [
        r"什么型号",
        r"参数",
        r"尺寸",
        r"尺码",
        r"真伪",
        r"真假",
        r"保修",
        r"功能",
        r"区别是",
        r"怎么用",
        r"能不能用",
        r"兼容",
        r"多大",
        r"多重",
        r"颜色",
        r"规格",
    ],
    "OUT_OF_SCOPE_MERCHANT_SIDE": [
        r"商家朋友",
        r"商家ID",
        r"后台",
        r"评价管理",
        r"入驻",
        r"服务单",
        r"客户补您钱",
        r"仓库收到给客户退款",
        r"运营",
        r"店铺后台",
    ],
}

# Scene patterns: separate strong / weak signals.
SCENE_PATTERNS: dict[str, dict[str, list[str]]] = {
    "RETURN_REFUND": {
        "strong": [r"退货", r"退款", r"退换货", r"申请售后", r"换货", r"返修"],
        "weak": [r"不想要了", r"不要了", r"退掉", r"退回去", r"售后"],
    },
    "QUALITY_ISSUE": {
        "strong": [r"坏了", r"破损", r"碎了", r"质量问题", r"故障", r"损坏"],
        "weak": [r"不能用", r"有问题", r"失灵", r"漏液", r"变质", r"过期"],
    },
    "WRONG_OR_MISSING_ITEM": {
        "strong": [r"少发", r"漏发", r"错发", r"发错", r"少收到", r"缺件"],
        "weak": [r"怎么只收到", r"少了一个", r"没给我发", r"没收到.*配件"],
    },
    "LOGISTICS_EXCEPTION": {
        "strong": [r"丢件", r"丢了", r"物流不更新", r"一直不更新", r"破损了", r"没收到.*显示完成"],
        "weak": [r"催单", r"催促", r"延误", r"显示已收货", r"物流异常", r"一直没动静"],
    },
    "LOGISTICS_QUERY": {
        "strong": [r"今天能到吗", r"什么时候到", r"几时能送", r"预计.*送达", r"到哪了", r"站点", r"自提"],
        "weak": [r"提前送", r"配送员", r"联系电话", r"保持手机畅通", r"什么时候发货", r"到货时间"],
    },
    "ORDER_CANCEL_MODIFY": {
        "strong": [r"取消订单", r"申请取消", r"修改地址", r"改地址", r"收货地址", r"重新下单"],
        "weak": [r"不想买了", r"不要了", r"取消", r"拒收", r"改一下地址"],
    },
    "PROMOTION_COUPON": {
        "strong": [r"优惠券", r"不能叠加", r"满减", r"活动价"],
        "weak": [r"优惠", r"促销", r"plus券", r"店铺券", r"平台券"],
    },
    "PRICE_PROTECTION": {
        "strong": [r"价保", r"保价", r"价格保护"],
        "weak": [r"差价", r"退给我.*差价", r"退到哪里", r"原路返回", r"为什么退.*这么少"],
    },
    "REFUND_PROGRESS": {
        "strong": [r"退款中", r"退款处理", r"退款审核", r"退款多久"],
        "weak": [r"多久到账", r"财务审核", r"什么时候退回来", r"还没到账"],
    },
    "INVOICE_REQUEST": {
        "strong": [r"发票", r"电子发票", r"纸质发票", r"专票", r"增值税"],
        "weak": [r"购物清单", r"抬头", r"换开", r"重开", r"公章", r"报销"],
    },
}

SCENE_PRIORITY = [
    "INVOICE_REQUEST",
    "PRICE_PROTECTION",
    "ORDER_CANCEL_MODIFY",
    "WRONG_OR_MISSING_ITEM",
    "QUALITY_ISSUE",
    "RETURN_REFUND",
    "LOGISTICS_EXCEPTION",
    "LOGISTICS_QUERY",
    "REFUND_PROGRESS",
    "PROMOTION_COUPON",
]

# Scene family: useful for context inheritance.
SCENE_FAMILY = {
    "RETURN_REFUND": "AFTER_SALES",
    "QUALITY_ISSUE": "AFTER_SALES",
    "WRONG_OR_MISSING_ITEM": "AFTER_SALES",
    "REFUND_PROGRESS": "AFTER_SALES",
    "LOGISTICS_EXCEPTION": "LOGISTICS",
    "LOGISTICS_QUERY": "LOGISTICS",
    "ORDER_CANCEL_MODIFY": "ORDER",
    "PROMOTION_COUPON": "PROMOTION",
    "PRICE_PROTECTION": "PROMOTION",
    "INVOICE_REQUEST": "INVOICE",
}

ORDER_HINT_PAT = re.compile(r"\[订单x\]|\[订单编号:|\b订单号\b|咨询订单号")
LINK_HINT_PAT = re.compile(r"\[链接x\]")
PRODUCT_SNAPSHOT_PAT = re.compile(r"\[商品快照")
DIGIT_MASK_PAT = re.compile(r"\[数字x\]")
DATE_MASK_PAT = re.compile(r"\[日期x\]|")
ADDRESS_MASK_PAT = re.compile(r"\[地址x\]")
NAME_MASK_PAT = re.compile(r"\[姓名x\]")

# Context dependency markers.
CONTEXT_MARKERS = [
    r"^这个$", r"^那个$", r"^这个订单$", r"^刚才$", r"^之前$",
    r"^对$", r"^是的$", r"^嗯$", r"^好$", r"^好的$", r"^那我呢$", r"^然后呢",
    r"^那怎么办$", r"^那算了$", r"^这个可以吗$", r"^那我要",
]

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Turn:
    session_id: str
    user_id: str
    speaker: str  # user | assistant
    merchant_flag: int
    raw_fields: list[str]
    text: str


# ----------------------------
# Utils
# ----------------------------
def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def safe_search(pattern: str, text: str) -> bool:
    return re.search(pattern, text) is not None


def count_matches(patterns: list[str], text: str) -> int:
    return sum(1 for p in patterns if safe_search(p, text))


def is_end_chat_text(text: str) -> bool:
    return any(safe_search(p, text) for p in END_CHAT_PATTERNS)


def is_weak_noise_text(text: str) -> bool:
    if len(text) <= 1:
        return True
    return any(safe_search(p, text) for p in WEAK_NOISE_PATTERNS)


def has_context_dependency(text: str) -> bool:
    if len(text) <= 8 and any(safe_search(p, text) for p in CONTEXT_MARKERS):
        return True
    if text in {"对", "是的", "嗯", "好", "好的", "这个", "那个"}:
        return True
    return False


def build_noise_tags(text: str) -> list[str]:
    tags: list[str] = []
    if len(text) <= 8:
        tags.append("short_query")
    if "？" in text or "?" in text:
        tags.append("question_mark")
    if ORDER_HINT_PAT.search(text):
        tags.append("has_order_ref")
    if DIGIT_MASK_PAT.search(text):
        tags.append("has_masked_number")
    if ADDRESS_MASK_PAT.search(text):
        tags.append("has_masked_address")
    if NAME_MASK_PAT.search(text):
        tags.append("has_masked_name")
    if LINK_HINT_PAT.search(text):
        tags.append("has_link")
    if PRODUCT_SNAPSHOT_PAT.search(text):
        tags.append("has_product_snapshot")
    return tags


def parse_chat_line(line: str) -> Turn | None:
    line = line.rstrip("\n")
    if not line.strip():
        return None

    parts = line.split("\t")
    if len(parts) < 7:
        return None

    session_id = parts[0].strip()
    user_id = parts[1].strip()

    try:
        speaker_flag = int(parts[2].strip())
    except Exception:
        return None

    try:
        merchant_flag = int(parts[4].strip())
    except Exception:
        merchant_flag = 0

    text = normalize_text(parts[-1])
    speaker = "assistant" if speaker_flag == 1 else "user"

    return Turn(
        session_id=session_id,
        user_id=user_id,
        speaker=speaker,
        merchant_flag=merchant_flag,
        raw_fields=parts,
        text=text,
    )


def iter_turns(path: Path) -> Iterable[Turn]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = parse_chat_line(line)
            if item is not None:
                yield item


def group_sessions(turns: Iterable[Turn]) -> dict[str, list[Turn]]:
    sessions: dict[str, list[Turn]] = defaultdict(list)
    for t in turns:
        sessions[t.session_id].append(t)
    return sessions


def score_out_of_scope(text: str, merchant_flag: int) -> tuple[float, list[dict[str, Any]]]:
    hints: list[dict[str, Any]] = []
    total = 0.0

    # merchant_flag only contributes a weak hint; never direct exclusion.
    if merchant_flag == 1:
        hints.append({"label": "merchant_flag_1", "score": 0.15, "matched": ["merchant_flag"]})
        total += 0.15

    for label, patterns in OUT_OF_SCOPE_PATTERNS.items():
        matched = [p for p in patterns if safe_search(p, text)]
        if matched:
            # strong category hit: score by number of matched expressions.
            # one hit = 0.7, two or more hits = 1.2+
            score = 0.7 + 0.25 * max(len(matched) - 1, 0)
            hints.append({"label": label, "score": round(score, 3), "matched": matched})
            total += score

    return round(total, 3), hints


def score_scene_candidates(
    text: str,
    history: list[dict[str, str]],
    inherited_scene: str | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    recent_user_texts = [x["content"] for x in history if x["role"] == "user"][-2:]
    history_text = " ".join(recent_user_texts)

    for scene in SCENE_PRIORITY:
        strong_patterns = SCENE_PATTERNS[scene]["strong"]
        weak_patterns = SCENE_PATTERNS[scene]["weak"]

        strong_hit = [p for p in strong_patterns if safe_search(p, text)]
        weak_hit = [p for p in weak_patterns if safe_search(p, text)]
        hist_hit = [p for p in strong_patterns + weak_patterns if history_text and safe_search(p, history_text)]

        score = 0.0
        reasons: list[str] = []

        if strong_hit:
            score += 0.85 + 0.18 * max(len(strong_hit) - 1, 0)
            reasons.append(f"current_strong:{len(strong_hit)}")
        if weak_hit:
            score += 0.35 + 0.12 * max(len(weak_hit) - 1, 0)
            reasons.append(f"current_weak:{len(weak_hit)}")
        if hist_hit:
            score += 0.20
            reasons.append("recent_history_match")

        # context inheritance: if current turn is short/elliptical and previous resolved scene exists,
        # give that scene a moderate bonus rather than forcing exclusion.
        if inherited_scene == scene and has_context_dependency(text):
            score += 0.38
            reasons.append("inherit_same_scene")

        if inherited_scene and SCENE_FAMILY.get(inherited_scene) == SCENE_FAMILY.get(scene) and has_context_dependency(text):
            score += 0.18
            reasons.append("inherit_same_family")

        # small score bonus for explicit references when scene is likely order/after-sales/logistics
        if ORDER_HINT_PAT.search(text) and scene in {"RETURN_REFUND", "REFUND_PROGRESS", "ORDER_CANCEL_MODIFY", "LOGISTICS_EXCEPTION", "LOGISTICS_QUERY", "INVOICE_REQUEST", "PRICE_PROTECTION"}:
            score += 0.08
            reasons.append("has_order_ref")

        if score > 0:
            candidates.append({
                "scene": scene,
                "score": round(score, 3),
                "reasons": reasons,
            })

    candidates.sort(key=lambda x: (x["score"], -SCENE_PRIORITY.index(x["scene"])), reverse=True)
    return candidates


def choose_bucket(
    text: str,
    scene_candidates: list[dict[str, Any]],
    out_of_scope_score: float,
    out_of_scope_hints: list[dict[str, Any]],
    history_len: int,
) -> tuple[str, dict[str, Any]]:
    """
    Buckets:
      - strong_in_scope
      - multiturn_in_scope
      - review_pool
      - true_out_of_scope
    """
    debug: dict[str, Any] = {}

    if is_end_chat_text(text):
        return "true_out_of_scope", {"reason": "end_chat_text"}

    if is_weak_noise_text(text) and not scene_candidates:
        return "true_out_of_scope", {"reason": "weak_noise_no_scene"}

    top1 = scene_candidates[0] if scene_candidates else None
    top2 = scene_candidates[1] if len(scene_candidates) > 1 else None

    top1_score = top1["score"] if top1 else 0.0
    top2_score = top2["score"] if top2 else 0.0
    margin = top1_score - top2_score
    needs_context = has_context_dependency(text) or history_len >= 4

    debug.update({
        "top1_score": top1_score,
        "top2_score": top2_score,
        "margin": round(margin, 3),
        "needs_context": needs_context,
        "out_of_scope_score": out_of_scope_score,
        "out_of_scope_labels": [x["label"] for x in out_of_scope_hints],
    })

    # Truly out of scope only when strong oos evidence and no real in-scope signal.
    if out_of_scope_score >= 1.2 and top1_score < 0.35:
        debug["reason"] = "strong_out_of_scope_without_in_scope_signal"
        return "true_out_of_scope", debug

    # Strong single-turn candidate.
    if top1_score >= 0.78 and margin >= 0.15 and out_of_scope_score < 1.2 and not needs_context:
        debug["reason"] = "high_confidence_single_turn"
        return "strong_in_scope", debug

    # Multi-turn but still clearly usable.
    if top1_score >= 0.55 and out_of_scope_score < 1.2 and needs_context:
        debug["reason"] = "usable_but_context_dependent"
        return "multiturn_in_scope", debug

    # Medium confidence, or in-scope / oos mixed. Keep for review instead of discarding.
    if top1_score >= 0.38:
        debug["reason"] = "medium_confidence_or_mixed_signal"
        return "review_pool", debug

    # If there is some out-of-scope hint but also weak scene evidence, still review.
    if out_of_scope_score > 0 and top1_score > 0:
        debug["reason"] = "mixed_scope_signal"
        return "review_pool", debug

    # If very short but history exists, keep in review for possible later recovery.
    if needs_context and history_len > 0:
        debug["reason"] = "context_dependent_but_scene_unclear"
        return "review_pool", debug

    debug["reason"] = "low_signal_default"
    return "true_out_of_scope", debug


# ----------------------------
# Build examples per session
# ----------------------------
def build_examples_from_session(session_turns: list[Turn], history_keep: int = 8) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    dialog_history: list[dict[str, str]] = []

    # Recent high-confidence scene used for inheritance.
    last_confident_scene: str | None = None

    for idx, turn in enumerate(session_turns):
        if turn.speaker != "user":
            dialog_history.append({"role": "assistant", "content": turn.text})
            continue

        text = turn.text
        history_snapshot = dialog_history[-history_keep:]

        # collect immediate assistant replies after current user turn (for later DPO/reference, not truth)
        next_assistant_replies: list[str] = []
        j = idx + 1
        while j < len(session_turns) and session_turns[j].speaker == "assistant":
            next_assistant_replies.append(session_turns[j].text)
            j += 1

        out_score, out_hints = score_out_of_scope(text, turn.merchant_flag)
        scene_candidates = score_scene_candidates(text, history_snapshot, inherited_scene=last_confident_scene)
        bucket, bucket_debug = choose_bucket(
            text=text,
            scene_candidates=scene_candidates,
            out_of_scope_score=out_score,
            out_of_scope_hints=out_hints,
            history_len=len(history_snapshot),
        )

        top_scene = scene_candidates[0]["scene"] if scene_candidates else None
        if bucket in {"strong_in_scope", "multiturn_in_scope"} and top_scene is not None:
            last_confident_scene = top_scene

        example = {
            "sample_id": f"{turn.session_id}_{idx}",
            "session_id": turn.session_id,
            "turn_index": idx,
            "user_id": turn.user_id,
            "current_user_query": text,
            "dialog_history": history_snapshot,
            "next_assistant_replies": next_assistant_replies,
            "scene_candidates": scene_candidates,
            "out_of_scope_score": out_score,
            "out_of_scope_hints": out_hints,
            "bucket": bucket,
            "bucket_debug": bucket_debug,
            "noise_tags": build_noise_tags(text),
            "meta": {
                "source": "JDDC_chat_txt",
                "merchant_flag": turn.merchant_flag,
            },
        }
        examples.append(example)

        dialog_history.append({"role": "user", "content": text})

    return examples


# ----------------------------
# IO
# ----------------------------
def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="High-recall JDDC parser for ecommerce customer-service post-training dataset construction.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Path to raw JDDC chat txt. First pass: use raw/chat_0.1per.txt")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory to save parsed jsonl outputs")
    parser.add_argument("--history-keep", type=int, default=8, help="How many previous turns to keep in dialog_history")
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    history_keep = args.history_keep

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    turns = list(iter_turns(input_path))
    sessions = group_sessions(turns)

    strong_rows: list[dict[str, Any]] = []
    multiturn_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    oos_rows: list[dict[str, Any]] = []

    scene_counter = Counter()
    bucket_counter = Counter()
    oos_label_counter = Counter()

    for _, session_turns in sessions.items():
        rows = build_examples_from_session(session_turns, history_keep=history_keep)
        for row in rows:
            bucket = row["bucket"]
            bucket_counter[bucket] += 1

            if row["scene_candidates"]:
                scene_counter[row["scene_candidates"][0]["scene"]] += 1

            for hint in row["out_of_scope_hints"]:
                oos_label_counter[hint["label"]] += 1

            if bucket == "strong_in_scope":
                strong_rows.append(row)
            elif bucket == "multiturn_in_scope":
                multiturn_rows.append(row)
            elif bucket == "review_pool":
                review_rows.append(row)
            else:
                oos_rows.append(row)

    write_jsonl(output_dir / "strong_in_scope.jsonl", strong_rows)
    write_jsonl(output_dir / "multiturn_in_scope.jsonl", multiturn_rows)
    write_jsonl(output_dir / "review_pool.jsonl", review_rows)
    write_jsonl(output_dir / "true_out_of_scope.jsonl", oos_rows)

    stats = {
        "input_file": str(input_path),
        "num_turns": len(turns),
        "num_sessions": len(sessions),
        "num_strong_in_scope": len(strong_rows),
        "num_multiturn_in_scope": len(multiturn_rows),
        "num_review_pool": len(review_rows),
        "num_true_out_of_scope": len(oos_rows),
        "top_scene_distribution": dict(scene_counter),
        "bucket_distribution": dict(bucket_counter),
        "out_of_scope_hint_distribution": dict(oos_label_counter),
    }
    (output_dir / "parse_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
