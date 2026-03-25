from __future__ import annotations

"""
parse_jddc.py
=================

第一阶段 JDDC 解析脚本：
只做“结构恢复 + 用户轮抽取 + 最粗桶分配”，不做 scene 分类、权限判断、槽位抽取。

职责边界（严格控制）：
1. 解析 JDDC 原始 txt 行；
2. 按 session_id 还原会话；
3. 仅对 user turn 生成中间态样本；
4. 保留最近若干轮上下文，供二阶段 LLM 使用；
5. 只做最粗粒度桶分配：
   - ready_for_routing
   - needs_context
   - discard_noise

明确不做：
- 不做 scene 分类
- 不做 in-scope / out-of-scope 语义判断
- 不做 merchant_flag 语义使用
- 不做 structured_input 抽取
- 不做 oracle 生成

为什么这样设计：
- JDDC 的结构字段（session_id / user_id / 是否客服发送 / 内容）足够支持“会话恢复”；
- 但 scene、权限边界、结构化槽位都属于语义理解，应该交给二阶段 LLM；
- 第一阶段 parser 的目标是“高召回地恢复用户问题”，而不是“高精度地做业务分类”。

本版相较前一版的重要修正：
- 修正 1：新增“上下文快照行过滤”，这类行不单独生成 sample，只留在 dialog_history 里；
- 修正 2：把明显收尾句/确认句从 ready_for_routing 调整出去；
- 修正 3：如果短句里带明确业务词（修改地址、发票、价保、退款、物流、自提、送货、购物清单等），
         即便句子较短，也优先进入 ready_for_routing，而不是 needs_context。
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# =========================
# 默认路径
# =========================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "raw" / "chat_0.1per.txt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "parsed"


# =========================
# 粗清洗规则（仅限 parser 阶段）
# =========================
# 1) 纯噪声：这些文本本身几乎不包含可路由业务信息，继续送给 LLM 只会浪费成本。
NOISE_ONLY_PATTERNS = [
    r"^你好+$",
    r"^您好+$",
    r"^在吗$",
    r"^嗯+$",
    r"^哦+$",
    r"^哈+$",
    r"^好的+$",
    r"^ok$",
    r"^OK$",
    r"^谢谢+$",
    r"^多谢+$",
    r"^没有了$",
    r"^没了$",
    r"^拜拜$",
    r"^再见$",
    r"^是的$",
    r"^对的$",
    r"^对$",
]

# 2) 收尾/确认句：不一定是垃圾，但通常不该直接进入 ready_for_routing。
#    这类句子如果有上下文，优先丢到 needs_context；没有上下文再进 discard_noise。
ENDING_OR_ACK_PATTERNS = [
    r"那就算了",
    r"那算了",
    r"行吧",
    r"好吧",
    r"知道了",
    r"明白了",
]

# 3) 指代/承接类：这些句子往往单独看不完整，后续需要 LLM 结合上下文处理。
ANAPHORA_PATTERNS = [
    r"这个",
    r"那个",
    r"这单",
    r"那单",
    r"这个订单",
    r"那个订单",
    r"上面",
    r"刚才",
    r"之前",
    r"然后呢",
    r"那我",
    r"那就",
    r"那这样",
    r"这样的话",
    r"可以吗$",
]

# 4) 业务关键词：只要命中这些词，即使句子较短，也优先交给二阶段 LLM，而不是 needs_context。
#    目的是提高离线语料采集的召回率，避免把明显业务问题扔进低优先级桶。
BUSINESS_KEYWORD_PATTERNS = [
    r"修改地址",
    r"改地址",
    r"开发票",
    r"发票",
    r"购物清单",
    r"价保",
    r"保价",
    r"退款",
    r"退货",
    r"售后",
    r"物流",
    r"快递",
    r"自提",
    r"送货",
    r"配送",
    r"取消订单",
    r"取消",
    r"优惠券",
    r"满减",
    r"错发",
    r"漏发",
    r"少发",
    r"破损",
    r"坏了",
]

# 5) 结构快照行：这些文本是上下文补充，不应单独生成 sample。
#    它们保留在 history 中，对 LLM 理解当前用户问题仍有帮助。
SNAPSHOT_ONLY_PATTERNS = [
    r"^\[订单编号:.*\]$",
    r"^\[订单x\]$",
    r"^\[链接x\]$",
    r"^\[商品快照\]$",
    r"^顾客通过点击web咚咚.*$",
]

# 用于打上下文标签，而不是做语义分类。
MASKED_PATTERNS = {
    "has_order_snapshot": re.compile(r"\[订单x\]|\[订单编号:"),
    "has_link": re.compile(r"\[链接x\]"),
    "has_product_snapshot": re.compile(r"\[商品快照\]"),
    "has_masked_fields": re.compile(r"\[(订单x|地址x|姓名x|数字x|金额x|日期x|时间x)\]"),
}


# =========================
# 数据结构
# =========================
@dataclass
class Turn:
    """原始对话的一行记录。"""

    session_id: str
    user_id: str
    speaker: str  # "user" | "assistant"
    text: str
    raw_fields: list[str]
    line_no: int
    sku: str | None = None
    aux_flag_1: str | None = None
    aux_flag_2: str | None = None


# =========================
# 基础工具函数
# =========================
def normalize_text(text: str) -> str:
    """做非常轻量的规范化，避免过度清洗导致信息损失。"""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def match_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def is_null_like(text: str) -> bool:
    """JDDC 里会出现 NULL/nan 之类占位文本，parser 阶段直接视为空。"""
    return text == "" or text.lower() in {"null", "nan", "none"}


def is_snapshot_only(text: str) -> bool:
    """
    判断是否为“上下文快照行”。

    满足下面条件之一时，我们不把它当作当前用户问题：
    - 基本完全匹配结构快照模式；
    - 以 [] 包裹，核心只是订单/金额/时间/链接/快照等结构信息；
    - 不包含真正自然语言问句。
    """
    if match_any(SNAPSHOT_ONLY_PATTERNS, text):
        return True

    if text.startswith("[") and text.endswith("]"):
        # 如果是中括号包裹，并且内容主要是订单/金额/时间/链接类快照，则视为上下文块。
        snapshot_keywords = ["订单", "金额", "时间", "链接", "快照", "地址", "姓名"]
        natural_question_markers = ["吗", "么", "怎么", "为什么", "可以", "能不能", "?", "？"]
        if any(k in text for k in snapshot_keywords) and not any(m in text for m in natural_question_markers):
            return True

    return False


def is_discard_noise(text: str) -> bool:
    """
    判断是否为纯噪声。

    注意：这里是“保守丢弃”，只丢弃非常明显没有业务价值的文本。
    我们宁可多留一些 review 成本，也不希望离线语料采集阶段错杀太多有效样本。
    """
    if is_null_like(text):
        return True
    if len(text) <= 1:
        return True
    if match_any(NOISE_ONLY_PATTERNS, text):
        return True
    return False


def has_business_keyword(text: str) -> bool:
    """判断短句里是否有明显业务词，命中则优先交给后续 LLM。"""
    return match_any(BUSINESS_KEYWORD_PATTERNS, text)


def needs_context(text: str, history: list[dict[str, str]]) -> bool:
    """
    判断当前 user turn 是否明显依赖上下文。

    这是一个“保守的 needs_context 规则”：
    - 只有短句 + 指代/承接表达，才优先放进 needs_context；
    - 如果虽然短，但带明显业务词，则不要扔进 needs_context，而是交给二阶段 LLM；
    - 没有历史上下文时，不强行判 needs_context。
    """
    if not history:
        return False

    # 修正 3：带明确业务名词的短句，优先进入 ready_for_routing。
    if has_business_keyword(text):
        return False

    # 很短且像承接句，一般需要上下文。
    if len(text) <= 8 and match_any(ANAPHORA_PATTERNS, text):
        return True

    # 中短句 + 指代/承接表达，也倾向依赖上下文。
    if len(text) <= 14 and match_any(ANAPHORA_PATTERNS, text):
        return True

    # 很短、结尾是语气词，且没有业务词，也更像延续对话。
    if len(text) <= 12 and re.search(r"(吗|呢|啊|呀|吧)$", text):
        return True

    return False


def classify_bucket(text: str, history: list[dict[str, str]]) -> str:
    """
    做最粗桶分配。

    规则优先级：
    1. 纯噪声 -> discard_noise
    2. 明显收尾/确认句 -> 优先 needs_context（有历史），否则 discard_noise
    3. 依赖上下文 -> needs_context
    4. 其他 -> ready_for_routing
    """
    if is_discard_noise(text):
        return "discard_noise"

    # 修正 2：明显收尾句不要进 ready_for_routing。
    if match_any(ENDING_OR_ACK_PATTERNS, text):
        if history:
            return "needs_context"
        return "discard_noise"

    if needs_context(text, history):
        return "needs_context"

    return "ready_for_routing"


def detect_context_flags(text: str) -> dict[str, bool]:
    """
    只打上下文相关的弱标签，供二阶段 LLM 使用。
    这些标签不是业务真值，不参与任何 scene / 权限判断。
    """
    flags = {name: False for name in MASKED_PATTERNS.keys()}
    for name, pat in MASKED_PATTERNS.items():
        if pat.search(text):
            flags[name] = True
    return flags


# =========================
# 解析原始行
# =========================
def parse_chat_line(line: str, line_no: int) -> Turn | None:
    """
    解析 JDDC 原始行。

    这里采用“尽量保守”的策略：
    - session_id / user_id / speaker_flag 只要能稳拿，就先拿；
    - 其余字段不在 parser 阶段做强语义解释，只保存在 raw_fields 里；
    - 这样做是为了避免 parser 过早绑定我们对某列含义的假设。

    当前根据实际数据的稳定观察：
    - parts[0] 是 session_id
    - parts[1] 是 user_id
    - parts[2] 是 0/1，区分 user / assistant
    - 最后一列是 text
    - 中间列里常有 sku / 辅助字段，但 parser 阶段只做原样保留
    """
    line = line.rstrip("\n")
    if not line.strip():
        return None

    parts = line.split("\t")
    if len(parts) < 4:
        return None

    session_id = parts[0].strip()
    user_id = parts[1].strip()

    try:
        speaker_flag = int(parts[2].strip())
    except Exception:
        return None

    speaker = "assistant" if speaker_flag == 1 else "user"
    text = normalize_text(parts[-1])

    # sku 只是结构辅助，不在 parser 阶段做任何业务判断。
    sku = None
    if len(parts) >= 4:
        sku_candidate = parts[3].strip()
        if sku_candidate and sku_candidate.lower() not in {"null", "nan", "none"}:
            sku = sku_candidate

    aux_flag_1 = parts[4].strip() if len(parts) > 4 else None
    aux_flag_2 = parts[5].strip() if len(parts) > 5 else None

    return Turn(
        session_id=session_id,
        user_id=user_id,
        speaker=speaker,
        text=text,
        raw_fields=parts,
        line_no=line_no,
        sku=sku,
        aux_flag_1=aux_flag_1,
        aux_flag_2=aux_flag_2,
    )


def iter_turns(path: Path) -> Iterable[Turn]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            turn = parse_chat_line(line, idx)
            if turn is not None:
                yield turn


def group_sessions(turns: Iterable[Turn]) -> dict[str, list[Turn]]:
    sessions: dict[str, list[Turn]] = defaultdict(list)
    for t in turns:
        sessions[t.session_id].append(t)
    return sessions


# =========================
# 生成中间态样本
# =========================
def build_samples_from_session(session_turns: list[Turn], history_window: int = 8) -> tuple[list[dict[str, Any]], int]:
    """
    从单个 session 中恢复 user-turn 样本。

    返回：
    - samples: 当前 session 生成的样本列表
    - skipped_snapshot_count: 被识别为“上下文快照行”的 user turn 数量

    关键设计：
    - assistant turn 不生成 sample，但进入 history；
    - user turn 如果是 snapshot-only，则只进 history，不生成 sample；
    - 其他 user turn 一律生成 sample，再根据最粗规则分桶。
    """
    samples: list[dict[str, Any]] = []
    dialog_history: list[dict[str, str]] = []
    skipped_snapshot_count = 0

    for idx, turn in enumerate(session_turns):
        # assistant turn：只保留在上下文中
        if turn.speaker == "assistant":
            dialog_history.append({"role": "assistant", "content": turn.text})
            continue

        # user turn
        text = turn.text

        # 修正 1：快照行只保留到上下文，不单独成 sample
        if is_snapshot_only(text):
            skipped_snapshot_count += 1
            dialog_history.append({"role": "user", "content": text})
            continue

        history_snapshot = dialog_history[-history_window:]
        bucket = classify_bucket(text, history_snapshot)

        previous_assistant_reply = None
        for h in reversed(history_snapshot):
            if h["role"] == "assistant":
                previous_assistant_reply = h["content"]
                break

        # 仅收集紧随其后的连续 assistant 回复，作为后续多轮/DPO 参考，不是真值。
        next_assistant_replies: list[str] = []
        j = idx + 1
        while j < len(session_turns) and session_turns[j].speaker == "assistant":
            next_assistant_replies.append(session_turns[j].text)
            j += 1

        context_flags = detect_context_flags(text)
        context_flags["needs_context"] = bucket == "needs_context"

        sample = {
            "sample_id": f"{turn.session_id}_{idx}",
            "session_id": turn.session_id,
            "user_id": turn.user_id,
            "turn_index": idx,
            "line_no": turn.line_no,
            "speaker": "user",
            "current_user_query": text,
            "dialog_history": history_snapshot,
            "previous_assistant_reply": previous_assistant_reply,
            "next_assistant_replies": next_assistant_replies,
            "sku": turn.sku,
            "has_sku": turn.sku is not None,
            "coarse_bucket": bucket,
            "context_flags": context_flags,
            "meta": {
                "source": "JDDC_chat_0.1per",
                "raw_fields": turn.raw_fields,
                "aux_flag_1": turn.aux_flag_1,
                "aux_flag_2": turn.aux_flag_2,
            },
        }
        samples.append(sample)

        # 当前 user turn 无论是否成 sample，都应该进入后续上下文
        dialog_history.append({"role": "user", "content": text})

    return samples, skipped_snapshot_count


# =========================
# 落盘
# =========================
def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# CLI
# =========================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse JDDC raw txt into intermediate user-turn samples for LLM routing."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to raw JDDC txt file, e.g. raw/chat_0.1per.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save parsed jsonl files.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=8,
        help="How many recent turns to keep in dialog_history for each sample.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    turns = list(iter_turns(input_path))
    sessions = group_sessions(turns)

    parsed_user_turns: list[dict[str, Any]] = []
    ready_for_routing: list[dict[str, Any]] = []
    needs_context_rows: list[dict[str, Any]] = []
    discard_noise_rows: list[dict[str, Any]] = []

    bucket_counter = Counter()
    snapshot_skipped_total = 0

    for _, session_turns in sessions.items():
        rows, skipped_snapshot_count = build_samples_from_session(
            session_turns=session_turns,
            history_window=args.history_window,
        )
        snapshot_skipped_total += skipped_snapshot_count

        for row in rows:
            parsed_user_turns.append(row)
            bucket = row["coarse_bucket"]
            bucket_counter[bucket] += 1

            if bucket == "ready_for_routing":
                ready_for_routing.append(row)
            elif bucket == "needs_context":
                needs_context_rows.append(row)
            else:
                discard_noise_rows.append(row)

    write_jsonl(output_dir / "parsed_user_turns.jsonl", parsed_user_turns)
    write_jsonl(output_dir / "ready_for_routing.jsonl", ready_for_routing)
    write_jsonl(output_dir / "needs_context.jsonl", needs_context_rows)
    write_jsonl(output_dir / "discard_noise.jsonl", discard_noise_rows)

    stats = {
        "input_file": str(input_path),
        "num_turns": len(turns),
        "num_sessions": len(sessions),
        "num_parsed_user_turns": len(parsed_user_turns),
        "num_ready_for_routing": len(ready_for_routing),
        "num_needs_context": len(needs_context_rows),
        "num_discard_noise": len(discard_noise_rows),
        "num_snapshot_only_skipped": snapshot_skipped_total,
        "bucket_distribution": dict(bucket_counter),
        "history_window": args.history_window,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parse_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
