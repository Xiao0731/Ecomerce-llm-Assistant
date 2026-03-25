from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None

# 【核心整合】：不再自己定义数据结构，一切以 schema.py 为唯一标准
from schema import StructuredInput, OracleOutput

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RULE_CANDIDATES = [
    BASE_DIR / "policies" / "policy_rules_v1.yaml",
    BASE_DIR / "policies" / "policy_rules.yaml",
]

@dataclass
class RuleMatch:
    rule_id: str
    priority: int
    output: dict[str, Any]

# ==========================================
# 1. 槽位检查与追问引擎 
# ==========================================

SLOT_QUESTIONS = {
    "goods_type": "请问商品属于哪一类，例如普通商品、定制商品、虚拟商品或食品生鲜？",
    "quality_issue": "请问是单纯不想要了，还是商品本身存在质量问题？",
    "signed_days": "请问您签收至今大概几天了？",
    "is_opened": "请问商品目前是否已经拆封？",
    "is_used": "请问商品是否已经使用过？",
    "is_redeemed": "请问这个虚拟商品或服务是否已经兑换或核销？",
    "evidence_provided": "方便提供一下相关图片、视频或截图吗？",
    "wrong_item": "请问是收到了错发商品吗？",
    "missing_item": "请问是有商品或配件漏发吗？",
    "logistics_status": "请问当前物流状态显示是什么，例如运输中、已丢件等？",
    "delivery_type": "请问订单配送类型是什么，例如即时达、标准快递或当日达？",
    "delivery_delay_minutes": "大概比预计送达晚了多久？",
    "damaged_package": "请问包裹外包装是否有破损？",
    "order_status": "请问订单当前状态是什么，例如待发货、已发货或退款处理中？",
    "coupon_stackable": "页面是否提示优惠券不可叠加？",
    "price_protect_eligible": "页面是否显示当前订单支持价保？",
}

def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

def _collect_missing_slots(input_data: StructuredInput, required_slots: list[str]) -> list[str]:
    return [slot for slot in required_slots if getattr(input_data, slot, None) is None]

def _slots_to_questions(missing_slots: list[str]) -> list[str]:
    questions: list[str] = []
    for slot in missing_slots:
        question = SLOT_QUESTIONS.get(slot)
        if question and question not in questions:
            questions.append(question)
    return questions

def get_missing_slots_for_scene(scene: str, input_data: StructuredInput) -> list[str]:
    if scene == "RETURN_REFUND":
        base_missing = _collect_missing_slots(input_data, ["goods_type", "quality_issue"])
        if base_missing:
            return base_missing

        goods_type = input_data.goods_type
        quality_issue = input_data.quality_issue

        if quality_issue is True:
            return []
        if goods_type == "customized":
            return []
        if goods_type in {"virtual", "service"}:
            return _collect_missing_slots(input_data, ["is_redeemed"])
        if goods_type in {"food", "fresh", "medicine"}:
            return _collect_missing_slots(input_data, ["is_opened"])
        if goods_type == "general":
            return _collect_missing_slots(input_data, ["signed_days", "is_opened", "is_used"])
        return []

    if scene == "QUALITY_ISSUE":
        return _collect_missing_slots(input_data, ["evidence_provided"])

    if scene == "WRONG_OR_MISSING_ITEM":
        if input_data.wrong_item is None and input_data.missing_item is None:
            return ["wrong_item"]
        return _collect_missing_slots(input_data, ["evidence_provided"])

    if scene == "LOGISTICS_EXCEPTION":
        if input_data.logistics_status == "lost":
            return []
        if input_data.delivery_type is not None or input_data.delivery_delay_minutes is not None:
            return _collect_missing_slots(input_data, ["delivery_type", "delivery_delay_minutes"])
        if input_data.damaged_package is not None or input_data.evidence_provided is not None:
            return _collect_missing_slots(input_data, ["damaged_package", "evidence_provided"])
        return _collect_missing_slots(input_data, ["logistics_status"])

    if scene == "ORDER_CANCEL_MODIFY":
        return _collect_missing_slots(input_data, ["order_status"])
    if scene == "PROMOTION_COUPON":
        return _collect_missing_slots(input_data, ["coupon_stackable"])
    if scene == "PRICE_PROTECTION":
        return _collect_missing_slots(input_data, ["price_protect_eligible"])
    if scene == "REFUND_PROGRESS":
        return _collect_missing_slots(input_data, ["order_status"])

    return []

def check_missing_slots(scene: str, input_data: StructuredInput) -> list[str]:
    missing_slots = get_missing_slots_for_scene(scene, input_data)
    return _slots_to_questions(missing_slots)

# ==========================================
# 2. YAML 解析与匹配
# ==========================================
def load_yaml(path: Path) -> Any:
    if yaml is not None:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return parse_simple_yaml(path.read_text(encoding="utf-8"))

def parse_simple_yaml(text: str) -> Any:
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        stripped = raw_line.lstrip()
        if stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(stripped)
        lines.append((indent, stripped))

    def parse_block(index: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index
        indent, content = lines[index]
        if content.startswith("- "):
            return parse_list(index, indent)
        return parse_dict(index, indent, {})

    def parse_dict(index: int, indent: int, seed: dict[str, Any]) -> tuple[dict[str, Any], int]:
        mapping = dict(seed)
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indentation near: {content}")
            if content.startswith("- "):
                break
            key, raw_value = split_key_value(content)
            index += 1
            if raw_value is None:
                if index < len(lines) and lines[index][0] > current_indent:
                    value, index = parse_block(index)
                    mapping[key] = value
                else:
                    mapping[key] = {}
            else:
                mapping[key] = parse_scalar(raw_value)
        return mapping, index

    def parse_list(index: int, indent: int) -> tuple[list[Any], int]:
        items: list[Any] = []
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent != indent or not content.startswith("- "):
                break
            item_text = content[2:].strip()
            index += 1
            if not item_text:
                value, index = parse_block(index)
                items.append(value)
                continue
            if ":" in item_text:
                key, raw_value = split_key_value(item_text)
                item: dict[str, Any] = {}
                if raw_value is None:
                    if index < len(lines) and lines[index][0] > current_indent:
                        value, index = parse_block(index)
                        item[key] = value
                    else:
                        item[key] = {}
                else:
                    item[key] = parse_scalar(raw_value)
                while index < len(lines):
                    next_indent, next_content = lines[index]
                    if next_indent <= current_indent:
                        break
                    if next_content.startswith("- "):
                        break
                    item, index = parse_dict(index, next_indent, item)
                items.append(item)
                continue
            items.append(parse_scalar(item_text))
        return items, index

    parsed, _ = parse_block(0)
    return parsed

def split_key_value(text: str) -> tuple[str, str | None]:
    key, _, value = text.partition(":")
    key = key.strip()
    value = value.strip()
    if value == "":
        return key, None
    return key, value

def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true": return True
    if lowered == "false": return False
    if lowered in {"null", "none"}: return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner: return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try: return int(value)
    except ValueError: pass
    try: return float(value)
    except ValueError: pass
    return value

def parse_cli_scalar(value: str) -> Any:
    if value.startswith("[") or value.startswith("{"):
        try: return json.loads(value)
        except json.JSONDecodeError: return parse_scalar(value)
    return parse_scalar(value)

def resolve_rule_file(rule_file: str | None) -> Path:
    if rule_file:
        path = Path(rule_file)
        if not path.is_absolute():
            path = BASE_DIR / path
        return path
    for candidate in DEFAULT_RULE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No rule file found.")

def get_field_value(payload: dict[str, Any], field: str) -> Any:
    current: Any = payload
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current

def match_leaf(condition: dict[str, Any], payload: dict[str, Any]) -> bool:
    field = condition.get("field")
    if not field: return False
    actual = get_field_value(payload, field)
    operators = {
        "eq": lambda value: actual == value,
        "ne": lambda value: actual != value,
        "in": lambda value: actual in value if value is not None else False,
        "not_in": lambda value: actual not in value if value is not None else True,
        "gt": lambda value: actual is not None and actual > value,
        "gte": lambda value: actual is not None and actual >= value,
        "lt": lambda value: actual is not None and actual < value,
        "lte": lambda value: actual is not None and actual <= value,
        "exists": lambda value: (actual is not None) is bool(value),
    }
    for operator, expected in condition.items():
        if operator == "field": continue
        checker = operators.get(operator)
        if checker is None: raise ValueError(f"Unsupported operator: {operator}")
        if not checker(expected): return False
    return True

def match_condition(condition: dict[str, Any], payload: dict[str, Any]) -> bool:
    if not condition: return True
    if "all" in condition: return all(match_condition(item, payload) for item in condition["all"])
    if "any" in condition: return any(match_condition(item, payload) for item in condition["any"])
    return match_leaf(condition, payload)

# ==========================================
# 3. 核心评估逻辑
# ==========================================
def evaluate_case(scene: str, input_dict: dict[str, Any], rules: list[dict[str, Any]]) -> OracleOutput:
    try:
        structured_data = StructuredInput(**input_dict)
    except Exception:
        return OracleOutput(
            matched_rule_id="SYSTEM_ERROR_FALLBACK",
            decision="ESCALATE",
            reason_code="SYSTEM_ERROR",
            next_action="TRANSFER_TO_HUMAN",
            escalate_to_human=True,
        )

    if (
        structured_data.manual_review_required
        or structured_data.abusive_language
        or (
            structured_data.repeated_contact_count >= 3
            and structured_data.emotion in ["angry", "urgent"]
        )
    ):
        return OracleOutput(
            matched_rule_id="R001",
            decision="ESCALATE",
            reason_code="EMOTION_ESCALATION",
            next_action="TRANSFER_TO_HUMAN",
            should_apologize=True,
            escalate_to_human=True,
        )

    missing_qs = check_missing_slots(scene, structured_data)
    if missing_qs:
        return OracleOutput(
            matched_rule_id="DYNAMIC_CLARIFY",
            decision="CLARIFY",
            reason_code="INSUFFICIENT_INFORMATION",
            next_action="ASK_FOR_ORDER_INFO",
            need_clarification=True,
            clarify_questions=missing_qs,
            should_apologize=True,
        )

    eval_payload = _model_dump(structured_data)
    eval_payload["scene"] = scene

    for rule in sorted(rules, key=lambda item: item.get("priority", 0), reverse=True):
        when = rule.get("when", {})
        if match_condition(when, eval_payload):
            out = rule.get("then", {})
            return OracleOutput(
                matched_rule_id=str(rule.get("id", "UNKNOWN")),
                decision=out.get("decision", "CLARIFY"),
                reason_code=out.get("reason_code", "UNKNOWN"),
                next_action=out.get("next_action", "ASK_FOR_ORDER_INFO"),
                should_apologize=out.get("should_apologize", False),
                escalate_to_human=out.get("escalate_to_human", False),
            )

    return OracleOutput(
        matched_rule_id="R999",
        decision="CLARIFY",
        reason_code="INSUFFICIENT_INFORMATION",
        next_action="ASK_FOR_ORDER_INFO",
    )

# ==========================================
# 4. CLI 启动与工具
# ==========================================
def load_cases(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = []
    if args.case_file:
        case_path = Path(args.case_file)
        if not case_path.is_absolute():
            case_path = BASE_DIR / case_path
        content = json.loads(case_path.read_text(encoding="utf-8"))
        if isinstance(content, list):
            cases.extend(
                [
                    (item.get("name", f"case_{index + 1}"), item.get("input", item))
                    for index, item in enumerate(content)
                ]
            )
        elif isinstance(content, dict):
            cases.append(("file_case", content))

    if getattr(args, "kv", None):
        kv_case: dict[str, Any] = {}
        for item in args.kv:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            kv_case[key] = parse_cli_scalar(value)
        cases.append(("kv_case", kv_case))
    return cases

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Policy Engine integrated with Global Schema")
    parser.add_argument("--rule-file", help="Path to the rule YAML file.")
    parser.add_argument("--case-file", help="Path to a JSON file containing cases.")
    parser.add_argument("--kv", nargs="*", help="Key-value input for quick testing.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    rule_path = resolve_rule_file(args.rule_file)
    rules_config = load_yaml(rule_path)
    rules = rules_config.get("rules", [])

    results = []
    for case_name, case_payload in load_cases(args):
        scene = case_payload.get("scene", "UNKNOWN")
        decision_obj = evaluate_case(scene, case_payload, rules)
        
        results.append(
            {
                "case_name": case_name,
                "input_scene": scene,
                "decision": _model_dump(decision_obj),
            }
        )

    output = {"results": results}
    indent = 2 if args.pretty else None
    print(json.dumps(output, ensure_ascii=False, indent=indent))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())