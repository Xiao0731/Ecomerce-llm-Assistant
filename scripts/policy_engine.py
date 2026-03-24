from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RULE_CANDIDATES = [
    BASE_DIR / "policies" / "policy_rules_v1.yaml",
    BASE_DIR / "policies" / "policy_rules.yaml",
]
DEFAULT_REASON_CODES = BASE_DIR / "policies" / "reason_codes.yaml"

# ==========================================
# 1. Pydantic 模型定义 (对齐 Schema 和 指引 2.1)
# ==========================================

class StructuredInput(BaseModel):
    # 风控与用户状态 (这些可以有默认值，因为没传就代表正常)
    emotion: Literal["neutral", "anxious", "angry", "urgent"] = "neutral"
    abusive_language: bool = False
    repeated_contact_count: int = 0
    manual_review_required: bool = False

    # 订单与商品基础状态
    order_status: Optional[str] = None
    goods_type: Optional[str] = None  # 改为 None
    
    # 状态标志 (必须为 None，区分“没说”和“说了没有”)
    is_opened: Optional[bool] = None  # 改为 None
    is_used: Optional[bool] = None    # 改为 None
    is_redeemed: Optional[bool] = None
    signed_days: Optional[int] = None

    # 问题与证据
    quality_issue: Optional[bool] = None # 改为 None
    issue_type: Optional[str] = None
    evidence_provided: Optional[bool] = None # 改为 None
    wrong_item: Optional[bool] = None
    missing_item: Optional[bool] = None
    damaged_package: Optional[bool] = None

    # 物流状态
    delivery_type: Optional[str] = None
    delivery_delay_minutes: Optional[int] = None
    logistics_status: Optional[str] = None

    # 营销状态
    coupon_stackable: Optional[bool] = None
    price_protect_eligible: Optional[bool] = None


class PolicyDecision(BaseModel):
    decision: str
    reason_code: str
    next_action: str
    need_clarification: bool = False
    clarify_questions: List[str] = Field(default_factory=list)
    should_apologize: bool = False
    escalate_to_human: bool = False


@dataclass
class RuleMatch:
    rule_id: str
    priority: int
    output: dict[str, Any]


# ==========================================
# 2. 槽位检查与追问引擎 (实现指引 2.2)
# ==========================================

SCENE_REQUIREMENTS = {
    "RETURN_REFUND": {
        "required_slots": ["goods_type", "is_opened", "is_used", "signed_days", "quality_issue"],
        "questions": {
            "signed_days": "请问您签收至今大概几天了？",
            "is_opened": "请问商品目前是否已经拆封或使用？",
            "is_used": "请问商品是否已经使用过？",
            "quality_issue": "请问是单纯不想要了，还是商品本身存在质量问题？"
        }
    },
    "QUALITY_ISSUE": {
        "required_slots": ["issue_type", "signed_days", "evidence_provided"],
        "questions": {
            "issue_type": "请问具体是什么问题，例如破损、无法使用，还是少件？",
            "evidence_provided": "方便提供一下商品问题的图片或视频吗？",
            "signed_days": "请问您是签收当天发现的吗，还是签收后几天发现的？"
        }
    },
    "WRONG_OR_MISSING_ITEM": {
        "required_slots": ["wrong_item", "missing_item", "evidence_provided"],
        "questions": {
            "wrong_item": "请问是收到错商品，还是少发了商品/配件？",
            "missing_item": "请问是收到错商品，还是少发了商品/配件？",
            "evidence_provided": "方便拍一下外包装和实收商品给我吗？"
        }
    },
    "LOGISTICS_EXCEPTION": {
        "required_slots": ["logistics_status", "delivery_delay_minutes", "damaged_package"],
        "questions": {
            "logistics_status": "请问物流当前显示是什么状态？",
            "delivery_delay_minutes": "大概比预计送达晚了多久？",
            "damaged_package": "包裹外包装是否有破损？"
        }
    },
    "ORDER_CANCEL_MODIFY": {
        "required_slots": ["order_status"],
        "questions": {
            "order_status": "请问订单目前是待发货、已发货，还是已签收？"
        }
    },
    "PROMOTION_COUPON": {
        "required_slots": ["coupon_stackable"],
        "questions": {
            "coupon_stackable": "页面是否提示不可叠加或不满足使用门槛？"
        }
    },
    "PRICE_PROTECTION": {
        "required_slots": ["price_protect_eligible"],
        "questions": {
            "price_protect_eligible": "请问您下单后是否在价保期内发现降价，且页面显示支持价保？"
        }
    },
    "REFUND_PROGRESS": {
        "required_slots": ["order_status"],
        "questions": {
            "order_status": "请问您目前的订单状态是退款处理中吗？"
        }
    }
}


def check_missing_slots(scene: str, input_data: StructuredInput) -> list[str]:
    """检查缺失的槽位并返回对应的问题列表"""
    if scene not in SCENE_REQUIREMENTS:
        return []
    
    reqs = SCENE_REQUIREMENTS[scene]
    missing_questions = []
    
    for slot in reqs["required_slots"]:
        val = getattr(input_data, slot)
        # 判断槽位是否缺失 (注意: bool 类型的 False 不是缺失, None 才是缺失)
        if val is None:
             q = reqs["questions"].get(slot)
             if q and q not in missing_questions: # 防止重复问题
                 missing_questions.append(q)
                 
    return missing_questions

# ==========================================
# 3. YAML 解析与规则评估 (保留之前的优秀实现，略作调整适应新结构)
# ==========================================

# ... (此处保留 load_yaml, parse_simple_yaml, split_key_value, parse_scalar, parse_cli_scalar, resolve_rule_file 完全不变)

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

    index = 0

    def parse_block(expected_indent: int) -> Any:
        nonlocal index
        if index >= len(lines):
            return {}
        indent, content = lines[index]
        if indent < expected_indent:
            return {}
        if content.startswith("- "):
            return parse_list(expected_indent)
        return parse_dict(expected_indent)

    def parse_list(expected_indent: int) -> list[Any]:
        nonlocal index
        items: list[Any] = []
        while index < len(lines):
            indent, content = lines[index]
            if indent < expected_indent:
                break
            if indent != expected_indent or not content.startswith("- "):
                break
            item_content = content[2:].strip()
            index += 1
            if not item_content:
                items.append(parse_block(expected_indent + 2))
                continue
            if ":" in item_content:
                key, raw_value = split_key_value(item_content)
                item: dict[str, Any] = {}
                if raw_value is None:
                    item[key] = parse_block(expected_indent + 4)
                else:
                    item[key] = parse_scalar(raw_value)
                while index < len(lines):
                    next_indent, next_content = lines[index]
                    if next_indent < expected_indent + 2:
                        break
                    if next_indent == expected_indent and next_content.startswith("- "):
                        break
                    if next_indent != expected_indent + 2 or next_content.startswith("- "):
                        break
                    child_key, child_raw_value = split_key_value(next_content)
                    index += 1
                    if child_raw_value is None:
                        item[child_key] = parse_block(expected_indent + 4)
                    else:
                        item[child_key] = parse_scalar(child_raw_value)
                items.append(item)
                continue
            items.append(parse_scalar(item_content))
        return items

    def parse_dict(expected_indent: int) -> dict[str, Any]:
        nonlocal index
        mapping: dict[str, Any] = {}
        while index < len(lines):
            indent, content = lines[index]
            if indent < expected_indent:
                break
            if indent != expected_indent or content.startswith("- "):
                break
            key, raw_value = split_key_value(content)
            index += 1
            if raw_value is None:
                mapping[key] = parse_block(expected_indent + 2)
            else:
                mapping[key] = parse_scalar(raw_value)
        return mapping
    return parse_block(0)

def split_key_value(text: str) -> tuple[str, str | None]:
    key, _, value = text.partition(":")
    key = key.strip()
    value = value.strip()
    if value == "":
        return key, None
    return key, value

def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value

def parse_cli_scalar(value: str) -> Any:
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return parse_scalar(value)
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
    # 支持点号语法，例如 structured_input.goods_type
    current: Any = payload
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def match_leaf(condition: dict[str, Any], payload: dict[str, Any]) -> bool:
    field = condition.get("field")
    if not field:
        return False

    actual = get_field_value(payload, field)
    operators = {
        "eq": lambda value: actual == value,
        "in": lambda value: actual in value if value is not None else False,
        "gt": lambda value: actual is not None and actual > value,
        "gte": lambda value: actual is not None and actual >= value,
        "lt": lambda value: actual is not None and actual < value,
        "lte": lambda value: actual is not None and actual <= value,
        "exists": lambda value: (actual is not None) is bool(value),
    }

    for operator, expected in condition.items():
        if operator == "field":
            continue
        checker = operators.get(operator)
        if checker is None:
            raise ValueError(f"Unsupported operator: {operator}")
        if not checker(expected):
            return False
    return True


def match_condition(condition: dict[str, Any], payload: dict[str, Any]) -> bool:
    if not condition:
        return True
    if "all" in condition:
        return all(match_condition(item, payload) for item in condition["all"])
    if "any" in condition:
        return any(match_condition(item, payload) for item in condition["any"])
    return match_leaf(condition, payload)


# ==========================================
# 4. 核心调度与执行
# ==========================================

def evaluate_case(scene: str, input_dict: dict, rules: list) -> PolicyDecision:
    """整合 Pydantic 校验、缺槽检查和 YAML 规则评估的核心函数"""
    
    # 1. Pydantic 强类型转换与填充默认值
    try:
        structured_data = StructuredInput(**input_dict)
    except Exception as e:
        # 如果类型严重错误，兜底报错
        return PolicyDecision(
            decision="ESCALATE", reason_code="SYSTEM_ERROR", 
            next_action="TRANSFER_TO_HUMAN", escalate_to_human=True
        )

    # 2. 高优先级风控检查 (指引的 Step 1)
    if structured_data.manual_review_required or structured_data.abusive_language or (structured_data.repeated_contact_count >= 3 and structured_data.emotion in ["angry", "urgent"]):
         return PolicyDecision(
            decision="ESCALATE", reason_code="EMOTION_ESCALATION", 
            next_action="TRANSFER_TO_HUMAN", should_apologize=True, escalate_to_human=True
        )

    # 3. 缺槽检查 (指引的 Step 3)
    missing_qs = check_missing_slots(scene, structured_data)
    if missing_qs:
        return PolicyDecision(
            decision="CLARIFY",
            reason_code="INSUFFICIENT_INFORMATION",
            next_action="ASK_FOR_ORDER_INFO",
            need_clarification=True,
            clarify_questions=missing_qs,
            should_apologize=True
        )

    # 4. 送入 YAML 引擎进行逻辑裁决
    # 【修复 1】使用 model_dump() 替代废弃的 dict()
    eval_payload = structured_data.model_dump() 
    eval_payload["scene"] = scene
    
    matches: list[RuleMatch] = []
    for rule in sorted(rules, key=lambda item: item.get("priority", 0), reverse=True):
        when = rule.get("when", {})
        if match_condition(when, eval_payload):
            matches.append(RuleMatch(
                rule_id=str(rule.get("id", "UNKNOWN")),
                priority=int(rule.get("priority", 0)),
                output=rule.get("then", {})
            ))
            break # 命中最高优先级即退出

    if matches:
        out = matches[0].output
        return PolicyDecision(
            decision=out.get("decision", "CLARIFY"),
            reason_code=out.get("reason_code", "UNKNOWN"),
            next_action=out.get("next_action", "ASK_FOR_ORDER_INFO"),
            should_apologize=out.get("should_apologize", False),
            escalate_to_human=out.get("escalate_to_human", False)
        )
        
    # 5. 绝对兜底
    return PolicyDecision(
        decision="CLARIFY", reason_code="INSUFFICIENT_INFORMATION", next_action="ASK_FOR_ORDER_INFO"
    )

def load_cases(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    cases = []
    if args.case_file:
        case_path = Path(args.case_file)
        if not case_path.is_absolute():
            case_path = BASE_DIR / case_path
        content = json.loads(case_path.read_text(encoding="utf-8"))
        if isinstance(content, list):
            cases.extend([(item.get("name", f"case_{index + 1}"), item.get("input", item)) for index, item in enumerate(content)])
        elif isinstance(content, dict):
             cases.append(("file_case", content))
             
    # 【修复 2】把 --kv 解析逻辑加回来
    if getattr(args, "kv", None):
        kv_case = {}
        for item in args.kv:
            if "=" in item:
                k, v = item.split("=", 1)
                if v.lower() == "true": v = True
                elif v.lower() == "false": v = False
                elif v.isdigit(): v = int(v)
                kv_case[k] = v
        cases.append(("kv_case", kv_case))
        
    return cases

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Policy Engine with Slot Checking")
    parser.add_argument("--rule-file", help="Path to the rule YAML file.")
    parser.add_argument("--case-file", help="Path to a JSON file containing cases.")
    parser.add_argument("--kv", nargs="*", help="Key-value input for quick testing.") # 加回参数
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
        
        results.append({
            "case_name": case_name,
            "input_scene": scene,
            # 【修复 1】使用 model_dump() 替代 dict()
            "decision": decision_obj.model_dump() 
        })

    output = {"results": results}
    indent = 2 if args.pretty else None
    print(json.dumps(output, ensure_ascii=False, indent=indent))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())