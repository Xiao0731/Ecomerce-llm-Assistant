from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass
class RuleMatch:
    rule_id: str
    priority: int
    output: dict[str, Any]


SAMPLE_CASES: dict[str, dict[str, Any]] = {
    "refund_processing": {
        "scene": "REFUND_PROGRESS",
        "order_status": "refund_processing",
    },
    "return_allowed": {
        "scene": "RETURN_REFUND",
        "goods_type": "general",
        "signed_days": 5,
        "is_opened": False,
        "is_used": False,
        "quality_issue": False,
    },
    "customized_reject": {
        "scene": "RETURN_REFUND",
        "goods_type": "customized",
        "quality_issue": False,
    },
    "quality_need_evidence": {
        "scene": "QUALITY_ISSUE",
        "quality_issue": True,
        "evidence_provided": False,
    },
    "wrong_item_resend": {
        "scene": "WRONG_OR_MISSING_ITEM",
        "wrong_item": True,
        "evidence_provided": True,
    },
    "delay_compensate": {
        "scene": "LOGISTICS_EXCEPTION",
        "delivery_type": "instant",
        "delivery_delay_minutes": 45,
    },
    "emotion_escalate": {
        "manual_review_required": False,
        "abusive_language": True,
        "repeated_contact_count": 1,
        "emotion": "angry",
    },
}


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

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
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

    raise FileNotFoundError(
        "No rule file found. Tried: "
        + ", ".join(str(path) for path in DEFAULT_RULE_CANDIDATES)
    )


def get_field_value(payload: dict[str, Any], field: str) -> Any:
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
    if not condition:  # 【新增】处理 when: {} 的情况，表示无条件匹配
        return True
    if "all" in condition:
        return all(match_condition(item, payload) for item in condition["all"])
    if "any" in condition:
        return any(match_condition(item, payload) for item in condition["any"])
    return match_leaf(condition, payload)


def evaluate_rules(rules: list[dict[str, Any]], payload: dict[str, Any]) -> list[RuleMatch]:
    matches: list[RuleMatch] = []
    for rule in sorted(rules, key=lambda item: item.get("priority", 0), reverse=True):
        when = rule.get("when", {})
        if match_condition(when, payload):
            matches.append(
                RuleMatch(
                    rule_id=str(rule.get("id", "UNKNOWN")),
                    priority=int(rule.get("priority", 0)),
                    output=rule.get("then", {}),
                )
            )
    return matches


def load_cases(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    if args.case_json:
        return [("inline_case", json.loads(args.case_json))]

    if args.kv:
        case: dict[str, Any] = {}
        for item in args.kv:
            if "=" not in item:
                raise ValueError(f"Invalid --kv item: {item}. Expected key=value.")
            key, raw_value = item.split("=", 1)
            case[key] = parse_cli_scalar(raw_value)
        return [("kv_case", case)]

    if args.case_file:
        case_path = Path(args.case_file)
        if not case_path.is_absolute():
            case_path = BASE_DIR / case_path
        content = json.loads(case_path.read_text(encoding="utf-8"))
        if isinstance(content, list):
            # 如果 item 里有 "input" 键，就把里面的内容拿出来；顺便把 name 也拿出来
            return [(item.get("name", f"case_{index + 1}"), item.get("input", item)) for index, item in enumerate(content)]
        if isinstance(content, dict):
            if all(isinstance(value, dict) for value in content.values()):
                return [(name, value) for name, value in content.items()]
            return [("file_case", content)]
        raise ValueError("Case file must contain a JSON object or array.")

    if args.sample:
        return [(name, SAMPLE_CASES[name]) for name in args.sample]

    return list(SAMPLE_CASES.items())


def format_result(
    case_name: str,
    case_payload: dict[str, Any],
    matches: list[RuleMatch],
    reason_codes: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_name": case_name,
        "input": case_payload,
        "matched": bool(matches),
        "matched_rules": [
            {
                "rule_id": match.rule_id,
                "priority": match.priority,
                "decision": match.output.get("decision"),
                "reason_code": match.output.get("reason_code"),
                "next_action": match.output.get("next_action"),
            }
            for match in matches
        ],
        "final": matches[0].output if matches else None,
    }

    if matches:
        reason_code = matches[0].output.get("reason_code")
        if reason_code and reason_code in reason_codes:
            result["reason_detail"] = reason_codes[reason_code]

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple policy rule engine for validating YAML policy rules."
    )
    parser.add_argument(
        "--rule-file",
        help="Path to the rule YAML file. Defaults to policies/policy_rules_v1.yaml, then policies/policy_rules.yaml.",
    )
    parser.add_argument(
        "--reason-codes",
        default=str(DEFAULT_REASON_CODES),
        help="Path to the reason code YAML file.",
    )
    parser.add_argument(
        "--case-file",
        help="Path to a JSON file containing one case object, a named-case object, or a list of case objects.",
    )
    parser.add_argument(
        "--case-json",
        help='Inline JSON case, for example: --case-json \'{"scene":"REFUND_PROGRESS","order_status":"refund_processing"}\'',
    )
    parser.add_argument(
        "--kv",
        nargs="*",
        help="Key-value input for quick testing, for example: --kv scene=RETURN_REFUND signed_days=5 is_opened=false",
    )
    parser.add_argument(
        "--sample",
        action="append",
        choices=sorted(SAMPLE_CASES.keys()),
        help="Run one or more built-in sample cases. If omitted, all sample cases are executed.",
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="List available built-in sample cases and exit.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_samples:
        for sample_name in sorted(SAMPLE_CASES):
            print(sample_name)
        return 0

    rule_path = resolve_rule_file(args.rule_file)
    if not rule_path.exists():
        raise FileNotFoundError(f"Rule file not found: {rule_path}")

    reason_path = Path(args.reason_codes)
    if not reason_path.is_absolute():
        reason_path = BASE_DIR / reason_path

    rules_config = load_yaml(rule_path)
    rules = rules_config.get("rules", [])
    reason_codes = load_yaml(reason_path) if reason_path.exists() else {}

    results = []
    for case_name, case_payload in load_cases(args):
        matches = evaluate_rules(rules, case_payload)
        results.append(format_result(case_name, case_payload, matches, reason_codes))

    output = {
        "rule_file": str(rule_path),
        "rule_count": len(rules),
        "results": results,
    }
    indent = 2 if args.pretty else None
    print(json.dumps(output, ensure_ascii=False, indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
