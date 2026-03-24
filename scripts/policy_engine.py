from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field # 数据校验模型

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RULE_CANDIDATES = [
    BASE_DIR / "policies" / "policy_rules_v1.yaml",
    BASE_DIR / "policies" / "policy_rules.yaml",
]

# ==========================================
# 1. Pydantic 模型定义 (定义输入和输出的“形状”)
# ==========================================

# 继承自 BaseModel，这个类用来严格限制和校验大模型传来的 JSON 数据
class StructuredInput(BaseModel):
    # --- 风控与用户状态 ---
    # Literal 限制了 emotion 只能是这四个字符串之一，默认是 "neutral"
    emotion: Literal["neutral", "anxious", "angry", "urgent"] = "neutral"
    abusive_language: bool = False  # 是否有辱骂
    repeated_contact_count: int = 0 # 进线次数
    manual_review_required: bool = False # 是否必须人工审核

    # --- 订单与商品基础状态 ---
    order_status: Optional[str] = None # Optional 表示这个字段可以不传，给后期追问留空间
    goods_type: Optional[str] = None   

    # --- 状态标志 ---
    is_opened: Optional[bool] = None   # 是否拆封
    is_used: Optional[bool] = None     # 是否使用
    is_redeemed: Optional[bool] = None # 虚拟物品是否兑换
    signed_days: Optional[int] = None  # 签收天数

    # --- 问题与证据 ---
    quality_issue: Optional[bool] = None # 是否有质量问题
    issue_type: Optional[str] = None     # 问题类型
    evidence_provided: Optional[bool] = None # 是否提供证据
    wrong_item: Optional[bool] = None    # 是否错发
    missing_item: Optional[bool] = None  # 是否漏发
    damaged_package: Optional[bool] = None # 包裹是否破损

    # --- 物流状态 ---
    delivery_type: Optional[str] = None  # 配送类型
    delivery_delay_minutes: Optional[int] = None # 延误分钟数
    logistics_status: Optional[str] = None # 物流状态

    # --- 营销状态 ---
    coupon_stackable: Optional[bool] = None # 优惠券能否叠加
    price_protect_eligible: Optional[bool] = None # 是否符合价保条件

# 这个类用来定义我们引擎最终输出给大模型的指令格式
class PolicyDecision(BaseModel):
    matched_rule_id: Optional[str] = None # 命中的YAML 规则
    decision: str                         # 决策
    reason_code: str                      # 原因代码
    next_action: str                      # 下一步动作
    need_clarification: bool = False      # 是否需要追问用户
    clarify_questions: List[str] = Field(default_factory=list) # 具体的追问话术列表
    should_apologize: bool = False        # 是否需要大模型道歉
    escalate_to_human: bool = False       # 是否转人工


# 用于在匹配 YAML 规则时临时存储匹配结果的轻量级类
@dataclass
class RuleMatch:
    rule_id: str
    priority: int
    output: dict[str, Any]

# ==========================================
# 2. 槽位检查与追问引擎 
# 该部分代码决定什么时候反问用户，如何反问
# ==========================================

# 追问话术列表
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

# 兼容 Pydantic V1 和 V2 版本的导出字典方法
def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

# 工具函数：遍历指定的必填字段列表，看看输入数据里哪个字段是 None
def _collect_missing_slots(input_data: StructuredInput, required_slots: list[str]) -> list[str]:
    return [slot for slot in required_slots if getattr(input_data, slot, None) is None]

# 工具函数：把缺失的字段名转换成自然语言的反问句
def _slots_to_questions(missing_slots: list[str]) -> list[str]:
    questions: list[str] = []
    for slot in missing_slots:
        question = SLOT_QUESTIONS.get(slot)
        if question and question not in questions:
            questions.append(question)
    return questions

# 核心路由函数：根据不同的客服场景，动态决定还需要收集什么信息
def get_missing_slots_for_scene(scene: str, input_data: StructuredInput) -> list[str]:
    # 退货退款场景
    if scene == "RETURN_REFUND":
        # 第一层：必须先知道商品类型和有没有质量问题
        base_missing = _collect_missing_slots(input_data, ["goods_type", "quality_issue"])
        if base_missing: # 否则，追问
            return base_missing

        goods_type = input_data.goods_type
        quality_issue = input_data.quality_issue

        # 第二层：根据商品类型和是否有质量问题，选择不同的决策
        if quality_issue is True:
            return [] # 有质量问题，走质量售后
        if goods_type == "customized":
            return [] # 定制商品直接拒退
        if goods_type in {"virtual", "service"}: # 虚拟商品只关心是否已核销
            return _collect_missing_slots(input_data, ["is_redeemed"])
        if goods_type in {"food", "fresh", "medicine"}: # 生鲜只关心是否拆封
            return _collect_missing_slots(input_data, ["is_opened"])
        if goods_type == "general":
            # 排除上述情况后，常规商品要求最严，必须知道签收天数、是否拆封、是否使用
            return _collect_missing_slots(input_data, ["signed_days", "is_opened", "is_used"])
        return []

    # 质量问题场景：只核心关注有没有提供证据
    if scene == "QUALITY_ISSUE":
        return _collect_missing_slots(input_data, ["evidence_provided"])

    # 错发漏发场景
    if scene == "WRONG_OR_MISSING_ITEM":
        if input_data.wrong_item is None and input_data.missing_item is None:
            return ["wrong_item"] # 用户没说错发还是漏发，追问是否发错
        return _collect_missing_slots(input_data, ["evidence_provided"]) # 追问证据

    # 物流异常场景
    if scene == "LOGISTICS_EXCEPTION":
        if input_data.logistics_status == "lost":
            return [] # 丢件了，直接走退款

        # 如果用户提到了配送方式或延误时间，说明意图是抱怨“慢”
        if input_data.delivery_type is not None or input_data.delivery_delay_minutes is not None:
            return _collect_missing_slots(input_data, ["delivery_type", "delivery_delay_minutes"])
        
        # 如果用户提到了破损或发了照片，说明意图是抱怨“坏了”
        if input_data.damaged_package is not None or input_data.evidence_provided is not None:
            return _collect_missing_slots(input_data, ["damaged_package", "evidence_provided"])

        # 什么都没提，问一句当前物流状态
        return _collect_missing_slots(input_data, ["logistics_status"])

    if scene == "ORDER_CANCEL_MODIFY": # 取消订单
        return _collect_missing_slots(input_data, ["order_status"])

    if scene == "PROMOTION_COUPON":
        return _collect_missing_slots(input_data, ["coupon_stackable"])

    if scene == "PRICE_PROTECTION": # 价保
        return _collect_missing_slots(input_data, ["price_protect_eligible"])

    if scene == "REFUND_PROGRESS": # 退款
        return _collect_missing_slots(input_data, ["order_status"])

    return []

# 调用get_missing_slots_for_scene函数，获取缺失字段
# 再调用_slots_to_questions，根据缺失字段自动转为追问话术列表返回
def check_missing_slots(scene: str, input_data: StructuredInput) -> list[str]:
    missing_slots = get_missing_slots_for_scene(scene, input_data)
    return _slots_to_questions(missing_slots)


# ==========================================
# 3. YAML 解析器
# 这个模块为了摆脱对第三方包 PyYAML 的强制依赖而手写
# ==========================================
def load_yaml(path: Path) -> Any:
    # 如果安装了官方 yaml 库就用官方的，否则用我们下面手写的
    if yaml is not None:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return parse_simple_yaml(path.read_text(encoding="utf-8"))

# 手写 YAML 解析器的核心逻辑（通过缩进来判断层级）
def parse_simple_yaml(text: str) -> Any:
    lines: list[tuple[int, str]] = []
    # 预处理：去掉空行和注释，并计算每行的缩进量
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        stripped = raw_line.lstrip()
        if stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(stripped)
        lines.append((indent, stripped))

    # 递归解析块
    def parse_block(index: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index
        indent, content = lines[index]
        if content.startswith("- "):
            return parse_list(index, indent) # 解析列表
        return parse_dict(index, indent, {}) # 解析字典

    # 解析字典类型
    def parse_dict(index: int, indent: int, seed: dict[str, Any]) -> tuple[dict[str, Any], int]:
        mapping = dict(seed)
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent: # 缩进变小，说明当前字典结束了
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indentation near: {content}")
            if content.startswith("- "): # 遇到减号，说明进入列表了，跳出字典
                break

            key, raw_value = split_key_value(content) # 分离键和值
            index += 1
            if raw_value is None: # 如果没有值，说明嵌套了一个子结构
                if index < len(lines) and lines[index][0] > current_indent:
                    value, index = parse_block(index) # 递归解析子结构
                    mapping[key] = value
                else:
                    mapping[key] = {}
            else: # 如果有值，转换成 Python 的类型
                mapping[key] = parse_scalar(raw_value)
        return mapping, index

    # 解析列表类型
    def parse_list(index: int, indent: int) -> tuple[list[Any], int]:
        items: list[Any] = []
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent != indent or not content.startswith("- "):
                break # 不再是同一层级的列表项，结束解析

            item_text = content[2:].strip() # 去掉 "- "
            index += 1

            if not item_text: # 如果减号后面没东西，说明是嵌套对象
                value, index = parse_block(index)
                items.append(value)
                continue

            if ":" in item_text: # 列表里嵌套了字典
                key, raw_value = split_key_value(item_text)
                item: dict[str, Any] = {}
                if raw_value is None: # 字典的值又是嵌套结构
                    if index < len(lines) and lines[index][0] > current_indent:
                        value, index = parse_block(index)
                        item[key] = value
                    else:
                        item[key] = {}
                else: # 简单键值对
                    item[key] = parse_scalar(raw_value)

                # 继续读取同一列表项下的其他字典键值对
                while index < len(lines):
                    next_indent, next_content = lines[index]
                    if next_indent <= current_indent:
                        break # 不在同一层级了，跳出
                    if next_content.startswith("- "):
                        break # 是新列表项了，跳出
                    item, index = parse_dict(index, next_indent, item) # 递归解析

                items.append(item)
                continue

            # 简单的单元素列表，直接加进去
            items.append(parse_scalar(item_text))
        return items, index

    parsed, _ = parse_block(0) # 开始解析！
    return parsed

# 工具函数：把 "key: value" 这种字符串切分成两半
def split_key_value(text: str) -> tuple[str, str | None]:
    key, _, value = text.partition(":")
    key = key.strip()
    value = value.strip()
    if value == "":
        return key, None
    return key, value

# 核心类型转换器：把字符串变成 Python 原生对象 (非常重要！)
def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true": # 处理布尔
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}: # 处理空值
        return None
    if value.startswith("[") and value.endswith("]"): # 处理列表格式 [a, b, c]
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")): # 处理带引号的字符串
        return value[1:-1]
    try: # 尝试转成整数
        return int(value)
    except ValueError:
        pass
    try: # 尝试转成浮点数
        return float(value)
    except ValueError:
        pass
    return value # 其他一切当成纯字符串

# 处理命令行中 --kv 传进来的奇特参数格式
def parse_cli_scalar(value: str) -> Any:
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value) # 如果像 JSON 就用 JSON 解析
        except json.JSONDecodeError:
            return parse_scalar(value)
    return parse_scalar(value)

# 寻找规则文件路径，支持绝对路径、相对路径，或者回退到默认路径
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

# ==========================================
# 4. 规则匹配引擎 (核心判定逻辑)
# ==========================================

# 支持通过点号 "structured_input.goods_type" 来读取嵌套的字段
def get_field_value(payload: dict[str, Any], field: str) -> Any:
    current: Any = payload
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return None # 只要读不到就返回 None，防崩溃
        current = current[part]
    return current

# YAML 规则匹配逻辑：判断单个叶子节点（如 `eq`, `in`, `gt`）是否符合
def match_leaf(condition: dict[str, Any], payload: dict[str, Any]) -> bool:
    field = condition.get("field")
    if not field:
        return False # 连要比较的字段名都没有，直接判错

    actual = get_field_value(payload, field) # 拿到真实的值
    
    # 所有的操作符字典
    operators = {
        "eq": lambda value: actual == value, # 等于
        "ne": lambda value: actual != value, # 不等于
        "in": lambda value: actual in value if value is not None else False, # 包含于
        "not_in": lambda value: actual not in value if value is not None else True, # 不包含于
        "gt": lambda value: actual is not None and actual > value,   # 大于
        "gte": lambda value: actual is not None and actual >= value, # 大于等于
        "lt": lambda value: actual is not None and actual < value,   # 小于
        "lte": lambda value: actual is not None and actual <= value, # 小于等于
        "exists": lambda value: (actual is not None) is bool(value), # 是否存在该字段
    }

    # 遍历该条规则下的所有判断符，必须全满足才算通过（如果既有 in 又有 gt，就都要满足）
    for operator, expected in condition.items():
        if operator == "field": # 跳过 field 这个固定的名字字段
            continue
        checker = operators.get(operator)
        if checker is None: # 如果 YAML 里写了一个我们不支持的操作符，就抛出错误
            raise ValueError(f"Unsupported operator: {operator}")
        if not checker(expected):
            return False # 只要一个条件不满足，这片叶子就判定为失败
    return True

# YAML 规则匹配逻辑：递归判断所有的 `all` 和 `any` 和空条件
def match_condition(condition: dict[str, Any], payload: dict[str, Any]) -> bool:
    # 修复的兜底坑：如果条件是空字典（R999），无条件放行
    if not condition:
        return True
    # 递归匹配 `all`，也就是且，必须全部子条件满足才返回 True
    if "all" in condition:
        return all(match_condition(item, payload) for item in condition["all"])
    # 递归匹配 `any`，也就是或，只要有一个子条件满足就返回 True
    if "any" in condition:
        return any(match_condition(item, payload) for item in condition["any"])
    
    # 既不是全不是空也不是 any，那就交到底层去做叶子节点匹配
    return match_leaf(condition, payload)

# ==========================================
# 5. 系统总调度：串联前处理、强校验和最终判定
# ==========================================
# 输入场景和零散字典，经过重重安检，输出一份完整的系统决策

def evaluate_case(scene: str, input_dict: dict[str, Any], rules: list[dict[str, Any]]) -> PolicyDecision:
    # 步骤 1：Pydantic 强类型校验。防呆设计，把外界不规范的数据变成整齐的对象
    try:
        structured_data = StructuredInput(**input_dict)
    except Exception:
        # 如果类型错得离谱，直接跳出并转人工
        return PolicyDecision(
            decision="ESCALATE",
            reason_code="SYSTEM_ERROR",
            next_action="TRANSFER_TO_HUMAN",
            escalate_to_human=True,
        )
    # 步骤 2：高风险前置拦截，如果满足这三个条件任意一个，不走常规退款流程，直接转人工
    if (
        structured_data.manual_review_required
        or structured_data.abusive_language
        or (
            structured_data.repeated_contact_count >= 3
            and structured_data.emotion in ["angry", "urgent"]
        )
    ):
        return PolicyDecision(
            decision="ESCALATE",
            reason_code="EMOTION_ESCALATION",
            next_action="TRANSFER_TO_HUMAN",
            should_apologize=True,
            escalate_to_human=True,
        )
    # 步骤 3：缺槽拦截追问（调用核心路由）
    # 在扔进 YAML 前，先拦住问一下“你这订单信息全吗？”
    missing_qs = check_missing_slots(scene, structured_data)
    if missing_qs:
        return PolicyDecision(
            decision="CLARIFY",
            reason_code="INSUFFICIENT_INFORMATION",
            next_action="ASK_FOR_ORDER_INFO",
            need_clarification=True, # 需要模型进行追问
            clarify_questions=missing_qs, # 追问的话术列表
            should_apologize=True,
        )
    
    # 步骤 4：YAML 规则匹配
    # 把 Pydantic 对象变回字典，根据 YAML 去取值
    eval_payload = _model_dump(structured_data)
    eval_payload["scene"] = scene

    # 遍历按优先级从高到低排序好的规则
    for rule in sorted(rules, key=lambda item: item.get("priority", 0), reverse=True):
        when = rule.get("when", {}) # 匹配条件（走 match_condition -> match_leaf）
        if match_condition(when, eval_payload):
            out = rule.get("then", {}) # 命中
            return PolicyDecision(
                matched_rule_id=str(rule.get("id", "UNKNOWN")),
                decision=out.get("decision", "CLARIFY"),
                reason_code=out.get("reason_code", "UNKNOWN"),
                next_action=out.get("next_action", "ASK_FOR_ORDER_INFO"),
                should_apologize=out.get("should_apologize", False),
                escalate_to_human=out.get("escalate_to_human", False),
            )

    # 步骤 5：如果上面的步骤和 YAML 全都没命中，系统只能继续追问信息
    return PolicyDecision(
        decision="CLARIFY",
        reason_code="INSUFFICIENT_INFORMATION",
        next_action="ASK_FOR_ORDER_INFO",
    )

# ==========================================
# 6. 命令行入口配置
# 下面部分用于处理在命令行用文件或者快捷参数对规则环境进行测试的配置
# ==========================================

# 从命令行或 JSON 文件里提取测试用例
def load_cases(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = []

    # 如果传了 --case-file 测试文件
    if args.case_file:
        case_path = Path(args.case_file)
        if not case_path.is_absolute():
            case_path = BASE_DIR / case_path
        content = json.loads(case_path.read_text(encoding="utf-8"))
        if isinstance(content, list): # 处理数组型的测试用例
            cases.extend(
                [
                    (item.get("name", f"case_{index + 1}"), item.get("input", item))
                    for index, item in enumerate(content)
                ]
            )
        elif isinstance(content, dict): # 处理单对象测试用例
            cases.append(("file_case", content))

    # 如果传了 --kv 这个快捷参数，解析成目标字典格式
    if getattr(args, "kv", None):
        kv_case: dict[str, Any] = {}
        for item in args.kv:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            kv_case[key] = parse_cli_scalar(value) # 类型转换
        cases.append(("kv_case", kv_case))

    return cases

# 构造 CLI 命令解析器
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Policy Engine with route-aware slot checking")
    parser.add_argument("--rule-file", help="Path to the rule YAML file.")
    parser.add_argument("--case-file", help="Path to a JSON file containing cases.")
    parser.add_argument("--kv", nargs="*", help="Key-value input for quick testing.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser

# 程序执行的主入口！
def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # 找到 yaml 文件并在内存里把它构建成规则字典
    rule_path = resolve_rule_file(args.rule_file)
    rules_config = load_yaml(rule_path)
    rules = rules_config.get("rules", [])

    results = []
    # 从命令行参数拿到测试用例
    for case_name, case_payload in load_cases(args):
        scene = case_payload.get("scene", "UNKNOWN")
        # 把大权交给总调度函数
        decision_obj = evaluate_case(scene, case_payload, rules)
        
        # 将结果存在数组里，方便后续统一打印
        results.append(
            {
                "case_name": case_name,
                "input_scene": scene,
                "decision": _model_dump(decision_obj), # 把输出再转化成可以打在屏幕上的 JSON 格式
            }
        )

    output = {"results": results}
    indent = 2 if args.pretty else None
    # 打印：ensure_ascii=False 保证了中文字符能正常显示而不是乱码
    print(json.dumps(output, ensure_ascii=False, indent=indent))
    return 0

# Python 的标准起手式：如果是直接执行该文件，则进入 main 函数
if __name__ == "__main__":
    raise SystemExit(main())