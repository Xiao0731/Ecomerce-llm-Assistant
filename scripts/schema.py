from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# ==========================================
# 枚举值统一定义 (严格约束大模型输出边界)
# ==========================================

# "EMOTIONAL_COMPLAINT"已删：视为跨场景触发的风控状态，而不是业务主场景
# 通过规则环境的R001引擎驱动，直接在前置风控转人工
# "LOGISTICS_QUERY"新增：原本的"LOGISTICS_EXCEPTION"偏异常处理，但实际场景大多是非异常物流咨询
# 例如：预计到货时间、是否到站、能否自提、能否提前送等等
SceneEnum = Literal[
    "RETURN_REFUND", "QUALITY_ISSUE", "WRONG_OR_MISSING_ITEM", 
    "LOGISTICS_EXCEPTION", "LOGISTICS_QUERY", "ORDER_CANCEL_MODIFY", "PROMOTION_COUPON", 
    "PRICE_PROTECTION", "REFUND_PROGRESS", "INVOICE_REQUEST"
]

OrderStatusEnum = Literal[
    "pending_payment", "paid_unshipped", "packed", "shipped", 
    "delivered", "signed", "completed", "cancelled", 
    "refund_requested", "refund_processing", "refund_completed"
]

GoodsTypeEnum = Literal[
    "general", "food", "fresh", "customized", "virtual", "service", "medicine"
]

IssueTypeEnum = Literal[
    "damaged", "malfunction", "expired", "wrong_item", 
    "missing_item", "package_broken", "other"
]

DeliveryTypeEnum = Literal[
    "standard", "same_day", "instant"
]

LogisticsStatusEnum = Literal[
    "waiting_pickup", "in_transit", "delayed", "signed", "lost", "exception"
]

DecisionEnum = Literal[
    "RETURN", "REFUND", "RESEND", "COMPENSATE", "REJECT", 
    "CLARIFY", "PROCESSING", "ESCALATE", "GUIDE"
]

NextActionEnum = Literal[
    "START_RETURN_FLOW", "START_REFUND_FLOW", "START_RESEND_FLOW", 
    "ISSUE_COMPENSATION", "EXPLAIN_POLICY", "ASK_FOR_ORDER_INFO", 
    "ASK_FOR_PHOTOS", "EXPLAIN_COUPON_RULE", "EXPLAIN_REFUND_PROGRESS", 
    "GUIDE_CANCEL_SELF_SERVICE", "GUIDE_INVOICE_APPLICATION", "TRANSFER_TO_HUMAN",
    "EXPLAIN_DELIVERY_ESTIMATE", "EXPLAIN_STATION_STATUS", "EXPLAIN_SELF_PICKUP_POLICY", "EXPLAIN_DELIVERY_COORDINATION",
    "EXPLAIN_INVOICE_STATUS", "GUIDE_INVOICE_REISSUE", "EXPLAIN_INVOICE_TYPE_CHANGE", "EXPLAIN_INVOICE_MAILING_CHANGE",
    "EXPLAIN_PRICE_PROTECT_AMOUNT", "EXPLAIN_PRICE_PROTECT_RETURN_CHANNEL",
]

# ==========================================
# 1. 结构化输入 (对应 structured_input)
# ==========================================
class StructuredInput(BaseModel):
    # 风控与用户状态
    emotion: Literal["neutral", "anxious", "angry", "urgent"] = "neutral"
    abusive_language: bool = False
    repeated_contact_count: int = 0
    manual_review_required: bool = False

    # 订单与商品基础状态 (使用 Optional 结合 Literal)
    order_status: Optional[OrderStatusEnum] = None
    goods_type: Optional[GoodsTypeEnum] = None

    # 状态标志
    is_opened: Optional[bool] = None
    is_used: Optional[bool] = None
    is_redeemed: Optional[bool] = None
    signed_days: Optional[int] = None

    # 问题与证据
    quality_issue: Optional[bool] = None
    issue_type: Optional[IssueTypeEnum] = None
    evidence_provided: Optional[bool] = None
    wrong_item: Optional[bool] = None
    missing_item: Optional[bool] = None
    damaged_package: Optional[bool] = None

    # 物流状态
    delivery_type: Optional[DeliveryTypeEnum] = None
    delivery_delay_minutes: Optional[int] = None
    logistics_status: Optional[LogisticsStatusEnum] = None

    # 营销状态
    coupon_stackable: Optional[bool] = None
    price_protect_eligible: Optional[bool] = None

    # 物流咨询
    logistics_query_type: Optional[
        Literal[
            "ETA_QUERY",
            "STATION_STATUS_QUERY",
            "SELF_PICKUP_QUERY",
            "EARLY_DELIVERY_QUERY",
            "CONTACT_COURIER_QUERY"
        ]
    ] = None

    # 发票咨询
    invoice_query_type: Optional[
        Literal[
            "INVOICE_STATUS_QUERY",
            "INVOICE_REISSUE",
            "INVOICE_TYPE_CHANGE",
            "INVOICE_MAILING_CHANGE",
            "SHOPPING_LIST_QUERY"
        ]
    ] = None

    # 价保
    price_query_type: Optional[
        Literal[
            "PRICE_PROTECTION_ELIGIBILITY",
            "PRICE_PROTECTION_AMOUNT_QUERY",
            "PRICE_PROTECTION_RETURN_CHANNEL"
        ]
    ] = None

# ==========================================
# 2. 真值输出 (对应 oracle)
# ==========================================

class OracleOutput(BaseModel):
    matched_rule_id: str = "UNKNOWN"         # 加上默认值
    decision: DecisionEnum
    reason_code: str  
    next_action: NextActionEnum
    need_clarification: bool = False         # 加上默认值
    clarify_questions: List[str] = Field(default_factory=list) # 加上默认空数组
    should_apologize: bool = False           # 加上默认值
    escalate_to_human: bool = False          # 加上默认值，修复报错！

# ==========================================
# 3. 元数据 (对应 meta)
# ==========================================
# ==========================================
# 3. 元数据 (对应 meta)
# ==========================================

# 把 RoutingLabel 挪到类的外面，作为全局枚举类型
# 排除三类场景：训练SFT时过滤，DPO阶段可用
# 账号信息：账号找回、验证码、身份证、银行卡
# 商品信息：商品功能、参数、真伪、尺码、型号
# 商家端咨询
RoutingLabelEnum = Literal[
    "IN_SCOPE",
    "OUT_OF_SCOPE_ACCOUNT_SECURITY",
    "OUT_OF_SCOPE_PRODUCT_INFO",
    "OUT_OF_SCOPE_MERCHANT_SIDE"
]

class MetaInfo(BaseModel): # BaseModel内部所有变量默认当做数据字段
    origin: Literal["human_seed", "rule_synth", "llm_rewrite", "llm_expand", "manual_fix"]
    source_ref: str
    noise_tags: List[str] = Field(default_factory=list)
    style_tags: List[str] = Field(default_factory=list)
    
    # 在类里面把它声明为一个正常的字段，并给个默认值 "IN_SCOPE"
    routing_label: RoutingLabelEnum = "IN_SCOPE"


# ==========================================
# 4. 全局数据结构
# ==========================================

# 定义单轮对话的格式
class DialogTurn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

# CanonicalCase 
class CanonicalCase(BaseModel):
    case_id: str
    split: Literal["train", "dev", "test"] = "train"
    scene: SceneEnum
    user_query: str
    dialog_history: List[DialogTurn] = Field(default_factory=list) # 默认是空列表
    structured_input: StructuredInput
    oracle: OracleOutput
    meta: MetaInfo