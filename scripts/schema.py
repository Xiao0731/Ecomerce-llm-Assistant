from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# ==========================================
# 枚举值统一定义 (严格约束大模型输出边界)
# ==========================================

SceneEnum = Literal[
    "RETURN_REFUND", "QUALITY_ISSUE", "WRONG_OR_MISSING_ITEM", 
    "LOGISTICS_EXCEPTION", "ORDER_CANCEL_MODIFY", "PROMOTION_COUPON", 
    "PRICE_PROTECTION", "REFUND_PROGRESS", "INVOICE_REQUEST", "EMOTIONAL_COMPLAINT"
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
    "GUIDE_CANCEL_SELF_SERVICE", "GUIDE_INVOICE_APPLICATION", "TRANSFER_TO_HUMAN"
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

# ==========================================
# 2. 真值输出 (对应 oracle)
# ==========================================
class OracleOutput(BaseModel):
    matched_rule_id: str
    decision: DecisionEnum
    reason_code: str  # 原因码较多且可能扩展，暂用 str，也可后续写死
    next_action: NextActionEnum
    need_clarification: bool
    clarify_questions: List[str]
    should_apologize: bool
    escalate_to_human: bool

# ==========================================
# 3. 元数据 (对应 meta)
# ==========================================
class MetaInfo(BaseModel):
    origin: Literal["human_seed", "rule_synth", "llm_rewrite", "llm_expand", "manual_fix"]
    source_ref: str
    noise_tags: List[str] = Field(default_factory=list)
    style_tags: List[str] = Field(default_factory=list)

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
    
    # assistant_reply: str = ""