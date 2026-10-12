from minilaya import load
import json, time

# state = """Customer email: The invoice INV-1044 was paid twice on March 3. Please refund the duplicate $240 charge to the card ending 4412. The subscription itself should stay active."""

# questions = {
#     "department": {
#         "type": "choice",
#         "instructions": "Which team should handle this email?",
#         "criteria": {
#             "billing": "payments, invoices, refunds",
#             "technical": "bugs, outages, access",
#             "retention": "cancel or save the subscription",
#         },
#     },
#     "refund": {
#         "type": "noul",
#         "instructions": "The customer is asking for a refund of a duplicate charge.",
#     },
#     "urgency": {
#         "type": "score",
#         "instructions": "How urgently should this be handled?",
#         "criteria": ["no rush", "this week", "today", "immediately"],
#     },
# }

# ------------------------------------------------------------
# 2. 定义状态（State）：一段中文工单文本
# ------------------------------------------------------------
state = json.dumps({
    "from": "用户@example.com",
    "subject": "重复扣费，要求退款",
    "body": "你好，我这个月被扣了两次费用，请尽快处理退款。如果今天不解决，我只能取消服务了。"
})

# ------------------------------------------------------------
# 3. 定义类型化问题（Questions）
# ------------------------------------------------------------
questions = {
    # choice：从预定义选项中选一个，返回每个选项的概率分布
    "department": {
        "type": "choice",
        "instructions": "这个工单应该分派给哪个部门处理？",
        "criteria": {
            "billing": "账单、付款、退款相关",
            "technical": "故障、宕机、系统错误相关",
            "sales": "定价、新合同相关",
            "other": "其他所有情况"
        }
    },
    # score：在有序量表上评分，返回各分值档位的概率分布
    "urgency": {
        "type": "score",
        "instructions": "这个请求的紧急程度如何？",
        "criteria": ["不紧急", "较紧急", "非常紧急，有截止日期或阻塞问题"]
    },
    # noul：回答一个是/否问题，返回 P(true) 的概率值（0-1）
    "refund_requested": {
        "type": "noul",
        "instructions": "用户是否明确要求退款？"
    },
    # noul：判断是否有流失风险
    "churn_risk": {
        "type": "noul",
        "instructions": "用户是否威胁要取消或离开？"
    }
}

agent = load("runs/bert-typed-decisions")

start_time = time.time()
result = agent.predict(state, questions)
print(result)
print(f"Time elapsed: {time.time() - start_time} seconds")