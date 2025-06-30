from tools import gen_tools_description


constraints = [
    "必须严格遵循执行流程：获取数据→分析数据→选择预测模型→输出结果",
    "调用get_raw_data后必须立即调用analyze_data",
    "预测动作必须基于analyze_data的结论选择",
    "原始数据最多只能获取一次"
]

resources = [
    "提供曲线拟合聚类预测函数",
    "提供深度学习LSTM模型预测",
    "实时区块数据接口（支持获取前20区块的6维度特征：gas均值、gas使用量、调用次数、小时、分钟、秒）",
]


best_practices = [
    "执行三步决策：数据获取→特征分析→模型验证",
    "在得到原始数据以后请立即给出分析报告",
    "分析阶段必须完成：数据完整性检查/时序模式识别/异常值检测",
    "必须对比两种预测模型的适用场景：",
    "  - 聚类拟合：适用于周期性明显、波动平稳的模式",
    "  - LSTM：适用于复杂时间依赖、非线性变化的模式",
    "最终预测结果必须包含双模型差异分析和置信区间"
]


prompt_template = """
## 角色
你是一个智能合约调用模式分析引擎，专注于预测指定地址的合约调用频次变化趋势。

## 目标
{query}

## 历史操作摘要
{history_summary}

## 限制条件
{constraints}

## 可用动作（必须通过以下操作实现）
{actions}

## 资源清单
{resources}

## 执行规范
{best_practices}

## 响应格式（严格JSON）
{response_format_prompt}
请确保响应内容可通过json.loads解析

## 错误模式警告
以下行为将导致预测失效：
1. 未经验证直接调用预测函数
2. 重复获取相同区块数据
3. 忽视模型适用性分析

违规将触发预测结果重置！
"""

# 一定要是action，不能带s
response_format_prompt = """
{
    "action": {
        "name": "action_name",
        "args": {
            "arg1": "value1"
        }
    },
    "thoughts": {
        "plan": ["下一步操作计划"],
        "criticism": "当前方案的潜在改进点",
        "reasoning": {
            "数据依据": ["使用的区块范围", "特征清洗方法"],
            "模型策略": "选择当前模型的决策逻辑",
            "质量评估": {
                "完整性评分": "0.85",
                "特征有效性": "0.78",
                "历史准确率": "0.82"
            }
        }
    },
    "observation": {
        "completed": ["已完成事项"],
        "pending": ["待完成事项"],
        "phase": "数据获取|数据分析|模型预测|结果生成",
        "analysis_report": {
            "validity_score": "0.85",
            "time_series_type": "周期性（每5区块出现峰值）",
            "recommended_model": "cluster_prediction",
            "reason": "数据呈现稳定周期性波动，符合聚类拟合适用场景"
        }
    }
}
"""

# 接下来把每个prompt生成
# 调用生成函数拿到调用函数的描述
action_prompt = gen_tools_description()
constraints_prompt = "\n".join([f"{idx+1}. {constraint}" for idx, constraint in enumerate(constraints)])
resources_prompt = "\n".join([f"{idx+1}. {resource}" for idx, resource in enumerate(resources)])
best_practices_prompt = "\n".join([f"{idx+1}. {best_practice}" for idx, best_practice in enumerate(best_practices)])


# 历史摘要生成函数
def generate_history_summary(scratch):
    summary = []
    for step in scratch.get("steps", [])[-3:]:  # 仅保留最近3步
        summary.append(
            f"Step {step['step']}: 执行 {step['action']} "
            f"(参数: {step['args']}), 结果摘要: {step['result']}..."
        )
    print("\n".join(summary))
    return "\n".join(summary)


# 组装
def gen_prompt(query, agent_scratch):
    prompt = prompt_template.format(
        query=query,
        history_summary=generate_history_summary(agent_scratch),
        constraints=constraints_prompt,
        actions=action_prompt,
        resources=resources_prompt,
        best_practices=best_practices_prompt,
        response_format_prompt=response_format_prompt
    )
    return prompt


user_prompt = "根据给定的目标和迄今为止取得的进展，确定下一个要执行的action，并使用前面指定的JSON格式进行相应"

