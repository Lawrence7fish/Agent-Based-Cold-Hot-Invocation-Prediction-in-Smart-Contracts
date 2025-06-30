"""
1.写文件
2.读文件
3.追加文件
4.线性拟合聚类
5.深度学习模型
6.历史数据
"""
import os
import json
import re
import time
import joblib
from joblib import load
from tensorflow.keras.models import load_model
import numpy as np
from information import get_internal_cluster_pred, get_internal_lstm_pred, raw_data
from dl_process import window, feature_cols, valid_ratio, future_label
from cluster_process import degree


def _get_workdir_root():
    return os.environ.get('WORKDIR_ROOT', './data/llm_result')


WORKDIR_ROOT = _get_workdir_root()


def get_raw_data(address, current_block):
    if not validate_eth_address(address):
        return {"error": "非法地址格式", "code": 400}
    try:
        start_time = time.time()
        r_data = raw_data(address, current_block)
        return {
            "raw_data": np.array(r_data).tolist(),
            "execution_time": round(start_time - time.time(), 2)
        }
    except Exception as e:
        return {"error": str(e), "code": 400}

def validate_eth_address(address):
    return re.match(r'^0x[a-fA-F0-9]{40}$', address)


def cluster_prediction(address, current_block):
    if not validate_eth_address(address):
        return {"error": "非法地址格式", "code": 400}
    try:
        start_time = time.time()
        c_data = get_internal_cluster_pred(address, current_block)
        c_model = joblib.load('cluster_model.joblib')

        # cluster 预测
        full_counts = np.zeros(window)
        blocks = c_data['blockNumber'].values.astype(int)
        counts = c_data['counts'].values.astype(int)
        existing_indices = blocks - current_block + window - 1
        full_counts[existing_indices] = counts

        # 合法性检查
        valid_ratio_actual = np.sum(full_counts > 0) / window
        if valid_ratio_actual < valid_ratio:
            return "该账户无需预测，视为冷账户"

        x = np.arange(window) + 1
        y = full_counts
        fit = np.polyfit(x, y, degree)
        cluster_scaler = load('cluster_scaler.joblib')
        fit = cluster_scaler.transform(fit.reshape(1, -1))  # 进行标准化后预测类型
        c_pred_label = c_model.predict(fit)
        cluster_center = cluster_scaler.inverse_transform(c_model.cluster_centers_[c_pred_label[0]].reshape(1, -1))[
            0]  # 获取对应类别的中心参数
        print(cluster_center)
        c_pred = np.zeros(future_label)
        for i in range(future_label):
            future_step = window + i + 1
            for j in range(degree + 1):
                c_pred[i] += cluster_center[j] * (future_step ** (degree - j))
        # 增加结果解释
        return {
            "prediction": c_pred.tolist(),
            "execution_time": round(time.time() - start_time, 2)
        }
    except Exception as e:
        return {"error": str(e), "code": 500}


def lstm_prediction(address, current_block):
    if not validate_eth_address(address):
        return {"error": "非法地址格式", "code": 400}
    try:
        start_time = time.time()
        d_data = get_internal_lstm_pred(address, current_block)
        # 加载模型
        d_model = load_model('lstm_model')

        # cluster 预测
        full_counts = np.zeros(window)
        blocks = d_data['blockNumber'].values.astype(int)
        counts = d_data['counts'].values.astype(int)
        existing_indices = blocks - current_block + window - 1
        full_counts[existing_indices] = counts

        # 合法性检查
        valid_ratio_actual = np.sum(full_counts > 0) / window
        if valid_ratio_actual < valid_ratio:
            return "该账户无需预测，视为冷账户"

        # dl预测
        full_features = np.zeros((window, len(feature_cols)))
        full_features[existing_indices] = d_data[feature_cols].values

        # 载入训练时的scaler，保证标准化是一样的
        X_scaler = load('scaler_X.joblib')
        y_scaler = load('scaler_y.joblib')
        full_features_scaled = X_scaler.transform(full_features)  # 标准化
        input_data = full_features_scaled.reshape(1, window, -1)  # 变成(1,20,6)
        y_pred = d_model.predict(input_data)
        # 反标准化
        d_pred = y_scaler.inverse_transform(y_pred.reshape(-1, future_label))

        return {
            "prediction": d_pred[0].tolist(),
            "execution_time": round(time.time() - start_time, 2)
        }
    except Exception as e:
        return {"error": str(e), "code": 500}


def analyze_data(analyze_result, current_block):
    """LLM实际分析数据的空动作处理器"""
    # 此处不进行真实计算，而是强制LLM必须生成结构化分析
    return {
        "status": "analysis_required",  # 特殊状态标识
        "message": "等待LLM生成分析报告"
    }

tools_info = [
    {
        "name": "get_raw_data",
        "description": "获取前20个区块的信息，包括1个标记: '该条为为前几个块'，和6个特征：'gas_mean', 'gasUsed_mean', 'counts', 'hour', 'minute', 'second':\n"
                      "- 区块号（转换为前第几个块）、时间戳（转换为时分秒）\n"
                      "- gas消耗量均值、gas限额均值\n"
                      "- 调用次数统计\n"
                      "- 周期性时间编码特征",
        "args": [
            {"name": "address", "type": "string", "description": "账户地址"},
            {"name": "current_block", "type": "int", "description": "当前区块号"}
        ]
    },
    {
        "name": "cluster_prediction",
        "description": "基于多项式拟合的聚类模式预测，根据当前区块的前20个区块，给出未来若干个区块交易数量预测",
        "args": [
            {"name": "address", "type": "string", "description": "账户地址"},
            {"name": "current_block", "type": "int", "description": "当前区块号"}
        ]
    },
    {
        "name": "lstm_prediction",
        "description": "基于LSTM深度学习的时间序列预测，根据当前区块的前20个区块，给出未来若干个区块交易数量预测",
        "args": [
            {"name": "address", "type": "string", "description": "账户地址"},
            {"name": "current_block", "type": "int", "description": "当前区块号"}
        ]
    },
    {
        "name": "finish",
        "description": "完成用户目标（需同时满足：目标达成/遇到不可抗力/达到重试上限）",
        "args": [
            {"name": "answer", "type": "string", "description": "最后的目标结果"}
        ]
    },
    {
        "name": "analyze_data",
        "description": "对已获取的原始数据进行智能分析，必须完成以下任务：\n"
                   "1. 计算数据有效性指标（有效区块占比）\n"
                   "2. 检测时序连续性（最长连续活跃区块数）\n"
                   "3. 识别异常波动（标准差>2视为异常）\n"
                   "4. 生成模型选择建议",
        "args": [
            {"name": "analyze_result", "type": "string", "description": "根据get_raw_data返回的原始数据而分析得出的结果，便于下一轮调用做出决策"},
            {"name": "current_block", "type": "int", "description": "当前区块号"}
        ]
    }
]

# tools map供给llm来选调用的函数
tools_map = {
    "get_raw_data": get_raw_data,
    "cluster_prediction": cluster_prediction,
    "lstm_prediction": lstm_prediction,
    "analyze_data": analyze_data,
    "finish": lambda **args: args.get("answer", "")  # 空动作处理器
}


#  生成工具描述
def gen_tools_description():
    tools_description = []
    for idx, tool in enumerate(tools_info):
        args_desc = []
        for arg in tool['args']:
            args_desc.append({
                "name": arg["name"],
                "description": arg["description"],
                "type": arg["type"]
            })
        args_desc = json.dumps(args_desc, ensure_ascii=False)
        tool_description = f"{idx + 1}.{tool['name']}:{tool['description']}, args: {args_desc}"
        tools_description.append(tool_description)
    # 拼成一行字符串
    tools_prompt = "\n".join(tools_description)
    return tools_prompt

