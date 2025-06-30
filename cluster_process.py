import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, dump
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dl_process import future_label

threshold = 20
window = 25
slide = 1
valid_ratio = 0.4  # 用于验证的数据比例，数据太少的窗口会被丢弃
degree = 5
k = 15


# 用时：5.84s
def cluster_process():
    # 数据加载与预处理
    data = pd.read_csv("new_2_internal_transactions.csv")
    data = data[data['isError'] == 0]
    data = data[['blockNumber', 'to', 'contractAddress', 'type']]
    data['type'], type_mapping = pd.factorize(data['type'])
    # 改进：data['to'] = data.apply(lambda row: row['contractAddress'] if row['type']==1  else row['to'], axis=1)
    mask = data['type'] == 1
    data['to'] = np.where(mask, data['contractAddress'], data['to'])
    data = data.drop(columns=['type', 'contractAddress'])
    # 按照blockNumber和to进行分组，即每个区块每个合约被调用次数，以区块为最小粒度
    data = data.groupby(['blockNumber', 'to']).size().reset_index(name='counts')
    # 再按照合约地址进行分组
    sum_data = data.groupby(['to']).agg({'counts': 'sum'}).reset_index()
    print(len(sum_data))

    mean = sum_data['counts'].mean()
    std = sum_data['counts'].std()
    print('平均:', mean)
    print('标准差:', std)

    # 筛选出调用次数大
    filtered_data = data[data['to'].isin(sum_data[sum_data['counts'] > threshold]['to'])]
    print(len(filtered_data))
    group_data = filtered_data.groupby(['to'])

    def fit_per_window(group):
        blocks = group['blockNumber'].values.astype(int)
        counts = group['counts'].values

        coefficients = []
        n_blocks = len(blocks)

        X_test = []
        y_test = []

        i = 0
        while i < n_blocks - window + 1:
            # 当前窗口的起始块和结束块
            start_block = blocks[i]
            end_block = start_block + window

            # 使用二分查找确定窗口结束位置
            j = np.searchsorted(blocks, end_block, side='right')

            # 找到窗口内的所有区块（包括有交易的）
            window_mask = (blocks >= start_block) & (blocks < end_block)
            window_blocks = blocks[window_mask]
            window_counts = counts[window_mask]

            # 创建全零数组并填充实际存在的交易次数
            full_counts = np.zeros(window)
            # 计算相对坐标，将实际存在的交易次数填充到对应的位置
            existing_indices = window_blocks - start_block
            full_counts[existing_indices] = window_counts

            # 有效性检查，如果窗口内交易次数占比低于阈值，则跳过该窗口
            valid_ratio_actual = np.sum(full_counts > 0) / window
            if valid_ratio_actual < valid_ratio:
                # 寻找下一个有效起始点
                next_block_idx = j  # 默认跳到当前窗口结束位置
                if j < n_blocks:
                    # 计算下一个可能有效的位置：下一个区块的前window个位置
                    next_valid_start = max(blocks[j] - window + 1, start_block + 1)
                    next_block_idx = np.searchsorted(blocks, next_valid_start, side='left')

                i = next_block_idx
                continue

            try:
                # 使用相对时间偏移
                x = np.arange(window) + 1
                y = full_counts
                model = np.polyfit(x, y, degree)
                coefficients.append(model)
                test = np.polyfit(np.arange(window - future_label) + 1, full_counts[0:window - future_label], degree)
                X_test.append(test)
                y_test.append(full_counts[window - future_label:])
            except Exception as e:
                print(f"Error : {e}")
                pass

            # 动态步长调整：至少滑动1个区块
            i += max(slide, 1)

        return coefficients, X_test, y_test

    # 并行处理所有合约
    result = Parallel(n_jobs=-1)(
        delayed(fit_per_window)(group) for _, group in tqdm(group_data, desc='Processing groups')
    )

    # 收集所有窗口的拟合参数
    all_coefficients = []
    X_test = []
    y_test = []
    for r in result:
        all_coefficients.extend(r[0])
        X_test.extend(r[1])
        y_test.extend(r[2])
    print(len(X_test))

    # 转换为NumPy数组
    fit = np.array(all_coefficients)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    fit = fit[:int(0.8 * len(all_coefficients))]
    X_test = X_test[int(0.8 * len(all_coefficients)):, :]
    y_test = y_test[int(0.8 * len(all_coefficients)):, :]

    # 预处理
    scaler = StandardScaler()
    fit = scaler.fit_transform(fit)
    # 聚类
    kmeans = KMeans(n_clusters=k).fit(fit)
    # 反标准化
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # 画出曲线
    x = np.arange(window) + 1
    y = np.zeros((k, len(x)))
    for i in range(k):
        for j in range(degree + 1):
            y[i] += cluster_centers[i][j] * (x ** (degree - j))

    plt.figure()
    for i in range(k):
        plt.plot(x, y[i], label=f'cluster {i}')

    X_test_scaled = scaler.transform(X_test)
    y_pred_label = kmeans.predict(X_test_scaled)

    #  预测
    y_pred = np.zeros((len(y_test), future_label))
    for i in range(len(y_test)):
        for j in range(future_label):
            future_step = window + j + 1 - future_label
            for z in range(degree + 1):
                y_pred[i][j] += cluster_centers[y_pred_label[i]][z] * (future_step ** (degree - z))

    def evaluate_predictions(y_true, y_pred):
        # 展平维度计算整体指标
        flat_true = y_true.flatten()
        flat_pred = y_pred.flatten()

        # 平均绝对误差和均方误差
        print(f"Overall MAE: {mean_absolute_error(flat_true, flat_pred):.2f}")
        print(f"Overall MSE: {mean_squared_error(flat_true, flat_pred):.2f}")

        # 分步长计算指标
        for step in range(future_label):
            step_mae = mean_absolute_error(y_true[:, step], y_pred[:, step])
            print(f"Step {step + 1} MAE: {step_mae:.2f}")

    evaluate_predictions(y_test, y_pred)

    dump(scaler, 'cluster_scaler.joblib')
    dump(kmeans, 'cluster_model.joblib')


if __name__ == '__main__':
    cluster_process()