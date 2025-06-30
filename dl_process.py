import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

threshold = 20
window = 20
slide = 1
valid_ratio = 0.4
future_label = 5
feature_cols = ['gas_mean', 'gasUsed_mean', 'counts', 'hour', 'minute', 'second']


# 用时：85.54s
def dl_process():
    s_time = time.time()
    # 参数设定
    # 数据加载与预处理（解决 SettingWithCopyWarning）

    def load_and_preprocess(filepath):
        data = pd.read_csv(filepath)
        data = data[data['isError'] == 0].copy()
        data['type'], _ = pd.factorize(data['type'])
        mask = data['type'] == 1
        data['to'] = np.where(mask, data['contractAddress'], data['to'])
        data = data.drop(columns=['type', 'contractAddress'])

        # 聚合数据
        data = data.groupby(['blockNumber', 'timeStamp', 'to']).agg(
            gas_mean=('gas', 'mean'),
            gasUsed_mean=('gasUsed', 'mean'),
            counts=('gas', 'size')
        ).reset_index()

        # 过滤低频合约
        sum_data = data.groupby('to')['counts'].sum().reset_index()
        filtered_data = data[data['to'].isin(sum_data[sum_data['counts'] > threshold]['to'])].copy()

        # 时间特征
        filtered_data['timeStamp'] = pd.to_datetime(filtered_data['timeStamp'], unit='s')
        filtered_data['hour'] = filtered_data['timeStamp'].dt.hour
        filtered_data['minute'] = filtered_data['timeStamp'].dt.minute
        filtered_data['second'] = filtered_data['timeStamp'].dt.second
        return filtered_data.drop(columns=['timeStamp']).sort_values('blockNumber')

    # 加载数据
    filtered_data = load_and_preprocess("new_2_internal_transactions.csv")

    split_idx = int(0.8 * len(filtered_data))
    train_data = filtered_data.iloc[:split_idx]
    test_data = filtered_data.iloc[split_idx:]

    group_train_data = train_data.groupby(['to'])
    group_test_data = test_data.groupby(['to'])

    def create_sequences(group):
        blocks = group['blockNumber'].values.astype(int)
        counts = group['counts'].values

        i = 0
        n_blocks = len(blocks)

        feature = []
        labels = []
        while i < n_blocks - 1:
            # 当前窗口的起始区块号
            start_block = blocks[i]
            end_block = start_block + window

            # 使用二分查找确定窗口结束位置
            j = np.searchsorted(blocks, end_block, side='right')

            # 窗口内实际存在的区块索引
            window_mask = (blocks >= start_block) & (blocks < end_block)
            window_blocks = blocks[window_mask]
            window_counts = counts[window_mask]

            # 填充完整的区块序列（即使没有交易）
            full_features = np.zeros((window, len(feature_cols)))
            full_counts = np.zeros(window)
            existing_indices = window_blocks - start_block
            full_features[existing_indices] = group[window_mask][feature_cols].values
            full_counts[existing_indices] = window_counts

            # 有效性检查
            valid_ratio_actual = np.sum(full_counts > 0) / window
            if valid_ratio_actual < valid_ratio:
                # 动态调整起始点：跳到下一个有效位置
                next_block_idx = j  # 默认跳到当前窗口结束位置
                if j < n_blocks:
                    # 计算下一个可能有效的位置：下一个区块的前window个位置
                    next_valid_start = max(blocks[j] - window + 1, start_block + 1)
                    next_block_idx = np.searchsorted(blocks, next_valid_start, side='left')

                i = next_block_idx
                continue

            # 生成标签：用未来future_label个区块作为标签
            label = np.arange(future_label)
            future_start = end_block
            future_end = future_start + future_label
            future_mask = (blocks >= future_start) & (blocks < future_end)
            future_blocks = blocks[future_mask]
            future_counts = counts[future_mask]
            future_existing = future_blocks - end_block
            label[future_existing] = future_counts

            # 保存序列和标签
            feature.append(full_features)
            labels.append(label)
            # 滑动步长（至少滑动1个区块）
            i += max(slide, 1)

        return feature, labels

    create_train = Parallel(n_jobs=-1)(
        delayed(create_sequences)(group) for _, group in tqdm(group_train_data, desc='Train')
    )

    create_test = Parallel(n_jobs=-1)(
        delayed(create_sequences)(group) for _, group in tqdm(group_test_data, desc='Test')
    )

    # 创建时间序列数据集
    X_train, y_train = [], []
    X_test, y_test = [], []
    for train in create_train:
        X_train.extend(train[0])
        y_train.extend(train[1])
    for test in create_test:
        X_test.extend(test[0])
        y_test.extend(test[1])
    print(len(X_train), len(y_train))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(X_train.shape, y_train.shape)

    # 将三维数据展平为二维进行标准化
    n_samples, timesteps, n_features = X_train.shape
    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_2d).reshape(n_samples, timesteps, n_features)
    X_test_scaled = scaler_X.transform(X_test_2d).reshape(X_test.shape)

    # 标签标准化（新增）
    y_train = np.array(y_train).reshape(-1, future_label)  # 形状变为 (n_samples, 5)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)

    print(y_train.shape)
    model = Sequential([
        # Encoder
        LSTM(128, return_sequences=True, input_shape=(window, len(feature_cols))),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        # Decoder
        tf.keras.layers.RepeatVector(future_label),  # 关键：重复预测步长
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)  # 输出形状自动变为 (None, 5, 1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 训练模型
    history = model.fit(
        X_train_scaled,
        y_train_scaled.reshape(-1, future_label, 1),  # 调整为 (samples, 5, 1)
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
    )

    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, future_label))  # 形状恢复为 (n_samples, 5)
    print(y_test.shape, y_pred.shape)

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

    dump(scaler_X, 'scaler_X.joblib')
    dump(scaler_y, 'scaler_y.joblib')
    model.save('lstm_model')

# if __name__ == '__main__':
#     dl_process()