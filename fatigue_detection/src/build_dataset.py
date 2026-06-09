"""
数据集构建模块
负责：
1. 读取特征 CSV
2. 滑窗切分 (窗口大小: 300帧, 步长: 30帧)
3. 标注策略 (窗口内疲劳帧占比 > 50% 则标注为疲劳)
4. 输出正负样本均衡的 npy 数组，用于训练
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

WINDOW_SIZE = 300
STRIDE = 30

def create_windows(feature_data):
    """
    feature_data: DataFrame with columns ['EAR', 'MAR', 'pitch', 'yaw']
    Return: X (N, WINDOW_SIZE, 4), y (N,)
    """
    X, y = [], []
    num_frames = len(feature_data)
    
    # 基于经验阈值的伪标签标注： EAR < 0.2 或者 MAR > 0.5 视为疲劳帧
    is_fatigue = ((feature_data['EAR'] < 0.2) | (feature_data['MAR'] > 0.5)).astype(int)
    
    for start in range(0, num_frames - WINDOW_SIZE + 1, STRIDE):
        window = feature_data.iloc[start:start+WINDOW_SIZE]
        window_labels = is_fatigue.iloc[start:start+WINDOW_SIZE]
        
        # 窗口内疲劳帧占比 > 50% 则标注为疲劳(1)
        label = 1 if window_labels.mean() > 0.5 else 0
        
        X.append(window[['EAR', 'MAR', 'pitch', 'yaw']].values)
        y.append(label)
        
    return np.array(X), np.array(y)

def build_dataset(features_dir, output_dir):
    all_X, all_y = [], []
    
    if not os.path.exists(features_dir):
        print(f"错误: 特征目录 {features_dir} 不存在。")
        return
        
    for file in os.listdir(features_dir):
        if file.endswith('.csv'):
            filepath = os.path.join(features_dir, file)
            df = pd.read_csv(filepath)
            if len(df) < WINDOW_SIZE:
                print(f"跳过 {file}: 帧数不足 {WINDOW_SIZE}")
                continue
                
            X, y = create_windows(df)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                
    if not all_X:
        print("未能生成任何有效的数据窗口。请确保 CSV 中包含足够帧数的特征数据。")
        return
        
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    # 平衡数据集：通过对少数类进行过采样处理
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    
    if len(pos_idx) > 0 and len(neg_idx) > 0:
        max_len = max(len(pos_idx), len(neg_idx))
        pos_idx = np.random.choice(pos_idx, max_len, replace=True)
        neg_idx = np.random.choice(neg_idx, max_len, replace=True)
        bal_idx = np.concatenate([pos_idx, neg_idx])
        np.random.shuffle(bal_idx)
        X = X[bal_idx]
        y = y[bal_idx]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 按照 8:1:1 划分 train / val / test
    if len(X) > 2:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        # 防止样本极少时的报错
        if len(X_train_val) > 1:
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1/9, random_state=42) 
        else:
            X_train, X_val, y_train, y_val = X_train_val, X_train_val, y_train_val, y_train_val
    else:
        X_train, X_val, X_test = X, X, X
        y_train, y_val, y_test = y, y, y

    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    print(f"数据集已成功构建并保存至: {output_dir}")
    print(f"分布 - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_dir = os.path.join(base_dir, 'data', 'features')
    output_dir = os.path.join(base_dir, 'data', 'windows')
    build_dataset(features_dir, output_dir)
