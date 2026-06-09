"""
模型训练模块
负责：
1. 定义 LSTM 模型 (32 units)
2. 训练并评估模型 (Accuracy, AUC-ROC, F1-Score)
3. 保存最佳模型 (lstm_fatigue.h5)
"""

import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def build_model(input_shape=(300, 4)):
    model = Sequential([
        LSTM(32, input_shape=input_shape, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    return model

def train_model(train_x, train_y, val_x, val_y, model_save_path):
    model = build_model((train_x.shape[1], train_x.shape[2]))
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    return model, history

def evaluate_model(model, test_x, test_y):
    preds = model.predict(test_x)
    pred_labels = (preds > 0.5).astype(int)
    
    # 只有一类样本时，AUC 无法计算
    if len(np.unique(test_y)) > 1:
        auc = roc_auc_score(test_y, preds)
    else:
        auc = float('nan') 
    
    acc = accuracy_score(test_y, pred_labels)
    f1 = f1_score(test_y, pred_labels, zero_division=0)
    
    print(f"\n========== 评估结果 ==========")
    print(f"Accuracy : {acc:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"==============================")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    windows_dir = os.path.join(base_dir, 'data', 'windows')
    models_dir = os.path.join(base_dir, 'models')
    
    try:
        x_train = np.load(os.path.join(windows_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(windows_dir, 'y_train.npy'))
        x_val = np.load(os.path.join(windows_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(windows_dir, 'y_val.npy'))
        x_test = np.load(os.path.join(windows_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(windows_dir, 'y_test.npy'))
    except FileNotFoundError:
        print("未找到已构建的数据集 (npy 文件)，请先运行 build_dataset.py。")
        exit(1)
    
    model_save_path = os.path.join(models_dir, 'lstm_fatigue.h5')
    
    print("开始训练 LSTM 模型...")
    model, history = train_model(x_train, y_train, x_val, y_val, model_save_path)
    
    print("正在评估模型...")
    evaluate_model(model, x_test, y_test)
