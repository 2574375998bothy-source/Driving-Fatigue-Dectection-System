"""
实时推理 Pipeline
负责：
1. 连接摄像头或视频流
2. MediaPipe 提取特征
3. 滑动窗口缓冲
4. 阈值兜底判断 (PERCLOS) + LSTM 预测
5. 结果显示及报警输出
"""

import cv2
import numpy as np
import collections
from tensorflow.keras.models import load_model
from extract_features import calculate_ear, calculate_mar, estimate_head_pose

# TODO: 完善特征缓存和推理逻辑
WINDOW_SIZE = 300

def run_inference(model_path, source=0):
    model = load_model(model_path)
    feature_buffer = collections.deque(maxlen=WINDOW_SIZE)
    
    # 视频捕获与处理循环
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # TODO: 执行单帧处理，提取特征加入 buffer，当 buffer 满时执行模型推理和 PERCLOS 统计兜底
        
        cv2.imshow('Fatigue Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    pass
