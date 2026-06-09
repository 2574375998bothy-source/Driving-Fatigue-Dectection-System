import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import math
import os
import urllib.request

# 下载并加载 MediaPipe FaceLandmarker 模型
def load_face_landmarker():
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
        print("正在下载 face_landmarker.task 模型...")
        url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
        urllib.request.urlretrieve(url, model_path)
        print("模型下载完成！")
        
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

# ----------------- 关键点索引定义 -----------------
# 采用 MediaPipe Face Mesh 的 468 关键点索引
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # 左眼轮廓
RIGHT_EYE = [362, 385, 387, 263, 373, 380] # 右眼轮廓

# 嘴巴内圈关键点 (用于 MAR)
# 左嘴角(78), 右嘴角(308), 上嘴唇(81, 13, 311), 下嘴唇(178, 14, 402)
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_ear(landmarks, eye_indices):
    pts = [landmarks[i] for i in eye_indices]
    
    # 垂直距离
    v1 = euclidean_distance(pts[1], pts[5])
    v2 = euclidean_distance(pts[2], pts[4])
    # 水平距离
    h = euclidean_distance(pts[0], pts[3])
    
    if h == 0:
        return 0.0
    
    return (v1 + v2) / (2.0 * h)

def calculate_mar(landmarks):
    pts = [landmarks[i] for i in MOUTH]
    
    # 垂直距离
    v1 = euclidean_distance(pts[1], pts[7])
    v2 = euclidean_distance(pts[2], pts[6])
    v3 = euclidean_distance(pts[3], pts[5])
    # 水平距离
    h = euclidean_distance(pts[0], pts[4])
    
    if h == 0:
        return 0.0
        
    return (v1 + v2 + v3) / (2.0 * h)

def estimate_head_pose_from_matrix(transformation_matrix):
    """
    通过 MediaPipe 直接提供的面部转换矩阵计算 Pitch 和 Yaw。
    Transformation matrix 是一个 4x4 的姿态矩阵。
    """
    # 提取 3x3 旋转矩阵
    rmat = transformation_matrix[:3, :3]
    
    # 使用 OpenCV 将旋转矩阵分解为欧拉角
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    pitch = angles[0] * 360
    yaw = angles[1] * 360
    return pitch, yaw

def process_video(video_path, output_csv_path, show_video=True):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: 无法打开视频 {video_path}")
        return
        
    features_list = []
    frame_count = 0
    detector = load_face_landmarker()
    
    print(f"开始处理视频: {video_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        img_h, img_w, _ = frame.shape
        
        # 转换为 RGB 并构造 mp.Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 使用新的 Tasks API 进行检测
        detection_result = detector.detect(mp_image)
        
        if detection_result.face_landmarks and detection_result.facial_transformation_matrixes:
            face_landmarks = detection_result.face_landmarks[0]
            transformation_matrix = detection_result.facial_transformation_matrixes[0]
            
            # 提取 (x, y) 像素坐标
            landmarks_px = []
            for lm in face_landmarks:
                landmarks_px.append((lm.x * img_w, lm.y * img_h))
            
            # 计算 EAR 和 MAR
            left_ear = calculate_ear(landmarks_px, LEFT_EYE)
            right_ear = calculate_ear(landmarks_px, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(landmarks_px)
            
            # 使用官方给的变换矩阵计算姿态，更为准确
            pitch, yaw = estimate_head_pose_from_matrix(transformation_matrix)
            
            features_list.append({
                'frame': frame_count,
                'EAR': ear,
                'MAR': mar,
                'pitch': pitch,
                'yaw': yaw
            })
            
            if show_video:
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Pitch: {pitch:.1f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.1f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            default_val = features_list[-1] if features_list else {'frame': frame_count, 'EAR': 0.3, 'MAR': 0.0, 'pitch': 0.0, 'yaw': 0.0}
            default_val = dict(default_val)
            default_val['frame'] = frame_count
            features_list.append(default_val)
            
        if show_video:
            cv2.imshow('Feature Extraction Preview', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df = pd.DataFrame(features_list)
    df.to_csv(output_csv_path, index=False)
    print(f"处理完成，特征已保存至: {output_csv_path} (共 {frame_count} 帧)")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_video = os.path.join(base_dir, 'data', 'raw', 'test_video.mp4')
    output_csv = os.path.join(base_dir, 'data', 'features', 'test_features.csv')
    
    if os.path.exists(test_video):
        process_video(test_video, output_csv, show_video=True)
    else:
        print(f"请准备测试视频放置于: {test_video}")
