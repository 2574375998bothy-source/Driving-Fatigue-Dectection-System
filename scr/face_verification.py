from deepface import DeepFace
import os
import cv2

def verify_driver(frame, db_path="data/authorized_users"):
    """
    验证给定的视频帧中的人脸是否在授权用户数据库中。
    
    Args:
        frame: OpenCV BGR 图像帧
        db_path: 授权用户图片所在的目录路径
        
    Returns:
        bool: 如果匹配成功返回 True，否则返回 False
    """
    if not os.path.exists(db_path):
        os.makedirs(db_path, exist_ok=True)
        
    # 如果数据库中没有图片，默认拒绝访问
    if len([f for f in os.listdir(db_path) if f.endswith(('.jpg', '.png', '.jpeg'))]) == 0:
        print("警告: data/authorized_users 目录为空，请添加授权驾驶员照片。")
        return False
        
    try:
        # DeepFace.find 会在 db_path 中查找与 img_path(这里传入帧的 numpy array) 匹配的人脸
        # enforce_detection=False 避免在未检测到人脸时抛出异常
        # silent=True 避免在终端输出大量日志
        results = DeepFace.find(
            img_path=frame, 
            db_path=db_path, 
            enforce_detection=False, 
            silent=True,
            model_name="VGG-Face", # 可以换成 Facenet 或其他轻量模型
            detector_backend="opencv" # 使用 opencv 比较轻量
        )
        
        # results 是一个包含 Pandas DataFrame 的列表（每个检测到的人脸对应一个 df）
        # 如果 df 不为空，说明在数据库中找到了匹配项
        if len(results) > 0 and not results[0].empty:
            return True
            
        return False
    except Exception as e:
        print(f"人脸比对发生错误: {e}")
        return False
