import cv2
import numpy as np

def is_camera_occluded(frame, dark_threshold=15, variance_threshold=2.0):
    """
    检测摄像头是否被物理遮挡。
    主要通过检测画面是否过暗，或者画面细节（方差）极小来判断。
    
    Args:
        frame: OpenCV BGR 图像帧
        dark_threshold: 亮度阈值，低于此值认为过暗（例如用手捂住）
        variance_threshold: 方差阈值，低于此值认为画面极其平滑无细节（例如被纯色物体贴住）
        
    Returns:
        bool: 如果被遮挡返回 True，否则返回 False
    """
    if frame is None or frame.size == 0:
        return True
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 检查平均亮度
    mean_brightness = np.mean(gray)
    if mean_brightness < dark_threshold:
        return True
        
    # 检查图像细节 (拉普拉斯算子的方差)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < variance_threshold:
        return True
        
    return False
