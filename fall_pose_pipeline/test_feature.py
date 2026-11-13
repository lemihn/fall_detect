import os
import re
import math
import numpy as np
import cv2
from ultralytics import YOLO

def extract_features_from_person(keypoints_xy, keypoints_conf, box_xyxy):
    """
    keypoints_xy: (17, 2)
    keypoints_conf: (17,)
    box_xyxy: [x1, y1, x2, y2]
    Returns 36-dim feature vector
    """
    x1, y1, x2, y2 = box_xyxy
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    scale = max(w, h, 1.0)
    
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    
    # 17 normalized keypoints (x, y) -> 34 values
    norm_kpts = []
    for (kx, ky) in keypoints_xy:
        norm_x = (kx - cx) / scale
        norm_y = (ky - cy) / scale
        norm_kpts.extend([norm_x, norm_y])
        
    # Aspect ratio
    aspect_ratio = w / h
    
    # Torso vertical orientation angle
    # Shoulder center: kpts 5 (left shoulder) & 6 (right shoulder)
    # Hip center: kpts 11 (left hip) & 12 (right hip)
    sh_x = (keypoints_xy[5][0] + keypoints_xy[6][0]) / 2.0
    sh_y = (keypoints_xy[5][1] + keypoints_xy[6][1]) / 2.0
    hip_x = (keypoints_xy[11][0] + keypoints_xy[12][0]) / 2.0
    hip_y = (keypoints_xy[11][1] + keypoints_xy[12][1]) / 2.0
    
    dx = sh_x - hip_x
    dy = sh_y - hip_y
    # Angle with vertical axis (0: standing upright, ~pi/2: horizontal/lying)
    torso_angle = math.atan2(abs(dx), abs(dy) + 1e-4)
    
    features = norm_kpts + [aspect_ratio, torso_angle]
    return np.array(features, dtype=np.float32)

print("Helper function defined successfully.")
