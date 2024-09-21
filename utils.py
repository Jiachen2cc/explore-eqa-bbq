import base64
import numpy as np

def iou_bbox(pred_bbox, tar_bbox, threshold = 0.1, oriented = False):
    
    # 1. compute iou between pred_bbox and tar_bbox
    if not oriented:
      inter = np.maximum(0, np.minimum(pred_bbox[1], tar_bbox[1]) - np.maximum(pred_bbox[0], tar_bbox[0]))
      union = np.maximum(0, np.maximum(pred_bbox[1], tar_bbox[1]) - np.minimum(pred_bbox[0], tar_bbox[0]))
      inter_size = np.prod(inter)
      union_size = np.prod(union)
      if union_size > 0:
        iou = inter_size/union_size
      else:
        iou = 0
    else:
        ...
    # 2. if iou > threshold, return True, else return False
    print("iou",iou)  
    #exit(0)
    return iou

def oriented_to_raw_bbox(bbox):
    # 1. convert the oriented bbox to raw bbox
    bbox = np.array(bbox)
    xyz_max = np.max(bbox, axis=0)
    xyz_min = np.min(bbox, axis=0)
    return [xyz_min, xyz_max]

def center_bbox_to_raw_bbox(bbox):
    # 2. convert the center bbox to raw bbox
    bbox = np.array(bbox)
    xyz_min = bbox[:3] - bbox[3:6]/2
    xyz_max = bbox[:3] + bbox[3:6]/2
    return [xyz_min, xyz_max]
  
def raw_bbox_size(bbox):
    # bbox = [xyz_min, xyz_max]
    bbox = np.array(bbox)
    xyz_length = max(bbox[1] - bbox[0],0)
    return np.prod(xyz_length)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')