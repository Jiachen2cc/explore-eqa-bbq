import base64
import numpy as np
import pandas as pd

def iou_bbox(pred_bbox, tar_bbox, oriented = False):
    
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
  
def initialize_count(mode = "nr3d"):
  data = {
    'type': ['overall','easy','hard','view dependent','view independent'],
    'succ01_count': [0 for i in range(5)],
    'succ025_count': [0 for i in range(5)],
    'overall_count': [0 for i in range(5)]
  }
  df = pd.DataFrame(data)
  return df
    
def update_count(cur_df, iou, query):
  success01, success025 = int(iou >= 0.1), int(iou >= 0.25)
  print(f"iou: {iou}, success01: {success01}, success025: {success025}")
  cur_df.loc[cur_df['type'] == 'overall','overall_count'] += 1
  cur_df.loc[cur_df['type'] == 'overall','succ01_count'] += success01
  cur_df.loc[cur_df['type'] == 'overall','succ025_count'] += success025
  
  if query['is_easy']:
    cur_df.loc[cur_df['type'] == 'easy','overall_count'] += 1
    cur_df.loc[cur_df['type'] == 'easy','succ01_count'] += success01
    cur_df.loc[cur_df['type'] == 'easy','succ025_count'] += success025
  else:
    cur_df.loc[cur_df['type'] == 'hard','overall_count'] += 1
    cur_df.loc[cur_df['type'] == 'hard','succ01_count'] += success01
    cur_df.loc[cur_df['type'] == 'hard','succ025_count'] += success025
  
  if query['is_view_dep']:
    cur_df.loc[cur_df['type'] == 'view dependent','overall_count'] += 1
    cur_df.loc[cur_df['type'] == 'view dependent','succ01_count'] += success01
    cur_df.loc[cur_df['type'] == 'view dependent','succ025_count'] += success025
  else:
    cur_df.loc[cur_df['type'] == 'view independent','overall_count'] += 1
    cur_df.loc[cur_df['type'] == 'view independent','succ01_count'] += success01
    cur_df.loc[cur_df['type'] == 'view independent','succ025_count'] += success025
  
  return cur_df
  
  
    