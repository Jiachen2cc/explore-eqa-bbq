
def iou_bbox(pred_bbox, tar_bbox, threshold = 0.1):
    
    # 1. compute iou between pred_bbox and tar_bbox
    ...
    
    # 2. if iou > threshold, return True, else return False
    return ...


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')