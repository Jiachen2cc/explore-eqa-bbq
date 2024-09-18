import json
from utils import encode_image
class Scene:
    
    def __init__(self, scene_dir):
        self.scene_dir = scene_dir
        self.snapshot = {}
        self.gt_bbox = ...
        self.detected_bbox = ...
    
    
    def load_scene(self):
        # get snapshot, gt_bbox, detected_bbox
        snapshot_path = os.path.join(self.scene_dir,"snapshot_inclusive_merged.json")
        with open(snapshot_path,'r') as f:
            snapshot_data = json.load(f)
        for k in snapshot_data.keys():
            snapshot_img_path = os.path.join(self.scene_dir, results, k)
            snapshot_img = encode_image(snapshot_img_path)
            self.snapshot[k] = {'img': snapshot_img, 'objects': snapshot_data[k]}
        
        
            
        
    
    
    