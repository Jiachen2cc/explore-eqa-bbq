import json
from utils import encode_image
import os 
class Scene:
    
    def __init__(self, scene_dir):
        self.scene_dir = scene_dir
        self.snapshot = {}
        self.snapshot_objects = {}
        #self.gt_bbox = ...
        #self.detected_bbox = ...
    
    
    def load_scene(self):
        # get snapshot, gt_bbox, detected_bbox
        snapshot_path = os.path.join(self.scene_dir,"snapshots_inclusive_merged.json")
        with open(snapshot_path,'r') as f:
            snapshot_data = json.load(f)
        for k in snapshot_data.keys():
            snapshot_img_path = os.path.join(self.scene_dir,'gsa_vis_ram_withoutbg_allclasses_stride10',k)
            snapshot_img = encode_image(snapshot_img_path)
            self.snapshot[k] = snapshot_img
            self.snapshot_objects[k] = snapshot_data[k]
        
        
            
        
    
    
    