class Scene:
    
    def __init__(self, scene_dir):
        self.scene_dir = scene_dir
        self.snapshot = ...
        self.gt_bbox = ...
        self.detected_bbox = ...
    
    
    def load_scene(self):
        # get snapshot, gt_bbox, detected_bbox
        ...
    
    
    