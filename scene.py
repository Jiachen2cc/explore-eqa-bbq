import json
from utils import encode_image
import os 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
class Scene:
    
    def __init__(self, 
                scene_dir, 
                annotation_dir):
        self.scene_dir = scene_dir
        self.annotation_dir = annotation_dir
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
        
    
    def check_annonation(self,frame_key,save_dir = 'debug'):
        frame_index = int(frame_key[5:-4])
        frame_path = os.path.join(self.scene_dir,'gsa_vis_ram_withoutbg_allclasses_stride10',frame_key)
        annotation_path = os.path.join(self.annotation_dir,'instance-filt',f"{frame_index}.png")
        
        frame_img = mpimg.imread(frame_path)
        annotation_img = mpimg.imread(annotation_path)
        
        fig, axes = plt.subplots(1,2,
                    figsize=(10, 5), dpi = 600)
        axes[0].imshow(frame_img)
        axes[0].axis('off')
        axes[0].set_title('Frame Image')

        axes[1].imshow(annotation_img)
        axes[1].axis('off')
        axes[1].set_title('Annotation Image')
        
        # Adjust layout and save the figure
        plt.tight_layout()
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
        save_path = os.path.join(save_dir, frame_key)
        plt.savefig(save_path, bbox_inches='tight', 
            pad_inches=0.1, dpi = 600)
        plt.close()
        print(f"Combined image saved to {save_path}")
        
        exit(0)
        
        
        
        
        
        
        
            
        
    
    
    