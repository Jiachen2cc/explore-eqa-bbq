import json
from utils import encode_image
import os 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
class Scene:
    
    def __init__(self, 
                scene_dir,
                snapshot_file,
                frame_dir, 
                annotation_dir,
                visualization_dir = 'debug'):
        self.scene_dir = scene_dir
        self.snapshot_file = snapshot_file
        self.frame_dir = frame_dir
        self.annotation_dir = annotation_dir
        self.snapshot = {}
        self.snapshot_objects = {}
        self.visualization_dir = visualization_dir
        if not os.path.exists(self.visualization_dir):
            os.makedirs(self.visualization_dir)
    
        #self.gt_bbox = ...
        #self.detected_bbox = ...
    
    
    def load_scene(self):
        # get snapshot, gt_bbox, detected_bbox
        snapshot_path = os.path.join(self.scene_dir,self.snapshot_file)
        with open(snapshot_path,'r') as f:
            snapshot_data = json.load(f)
        for frame_key in snapshot_data.keys():
            frame_id = frame_key[5:-4]
            snapshot_img_path = os.path.join(self.frame_dir,f"{frame_id}-rgb.png")
            snapshot_img = encode_image(snapshot_img_path)
            self.snapshot[frame_key] = snapshot_img
            self.snapshot_objects[frame_key] = snapshot_data[frame_key]
        
    
    def check_annonation(self,
        frame_key, target_id, target_class,
        query_index, query):
        frame_id = frame_key[5:-4]
        frame_path = os.path.join(self.frame_dir,f"{frame_id}-rgb.png")
        detection_path = os.path.join(self.scene_dir,'gsa_vis_ram_withoutbg_allclasses_stride10',frame_key)
        annotation_path = os.path.join(self.annotation_dir,'instance-filt',f"{int(frame_id)}.png")
        
        frame_img = mpimg.imread(frame_path)
        detection_img = mpimg.imread(detection_path)
        annotation_img = Image.open(annotation_path)
        annotation_img = (np.array(annotation_img) == (target_id + 1)).astype(np.uint8)
        
        fig, axes = plt.subplots(1,3,
                    figsize=(15, 5), dpi = 600)
        axes[0].imshow(frame_img)
        axes[0].axis('off')
        axes[0].set_title('Frame Image')
        
        axes[1].imshow(detection_img)
        axes[1].axis('off')
        axes[1].set_title('Detection Image')
        
        axes[2].imshow(annotation_img)
        axes[2].axis('off')
        axes[2].set_title('Annotation Image')
        
        # Adjust layout and save the figure
        plt.suptitle(f'target id {target_id}, class {target_class}\n {query}', fontsize=16)
        plt.tight_layout()
        
        save_dir = os.path.join(self.visualization_dir,f"query{query_index}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
        save_path = os.path.join(save_dir, f"snapshot_chosen_{frame_key}")
        plt.savefig(save_path, bbox_inches='tight', 
            pad_inches=0.1, dpi = 600)
        plt.close()
        print(f"Combined image saved to {save_path}")
        
    
    def get_possible_query_answer(
        self, query, query_index,
        target_id, target_class):
        # find all snapshots intersected with target index
        candidate_count = 0
        for frame_key in self.snapshot.keys():
            frame_id = frame_key[5:-4]
            frame_path = os.path.join(self.frame_dir,f"{frame_id}-rgb.png")
            detection_path = os.path.join(self.scene_dir,'gsa_vis_ram_withoutbg_allclasses_stride10',frame_key)
            annotation_path = os.path.join(self.annotation_dir,'instance-filt',f"{int(frame_id)}.png")
            
            annotation_img = Image.open(annotation_path)
            annotation_img = (np.array(annotation_img) == (target_id+1)).astype(np.uint8)
            if np.sum(annotation_img) > 0:
                candidate_count += 1
                detection_img = mpimg.imread(detection_path)
                frame_img = mpimg.imread(frame_path)
                fig, axes = plt.subplots(1,3,
                    figsize=(15, 5), dpi = 600)
                axes[0].imshow(frame_img)
                axes[0].axis('off')
                axes[0].set_title('Frame Image')
                
                axes[1].imshow(detection_img)
                axes[1].axis('off')
                axes[1].set_title('Detection Image')
                
                axes[2].imshow(annotation_img)
                axes[2].axis('off')
                axes[2].set_title('Annotation Image')
                
                # Adjust layout and save the figure
                plt.suptitle(f'target id {target_id}, class {target_class}\n {query}', fontsize=16)
                plt.tight_layout()

                save_dir = os.path.join(self.visualization_dir,f"query{query_index}")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                save_path = os.path.join(save_dir, f"query_candidate_{frame_key}")
                plt.savefig(save_path, bbox_inches='tight', 
                    pad_inches=0.1, dpi = 600)
                plt.close()
                print(f"Combined image saved to {save_path}")
                
        if candidate_count == 0:
            print(f"No candidate found for target id {target_id}, class {target_class}")
            print(f"ERROR: Query can not be addressed due to data error")
        
        
        
        
        
        
        
            
        
    
    
    