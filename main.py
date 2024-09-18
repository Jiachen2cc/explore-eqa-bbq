import argparse
from scene import Scene
import os
from utils import iou_bbox
from prompt import get_predicted_object_id


def main():
    
    # 1 set up relevant arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_data_dir', type=str, default='data',
                        help = "the source directory of the scene data")
    parser.add_argument('--query_data_dir', type=str, default='data',
                        help = "the source directory of the query data")
    parser.add_argument('--bbox_data_dir', type=str, default='data',
                        help = "the source directory of the groundtruth bbox data")
    parser.add_argument('--snapshot_data_dir', type=str, default='data',
                        help = "the source directory of the snapshot data")
    
    # 2 construct the scene
    for scene_dir in os.listdir(args.scene_data_dir):
        scene = Scene(scene_dir)
        scene.load_scene()
        snapshot = ...
        object_info = ...
    
        # 3 get query, snapshot, object_id and format the prompt
        query = ...
        
        # 4 send prompt to LLM, get the predicted object id
        pred_id = get_predicted_object_id(query, snapshot)
        
        # 5 extract the predicted bbox and compare with gt_bbox
        pred_bbox = ...
        tar_bbox = ...
        
        # 6 store the results for this query
        if iou_bbox(pred_bbox, tar_bbox):
            success = True
            ...
    
        # 7 summary the results for all queries
    
    return 


if __name__ == "__main__":
    main()