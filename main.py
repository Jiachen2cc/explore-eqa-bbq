import argparse
from scene import Scene
import os
from utils import *
from prompt import get_predicted_object_id, get_predicted_object_id2
import json
from tqdm import tqdm
from omegaconf import OmegaConf


def main(cfg):
    
    with open(cfg.bbox_data_dir, 'r') as f:
        bbox_data = json.load(f)
    # 2 construct the scene
    for scene_dir in os.listdir(cfg.scene_data_dir):
        scene_id = scene_dir.split('-')[-1]
        if scene_id != "scene0030_00":
            continue
        gt_bbox = bbox_data[scene_id]
        scene = Scene(os.path.join(cfg.scene_data_dir, scene_dir),
                os.path.join(cfg.frame_dir, scene_dir),
                os.path.join(cfg.annotation_dir, scene_id),
                os.path.join(cfg.output_dir, cfg.visualization_dir, scene_id))
        scene.load_scene()
    
        # 3 get query, snapshot, object_id and format the prompt
        
        with open(os.path.join(cfg.query_data_dir, f'{scene_id}_annotation.json'), 'r') as f:
            query_data = json.load(f)
        # 4 send prompt to LLM, get the predicted object id
        success_count, query_count = 0, 0
        for query in tqdm(query_data[0:10]):
            query_count += 1
            target_id = query['target_id']
            print(f"Query: {query['utterance']}")
            print(f"Target object id: {target_id}")
            '''
            scene.get_possible_query_answer(
                query['utterance'], query_count,
                target_id, gt_bbox[str(target_id)]['label']
            )
            '''
            pred_bbox, frame_key = get_predicted_object_id(
                query['utterance'], scene.snapshot, scene.snapshot_objects, 
                True, 5
            )
            if cfg.save_visualization:
                scene.check_annonation(frame_key, 
                        target_id, gt_bbox[str(target_id)]['label'],
                        query_count, query['utterance'])
                
            # 5 extract the predicted bbox and compare with gt_bbox
            tar_bbox = gt_bbox[str(target_id)]['bbox']
            if not args.bbox_oriented:
                pred_bbox = oriented_to_raw_bbox(pred_bbox)
                tar_bbox = center_bbox_to_raw_bbox(tar_bbox)
                print("predicted_object: ", pred_bbox)
                print("target_object: ", tar_bbox)
            target_class = gt_bbox[str(target_id)]['label']
            print("Target class: ", target_class)
            '''
            for gt_id in gt_bbox.keys():
                if gt_bbox[gt_id]['label'] == target_class:
                    print("possible groundtruth_object: ", center_bbox_to_raw_bbox(gt_bbox[gt_id]['bbox']))
            for fkey in scene.snapshot.keys():
                for obj in scene.snapshot_objects[fkey]:
                    if obj['class_name'] == 'chair': #gt_bbox[str(target_id)]['label']:
                        print("candidate_detected_object: ", oriented_to_raw_bbox(obj['bbox']))
            '''
            # 6 store the results for this query 
            if iou_bbox(pred_bbox, tar_bbox) > 0.1:
                success_count += 1
            print("Success rate: ", success_count/query_count)
            
    
        # 7 summary the results for all queries
        break
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", 
                        default="cfg/test.yaml", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    main(cfg)