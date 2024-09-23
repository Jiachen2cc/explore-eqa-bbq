import argparse
from scene import Scene
import os
from utils import *
from prompt import get_predicted_object_id, get_predicted_object_id2
import json
from tqdm import tqdm
from omegaconf import OmegaConf
from statistics import merge_results


def main(cfg):
    print(f"=====================Start {cfg.exp_name}=====================")
    with open(cfg.bbox_data_dir, 'r') as f:
        bbox_data = json.load(f)
        
    overall_record = {}
    cfg.query_data_dir = f"{cfg.query_data_dir}/{cfg.query_set}_all_types_of_queries"
    # 2 construct the scene
    for scene_dir in os.listdir(cfg.scene_data_dir):
        scene_id = scene_dir.split('-')[-1]
        scene_out_dir = os.path.join(cfg.output_dir, scene_id)
        if not os.path.exists(scene_out_dir):
            os.makedirs(scene_out_dir)
        print(f"=====================Scene {scene_id}=====================")
        scene_record = initialize_count(cfg.query_set)
        gt_bbox = bbox_data[scene_id]
        scene = Scene(os.path.join(cfg.scene_data_dir, scene_dir),
                cfg.snapshot_file,
                os.path.join(cfg.frame_dir, scene_dir),
                os.path.join(cfg.annotation_dir, scene_id),
                os.path.join(cfg.output_dir, scene_id, cfg.visualization_dir))
        scene.load_scene()
    
        # 3 get query, snapshot, object_id and format the prompt
        with open(os.path.join(cfg.query_data_dir, f'{scene_id}_annotation.json'), 'r') as f:
            query_data = json.load(f)
        # 4 send prompt to LLM, get the predicted object id
        success_count, query_count = 0, 0
        for query in tqdm(query_data):
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
            if pred_bbox is not None:
                if cfg.save_visualization:
                    scene.check_annonation(frame_key, 
                            target_id, gt_bbox[str(target_id)]['label'],
                            query_count, query['utterance'])
                    
                # 5 extract the predicted bbox and compare with gt_bbox
                tar_bbox = gt_bbox[str(target_id)]['bbox']
                if not cfg.bbox_oriented:
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
                iou_score = iou_bbox(pred_bbox, tar_bbox, cfg.bbox_oriented)
            else:
                iou_score = 0
            scene_record = update_count(scene_record, iou_score, query)
            scene_record['acc01'] = scene_record['succ01_count']/(scene_record['overall_count']+1e-06)
            scene_record['acc025'] = scene_record['succ025_count']/(scene_record['overall_count']+1e-06)
            print('===================SUMMARY====================')
            print(scene_record)
            #scene_record.to_csv(f'{scene_id}_record.csv')
            scene_record.to_csv(os.path.join(scene_out_dir, 'record.csv'))
            
    
        # 7 summary the results for all queries
        overall_record[scene_id] = scene_record
        overall_result = merge_results(cfg.output_dir)
        print('===================OVERALL SUMMARY====================')
        print(overall_result)
        
        
    return overall_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", 
                        default="cfg/test_nr3d.yaml", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    main(cfg)