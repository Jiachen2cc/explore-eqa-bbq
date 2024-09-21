import argparse
from scene import Scene
import os
from utils import *
from prompt import get_predicted_object_id, get_predicted_object_id2
import json
from tqdm import tqdm


def main():
    
    # 1 set up relevant arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_data_dir', type=str, 
                        default='/project/pi_chuangg_umass_edu/yuncong/scene_data_scannet_retrieval/',
                        help = "the source directory of the scene data")
    parser.add_argument("--frame_dir", type = str,
                        default='/project/pi_chuangg_umass_edu/yuncong/OpenEqa/scannet/frames_bbq/',
                        help = "the directory for the frames")
    parser.add_argument('--query_data_dir', type=str, 
                        default='query/scannet_eval/nr3d_all_types_of_queries',
                        help = "the source directory of the query data")
    parser.add_argument('--bbox_data_dir', type=str, 
                        default='/project/pi_chuangg_umass_edu/yuncong/OpenEqa/scannet/scannet_bbox.json',
                        help = "the source directory of the groundtruth bbox data")
    parser.add_argument('--snapshot_data_dir', type=str, default='data',
                        help = "the source directory of the snapshot data")
    parser.add_argument('--bbox_oriented',
                        action='store_true',
                        help='whether to use oriented bounding box')
    parser.add_argument('--annotation_dir', type = str,
                        default = '/project/pi_chuangg_umass_edu/yuncong/OpenEqa/scannet/scans')
    parser.add_argument('--visualization_dir', type = str,
                        default = '/project/pi_chuangg_umass_edu/yuncong/results/bbq_debug')
    
    args = parser.parse_args()
    with open(args.bbox_data_dir, 'r') as f:
        bbox_data = json.load(f)
    # 2 construct the scene
    for scene_dir in os.listdir(args.scene_data_dir):
        scene_id = scene_dir.split('-')[-1]
        if scene_id != "scene0030_00":
            continue
        gt_bbox = bbox_data[scene_id]
        scene = Scene(os.path.join(args.scene_data_dir, scene_dir),
                os.path.join(args.frame_dir, scene_dir),
                os.path.join(args.annotation_dir, scene_id))
        scene.load_scene()
    
        # 3 get query, snapshot, object_id and format the prompt
        
        with open(os.path.join(args.query_data_dir, f'{scene_id}_annotation.json'), 'r') as f:
            query_data = json.load(f)
        # 4 send prompt to LLM, get the predicted object id
        success_count, query_count = 0, 0
        for query in tqdm(query_data):
            query_count += 1
            target_id = query['target_id']
            print(f"Query: {query['utterance']}")
            print(f"Target object id: {target_id}")
            pred_bbox, frame_key = get_predicted_object_id(
                query['utterance'], scene.snapshot, scene.snapshot_objects, 
                True, 5
            )
            scene.check_annonation(frame_key, 
                    target_id, gt_bbox[str(target_id)]['label'],
                    query_count, query['utterance'])
            #print(pred_bbox)
            #pred_bbox = list(scene.snapshot_objects.values())[0][0]['bbox']
            #print(pred_bbox)
            
            # 5 extract the predicted bbox and compare with gt_bbox
            tar_bbox = gt_bbox[str(target_id)]['bbox']
            if not args.bbox_oriented:
                pred_bbox = oriented_to_raw_bbox(pred_bbox)
                tar_bbox = center_bbox_to_raw_bbox(tar_bbox)
                print("predicted_object: ", pred_bbox)
                print("target_object: ", tar_bbox)
            target_class = gt_bbox[str(target_id)]['label']
            print("Target class: ", target_class)
            for gt_id in gt_bbox.keys():
                if gt_bbox[gt_id]['label'] == target_class:
                    print("possible groundtruth_object: ", center_bbox_to_raw_bbox(gt_bbox[gt_id]['bbox']))
            for fkey in scene.snapshot.keys():
                for obj in scene.snapshot_objects[fkey]:
                    if obj['class_name'] == 'chair': #gt_bbox[str(target_id)]['label']:
                        print("candidate_detected_object: ", oriented_to_raw_bbox(obj['bbox']))
            # 6 store the results for this query 
            if iou_bbox(pred_bbox, tar_bbox) > 0.1:
                success_count += 1
            print("Success rate: ", success_count/query_count)
            
    
        # 7 summary the results for all queries
        break
    return 


if __name__ == "__main__":
    main()