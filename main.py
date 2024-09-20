import argparse
from scene import Scene
import os
from utils import *
from prompt import get_predicted_object_id
import json


def main():
    
    # 1 set up relevant arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_data_dir', type=str, 
                        default='/project/pi_chuangg_umass_edu/yuncong/scene_data_scannet_retrieval/',
                        help = "the source directory of the scene data")
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
    
    args = parser.parse_args()
    with open(args.bbox_data_dir, 'r') as f:
        bbox_data = json.load(f)
    # 2 construct the scene
    for scene_dir in os.listdir(args.scene_data_dir):
        scene = Scene(os.path.join(args.scene_data_dir, scene_dir))
        scene.load_scene()
        scene_id = scene_dir.split('-')[-1]
        gt_bbox = bbox_data[scene_id]
    
        # 3 get query, snapshot, object_id and format the prompt
        with open(os.path.join(args.query_data_dir, f'{scene_id}_annotation.json'), 'r') as f:
            query_data = json.load(f)
        # 4 send prompt to LLM, get the predicted object id
        for query in query_data:
            target_id = query['target_id']
            print(f"Query: {query['utterance']}")
            print(f"Target object id: {target_id}")
            pred_bbox = get_predicted_object_id(query['utterance'], scene.snapshot, scene.snapshot_objects)
            print(pred_bbox)
            #pred_bbox = list(scene.snapshot_objects.values())[0][0]['bbox']
            #print(pred_bbox)
            
            # 5 extract the predicted bbox and compare with gt_bbox
            tar_bbox = gt_bbox[str(target_id)]['bbox']
            print(tar_bbox)
            if not args.bbox_oriented:
                pred_bbox = oriented_to_raw_bbox(pred_bbox)
                tar_bbox = center_bbox_to_raw_bbox(tar_bbox)
                print(pred_bbox)
                print(tar_bbox)
            # 6 store the results for this query
            if iou_bbox(pred_bbox, tar_bbox) > 0.5:
                success = True
                ...
            exit(0)
    
        # 7 summary the results for all queries
        break
    return 


if __name__ == "__main__":
    main()