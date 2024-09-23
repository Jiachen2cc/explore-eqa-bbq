import pandas as pd 
import json
import os

def extract_scene_id(stimulus_id):
    return stimulus_id.split('-')[0]
def extract_object_class(stimulus_id):
    object_class = stimulus_id.split('-')[1]
    object_class = object_class.replace('_', ' ')
    return object_class

def check_nr3d():
    df = pd.read_csv('query/scannet_eval/nr3d_all_types_of_queries/nr3d_filtered.csv')
    # Apply the custom function to the DataFrame
    df['scene_id'] = df['stimulus_id'].apply(extract_scene_id)
    df['object_class'] = df['stimulus_id'].apply(extract_object_class)

    with open('/project/pi_chuangg_umass_edu/yuncong/OpenEqa/scannet/scannet_bbox_clean.json', 'r') as f:
        bbox_data = json.load(f)
        
    scene_ids = ["0011_00", "0030_00", "0046_00", "0086_00", "0222_00", "0378_00", "0389_00", "0435_00"]
    for sid in scene_ids:
        sid = f"scene{sid}"
        sub_df = df[df['scene_id'] == sid]
        print(len(sub_df))
        #print(sub_df['target_id'])
        instance_target_dict = sub_df.set_index('target_id')['instance_type'].to_dict()
        subbbox = bbox_data[sid]
        for target_id in instance_target_dict.keys():
            target_id = str(target_id)
            if target_id not in subbbox.keys():
                print(f"sid")
                print(f"Target ID {target_id} not in bbox data")
                print(subbbox.keys())
                exit(0)
            else:
                gt_class = subbbox[target_id]['label']
                nr3d_class = instance_target_dict[int(target_id)]
                print(f"{sid} {target_id} scannet {gt_class}: nr3d {nr3d_class}")
                if gt_class != nr3d_class:
                    print(f"Target ID {target_id} not consistent")
                    print(f"scannet label: {subbbox[target_id]['label']}")
                    print(f"nr3d label: {instance_target_dict[int(target_id)]}")
                    exit(0)

def check_sr3d():
    df = pd.read_csv('query/scannet_eval/sr3d+.csv')
    # Apply the custom function to the DataFrame
    with open('/project/pi_chuangg_umass_edu/yuncong/OpenEqa/scannet/scannet_bbox_clean.json', 'r') as f:
        bbox_data = json.load(f)
        
    scene_ids = ["0011_00", "0030_00", "0046_00", "0086_00", "0222_00", "0378_00", "0389_00", "0435_00"]
    for sid in scene_ids:
        sid = f"scene{sid}"
        sub_df = df[df['scan_id'] == sid]
        print(len(sub_df))
        #print(sub_df['target_id'])
        instance_target_dict = sub_df.set_index('target_id')['instance_type'].to_dict()
        subbbox = bbox_data[sid]
        for target_id in instance_target_dict.keys():
            target_id = str(target_id)
            if target_id not in subbbox.keys():
                print(f"sid")
                print(f"Target ID {target_id} not in bbox data")
                print(subbbox.keys())
                exit(0)
            else:
                gt_class = subbbox[target_id]['label']
                nr3d_class = instance_target_dict[int(target_id)]
                print(f"{sid} {target_id} scannet {gt_class}: sr3d+ {nr3d_class}")
                if gt_class != nr3d_class:
                    print(f"Target ID {target_id} not consistent")
                    print(f"scannet label: {subbbox[target_id]['label']}")
                    print(f"sr3d+ label: {instance_target_dict[int(target_id)]}")
                    exit(0)

def merge_results(meta_folder):
    
    records = []
    count_key = ['succ01_count', 'succ025_count', 'overall_count']
    for scene in os.listdir(meta_folder):
        if not os.path.isdir(os.path.join(meta_folder, scene)):
            continue
        scene_record = pd.read_csv(os.path.join(meta_folder, scene, 'record.csv'))
        scene_record['scene_id'] = scene
        records.append(scene_record)
    sum_records = records[0].copy()
    for record in records[1:]:
        sum_records[count_key]  += record[count_key]
    sum_records['acc01'] = sum_records['succ01_count'] / sum_records['overall_count']
    sum_records['acc025'] = sum_records['succ025_count'] / sum_records['overall_count']
    sum_records['scene_id'] = 'overall'
    
    records.append(sum_records)
    df = pd.concat(records)
    df.to_csv(os.path.join(meta_folder, 'meta_record.csv'))
    
    return sum_records

def reformat_sr3d()
        
    

if __name__ == "__main__":
    #check_nr3d()
    #check_sr3d()
    
    merge_results('/work/pi_chuangg_umass_edu/yuncong/results/bbq_debug')
    
    
