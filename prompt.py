import openai
from openai import AzureOpenAI
from PIL import Image
import base64
from io import BytesIO
import os
import time
from typing import Optional
import logging
from two_stage_prompt import *

client = AzureOpenAI(
    azure_endpoint="https://yuncong.openai.azure.com/",
    api_key=os.getenv('AZURE_OPENAI_KEY'),
    api_version="2024-02-15-preview",
)

def format_content(contents):
    formated_content = []
    for c in contents:
        formated_content.append({"type": "text", "text": c[0]})
        if len(c) == 2:
            formated_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{c[1]}",  # yh: previously I always used jpeg format. The internet says that jpeg is smaller in size? I'm not sure.
                        "detail": "high"
                     }
                }
            )
    return formated_content
    
# send information to openai
def call_openai_api(sys_prompt, contents) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user","content": formated_content}
    ]
    while retry_count < max_tries:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",  # model = "deployment_name"
                messages=message_text,
                temperature=0.7,
                max_tokens=4096,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            return completion.choices[0].message.content
        except openai.RateLimitError as e:
            print("Rate limit error, waiting for 60s")
            time.sleep(30)
            retry_count += 1
            continue
        except Exception as e:
            print("Error: ", e)
            time.sleep(60)
            retry_count += 1
            continue

    return None


def format_prefiltering_prompt(
    question,
    class_list,
    top_k = 10,
    image_goal = None
):
    content = []
    sys_prompt = "You are an AI agent in a 3D indoor scene.\n"
    prompt = "The goal of the AI agent is to answer questions about the scene through exploration.\n"
    prompt += "To efficiently solve the problem, you should first rank objects in the scene based on their importance.\n"
    # prompt += "You should rank the objects based on how well they can help you answer the question.\n"
    # prompt += "More important objects should be more helpful in answering the question, and should be ranked higher and first explored.\n"
    # prompt += f"Only the top {top_k} ranked objects should be included in the response.\n"
    # prompt += "If there are not enough objects, you only need to rank the objects and return all of them in ranked order.\n" 
    prompt += "These are the rules for the task.\n"
    # prompt += "RULES:\n"
    prompt += "1. Read through the whole object list.\n"
    prompt += "2. Rank objects in the list based on how well they can help you answer the question.\n"
    prompt += f"3. Reprint the name of top {top_k} objects. "
    prompt += "If there are not enough objects, reprint all of them in ranked order. Each object should be printed on a new line.\n"
    prompt += "4. Do not print any object not included in the list or include any additional information in your response.\n"
    content.append((prompt,))
    #------------------format an example-------------------------
    prompt = "Here is an example of selecting top 3 ranked objects:\n"
    # prompt += "EXAMPLE: select top 3 ranked objects\n"
    prompt += "Question: What can I use to watch my favorite shows and movies?\n"
    prompt += "Following is a list of objects that you can choose, each object one line\n"
    prompt += "painting\nspeaker\nbox\ncabinet\nlamp\ncouch\npillow\ncabinet\ntv\nbook rack\nwall panel\npainting\nstool\ntv stand\n"
    prompt += "Answer: tv\ntv stand\nspeaker\n"
    content.append((prompt,))
    #------------------Task to solve----------------------------
    prompt = f"Following is the concrete content of the task and you should retrieve top {top_k} objects:\n"
    prompt += f"Question: {question}"
    if image_goal is not None:
        content.append((prompt, image_goal))
        content.append(("\n",))
    else:
        content.append((prompt+"\n",))
    prompt = "Following is a list of objects that you can choose, each object one line\n"
    for i, cls in enumerate(class_list):
        prompt += f"{cls}\n"
    prompt += "Answer: "
    content.append((prompt,))
    return sys_prompt,content

def get_prefiltering_classes(
    question,
    seen_classes,
    top_k=10,
    image_goal = None
): 
    prefiltering_sys,prefiltering_content = format_prefiltering_prompt(
        question, sorted(list(seen_classes)), top_k=top_k, image_goal=image_goal)
    logging.info("prefiltering prompt: \n", "".join([c[0] for c in prefiltering_content]))
    response = call_openai_api(prefiltering_sys, prefiltering_content)
    if response is None:
        return []
    # parse the response and return the top_k objects
    selected_classes = response.strip().split('\n')
    selected_classes = [cls for cls in selected_classes if cls in seen_classes]
    selected_classes = selected_classes[:top_k]
    logging.info(f"Prefiltering response: {selected_classes}")
    return selected_classes

def prefilter_snapshot(
    question,
    snapshot,
    objects_infos,
    top_k = 10
):
    seen_classes = set()
    for objects_info in objects_infos.values():
        detected_classes = [obj["class_name"] for obj in objects_info]
        seen_classes.update(set(detected_classes))
    selected_classes = get_prefiltering_classes(
        question, sorted(list(seen_classes)), top_k)
    print(selected_classes)
    # filter snapshots and objects with given classes
    selected_snapshot = {}
    selected_objects_infos = {}
    
    for k in snapshot.keys():
        selected_objects = [obj for obj in objects_infos[k] if obj["class_name"] in selected_classes]
        if len(selected_objects) > 0:
            selected_snapshot[k] = snapshot[k]
            selected_objects_infos[k] = selected_objects
    
    return selected_snapshot, selected_objects_infos

def format_prompt(query, snapshots, objects_infos):
    # Format the prompt
    sys_prompt = "Task: You are an agent in an indoor scene tasked with responding queries by observing the surroundings and exploring the environment. you are required to choose the queried object from a Snapshot.\n"
    content = []
    
    # 1 list basic info
    text = "Definitions:\n"
    text += "Snapshot: A focused observation of several objects. Choosing a snapshot means that you are selecting the observed objects in the snapshot as the target objects the user queried.\n"
    text += "Each snapshot would be followed by a list of object ids and their corresponding descriptions.\n"  
    content.append((text,))
    # 2 here is the query
    text = f"Query: Find the object describe as \'{query}\'\n"
    text += "The followings are all the snapshots that you can explore (followed with contained object ids and their descriptions)\n"
    content.append((text,))
    for i, frame_key in enumerate(snapshots.keys()):
        snapshot = snapshots[frame_key]
        obj_info = objects_infos[frame_key]
        content.append((f"Snapshot {i} ", snapshot))
        obj_text = ""
        for j, obj in enumerate(obj_info):
            obj_text += f"Object {j} {obj['class_name']}: {obj['caption']}\n"
            #obj_text += f"Object {j} {obj['class_name']}, "
        content.append((obj_text,))
    
    text = "Please choose a snapshot and an object from the snapshot to answer the query.\n"
    #text += "To approach this query, you can first choose a snapshot and then choose an object from the snapshot based on visual information and text descriptions.\n"
    text += "The answer should be in the format of 'Snapshot x, Object y', where x is the index of the snapshot and y chosen from the object id following snapshot x\n"
    text += "Explain the reason for your choice, put it in a new line after the choice.\n"
    content.append((text,))
    return sys_prompt, content


def get_predicted_object_id(
    query, 
    snapshots, 
    objects_infos,
    prefiltering = True,
    top_k = 5
    ):
    # filter given objects
    print(query)
    print(f"total objects {sum(len(obj_infos) for obj_infos in objects_infos.values())}")
    if prefiltering:
        snapshots, objects_infos = prefilter_snapshot(query, snapshots, objects_infos, top_k)
        print(f"total objects after prefiltering {sum(len(obj_infos) for obj_infos in objects_infos.values())}")
    # Get the predicted object id
    sys_prompt, content = format_prompt(query, snapshots, objects_infos)
    print(f"the input prompt:\n{sys_prompt + ''.join([c[0] for c in content])}")
    response = call_openai_api(sys_prompt, content)
    
    # parse the response
    
    response = response.strip()
    retry_limit = 3
    snapshot_id, object_id, frame_key = None, None, None
    pred_bbox = None
    while (retry_limit > 0):
        if "\n" in response:
            response = response.split("\n")
            response, reason = response[0], response[-1] 
        else:
            reason = None
        response = response.lower()
        print(response)
        print(f"reason for response {reason}")
        try:
            snapshot_id, object_id = response.split(", ")
            snapshot_id = snapshot_id.split(" ")
            object_id = object_id.split(" ")
        except:
            print("Invalid response, please try again")
            retry_limit -= 1
            continue

        if snapshot_id[0] == "snapshot" and 0 <= int(snapshot_id[1]) < len(snapshots):
            snapshot_id = int(snapshot_id[1])
        else:
            print("Invalid snapshot response, please try again")
            retry_limit -= 1
            continue
        frame_key = list(snapshots.keys())[snapshot_id]
        if object_id[0] == "object" and 0 <= int(object_id[1]) < len(objects_infos[frame_key]):
            object_id = int(object_id[1])
            print("object class name: ", objects_infos[frame_key][object_id]["class_name"])
            pred_bbox = objects_infos[frame_key][object_id]["bbox"]
        else:
            print("Invalid object response, please try again")
            retry_limit -= 1
            continue
        
        break
    
    # get the corresponding object bounding box based on the object id
    return pred_bbox, frame_key

def parse_response_with_reason(response):
    response = response.strip()
    if "\n" in response:
            response = response.split("\n")
            response, reason = response[0], response[-1] 
    else:
        reason = None
    response = response.lower()
    print(response)
    print(f"reason for response {reason}")
    return response

def get_predicted_object_id2(
    query, 
    snapshots, 
    objects_infos,
    prefiltering = True,
    top_k = 5
    ):
    # filter given objects
    print(query)
    print(f"total objects {sum(len(obj_infos) for obj_infos in objects_infos.values())}")
    if prefiltering:
        snapshots, objects_infos = prefilter_snapshot(query, snapshots, objects_infos, top_k)
        print(f"total objects after prefiltering {sum(len(obj_infos) for obj_infos in objects_infos.values())}")
    # Get the predicted object id
    sys_prompt, content = format_snapshot_prompt(query, snapshots, objects_infos)
    print(f"the input prompt:\n{sys_prompt + ''.join([c[0] for c in content])}")
    response = call_openai_api(sys_prompt, content)
    
    # parse the response
    
    response = response.strip()
    retry_limit = 3
    snapshot_id = None
    while (retry_limit > 0):
        response = parse_response_with_reason(response)
        try:
            snapshot_id = response.split(" ")
        except:
            print("Invalid response, please try again")
            retry_limit -= 1
            continue
        
        if snapshot_id[0] == "snapshot" and 0 <= int(snapshot_id[1]) < len(snapshots):
            snapshot_id = int(snapshot_id[1])
            break
        retry_limit -= 1
    
    frame_key = list(snapshots.keys())[snapshot_id]
    snapshot = snapshots[frame_key]
    objects_info = objects_infos[frame_key]
    sys_prompt, content = format_object_prompt(query, snapshot, objects_info)
    
    print(f"the input prompt:\n{sys_prompt + ''.join([c[0] for c in content])}")
    response = call_openai_api(sys_prompt, content)
    
    # parse the response
    
    retry_limit = 3
    object_id = None
    pred_bbox = None
    while (retry_limit > 0):
        response = parse_response_with_reason(response)
        try:
            object_id = response.split(" ")
        except:
            print("Invalid response, please try again")
            retry_limit -= 1
            continue
        
        if object_id[0] == "object" and 0 <= int(object_id[1]) < len(objects_info):
            object_id = int(object_id[1])
            break
        
        retry_limit -= 1
    
    # get the corresponding object bounding box based on the object id
    return objects_info[object_id]["bbox"], frame_key