import openai
from openai import AzureOpenAI
from PIL import Image
import base64
from io import BytesIO
import os
import time
from typing import Optional
import logging

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


def format_prompt(query, snapshots, objects_infos):
    # Format the prompt
    sys_prompt = "Task: You are an agent in an indoor scene tasked with answering queries by observing the surroundings and exploring the environment. To answer the question, you are required to choose an object from a Snapshot.\n"
    content = []
    
    # 1 list basic info
    text = "Definitions:\n"
    text += "Snapshot: A focused observation of several objects. Choosing a snapshot means that you are selecting the observed objects in the snapshot as the target objects to help answer the question.\n"
    text += "Each snapshot would be followed by a list of object ids and their corresponding descriptions.\n"  
    content.append((text,))
    # 2 here is the query
    text += f"Query: {query}\n"
    text += "The followings are all the snapshots that you can explore (followed with contained object ids and descriptions)\n"
    
    for i, (snapshot, obj_info) in enumerate(zip(snapshots, objects_infos)):
        content.append(f"Snapshot {i} ", snapshot_imgs[i])
        # may be revised based on concrete implementation
        text = ", ".join(obj_info)
        content.append((text,))
        content.append(("\n",))
    
    text += "Please choose a snapshot and an object id from the snapshot to help answer the question.\n"
    text += "The answer should be in the format of 'Snapshot x, Object y', where x is the index of the snapshot and y chosen from the object id following snapshot x\n"
    text += "You can explain the reason for your choice, but put it in a new line after the choice.\n"
    content.append((text,))
    return sys_prompt, content


def get_predicted_object_id(query, snapshots):
    # Get the predicted object id
    sys_prompt, contents = format_prompt(query, snapshots)
    response = call_openai_api(sys_prompt, contents)
    
    # parse the response
    
    response = response.strip()
    retry_limit = 3
    object_id = None
    for _ in range(retry_limit):
        if "\n" in response:
            response = response.split("\n")
            response, reason = response[0], response[1] 
        else:
            reason = None
        response = response.lower()
        try:
            snapshot_id, object_id = response.split(", ")
            snapshot_id = int(snapshot_id.split(" ")[-1])
            object_id = int(object_id.split(" ")[-1])
        
        # may add some validation check here
        if object_id is not None:
            break
        
    return object_id