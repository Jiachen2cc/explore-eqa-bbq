def format_snapshot_prompt(query, snapshots, objects_infos):
    # Format the prompt
    sys_prompt = "Task: You are an agent in an indoor scene tasked with responding queries by observing the surroundings and exploring the environment. you are required to choose the Snapshot that contains queried object.\n"
    content = []
    
    # 1 list basic info
    text = "Definitions:\n"
    text += "Snapshot: A focused observation of several objects. Choosing a snapshot means that you are selecting the observed objects in the snapshot as the target objects the user queried.\n"
    text += "Each snapshot would be followed by a list of contained object class names\n"
    text += "In this stage, you need to first summarize information from all snapshots\n"
    text += "For example, if the user query is \'Find the window next to the door\', but no door and window are covisible in the same snapshot, you need to discover their relation across multiple snapshots\n"
    text += "Then, you should return only ONE snapshot that contains the target object the user queried\n"
    content.append((text,))
    # 2 here is the query
    text = f"Query: Find the object describe as \'{query}\'\n"
    text += "The followings are all the snapshots that you can explore (followed with contained object classes)\n"
    content.append((text,))
    for i, frame_key in enumerate(snapshots.keys()):
        snapshot = snapshots[frame_key]
        obj_info = objects_infos[frame_key]
        content.append((f"Snapshot {i} ", snapshot))
        obj_text = ""
        for j, obj in enumerate(obj_info):
            #obj_text += f"Object {j} {obj['class_name']}: {obj['caption']}\n"
            obj_text += f"{obj['class_name']}, "
        content.append((obj_text,))
    
    text = "Please choose a snapshot to answer the query. A qualified snapshot should be followed by objects required by the user\n"
    text += "The answer should be in the format of 'Snapshot x', where x is the index of the snapshot\n"
    text += "Explain the reason for your choice, put it in a new line after the choice.\n"
    content.append((text,))
    return sys_prompt, content


def format_object_prompt(query, snapshot, objects_infos):
    # Format the prompt
    sys_prompt = "Task: You are an agent in an indoor scene tasked with responding queries by observing the surroundings and exploring the environment. you are required to choose the object that the user queried from current snapshot.\n"
    content = []
    
    # 1 list basic info
    text = "Definitions:\n"
    text += "Snapshot: A focused observation of several objects.\n"
    text += "Object: A specific object in the scene. Choosing an object means that you are selecting the object as the target object the user queried.\n"
    text += "Each object would be followed by its descriptions\n"  
    content.append((text,))
    # 2 here is the query
    text = f"Query: Find the object describe as \'{query}\'\n"
    text += "Following is the current snapshot that you can explore (followed with contained object classes)\n"
    content.append((text, snapshot))
    content.append(("\n",))
    obj_text = "Following are the objects you can choose to answer the question:\n"
    for j, obj in enumerate(objects_infos):
        obj_text += f"Object {j} {obj['class_name']}: {obj['caption']}\n"
    content.append((obj_text,))
        
    text = "Please choose an object to answer the query.\n"
    text += "The answer should be in the format of 'Object x', where x is the index of the object\n"
    text += "Explain the reason for your choice, put it in a new line after the choice.\n"
    content.append((text,))
    return sys_prompt, content