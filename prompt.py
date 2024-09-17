def call_openai_api(contents):
    # Call OpenAI API
    return "This is the response from OpenAI API"


def format_prompt(query, snapshots):
    # Format the prompt
    return "This is the prompt"


def get_predicted_object_id(query, snapshots):
    # Get the predicted object id
    sys_prompt, contents = format_prompt(query, snapshots)
    response = call_openai_api(contents)
    
    # parse the response
    pred_obj_id = ...
    return "This is the predicted object id"