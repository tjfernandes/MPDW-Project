import os
import requests
import json
import uuid

internal_url = 'http://10.10.255.202:5633'
external_url = "https://twiz.novasearch.org"

max_timeout = 10

def post_request(url: str, conversation: str, text: str):
    url = os.path.join(url, "structured")

    # check this file to understand the structure of the data
    with open(conversation) as f:
        data = json.load(f)

    data = {
        "dialog": data,
        "max_tokens": 100,
        "temperature": 0.0,
        "top_p": 1,
        "top_k": -1,
    }

    # Make the POST request
    response = requests.post(url, json=data, timeout=max_timeout)
    
    if response.status_code == 200:
        print("POST request successful for URL:", url)
        print("Response:", response.text)
    else:
        print("POST request failed with status code:", response.status_code)

def makeLLMFormattedDialog(system_tone, recipe_name, instructions):
    # Generate a unique dialog ID
    dialog_id = str(uuid.uuid4())
    
    # Create the recipe instructions list
    instructions_list = [{"stepText": instruction} for instruction in instructions]

    # Create the dialog list
    dialog_list = []

    # Create the main dictionary
    dialog_dict = {
        "dialog_id": dialog_id,
        "system_tone": system_tone,
        "task": {
            "recipe": {
                "displayName": recipe_name,
                "instructions": instructions_list
            }
        },
        "dialog": dialog_list
    }

    # Convert the dictionary to a JSON string
    dialog_json = json.dumps(dialog_dict, indent=4)

    return dialog_json



# Is this needed? Depends on whether the API can automatically add to the dialog

# def addToDialog(dialog_json, current_step, user, system):
#     # Convert the JSON string back to a dictionary
#     dialog_dict = json.loads(dialog_json)

#     # Add the new dialog to the dialog list
#     dialog_dict["dialog"].append({
#         "current_step": current_step,
#         "user": user,
#         "system": system
#     })

#     # Convert the dictionary back to a JSON string
#     dialog_json = json.dumps(dialog_dict, indent=4)

#     return dialog_json