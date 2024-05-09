import os
import requests
import json

external_url = "https://twiz.novasearch.org"
max_timeout = 10

def get_new_id():
    # Check if the file exists
    if os.path.exists('dialogs.json'):
        # If the file exists, open it and load the data
        with open('dialogs.json', 'r') as f:
            dialogs = json.load(f)
            
            return len(dialogs)
    else:
        return 0

def start_conversation(recipe, tone):
    dialog_id = get_new_id()
    recipe_name = recipe['_source']['title']
    
    # Extract the instructions
    instructions = [{'stepText': step['text']} for step in recipe['_source']['instructions']]
    
    # Define the dictionary
    dialog = {
        "dialog_id": str(dialog_id),
        "system_tone": tone,
        "task": {
            "recipe": {
                "displayName": recipe_name,
                "instructions": instructions
            }
        },
        "dialog": [
            
        ]
    }
    
    print("Dialog id: ", dialog['dialog_id'])
    makeRequest(dialog, "Let's start this recipe!")
    
    
def makeRequest(dialog, text):
    url = os.path.join(external_url, "structured")
    
    if len(dialog['dialog']) > 0:
        last_dialog = dialog['dialog'][-1]
        current_step = int(last_dialog['current_step'])
        if 'Step' in last_dialog['system']:
            current_step += 1
    else:
        current_step = 0
    
    dialog['dialog'].append({
        "current_step": current_step,
        "user": text,
    })
    
    data = {
        "dialog": dialog,
        "max_tokens": 100,
        "temperature": 0.8,
        "top_p": 1,
        "top_k": -1,
    }

    # Make the POST request
    response = requests.post(url, json=data, timeout=max_timeout)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("POST request successful for URL:", url)
        print("Response text:", response.text)
    
        dialog['dialog'][-1]["system"] = response.text
        
        
        # Check if the file exists
        if os.path.exists('dialogs.json'):
            # If the file exists, open it and load the data
            with open('dialogs.json', 'r') as f:
                dialogs = json.load(f)
        else:
            # If the file doesn't exist, create a new list of dialogs
            dialogs = []

        # Try to find the dialog with the matching dialog_id
        for i, existing_dialog in enumerate(dialogs):
            if existing_dialog['dialog_id'] == dialog['dialog_id']:
                # If found, update the dialog
                dialogs[i] = dialog
                break
        else:
            # If not found, append the new dialog
            dialogs.append(dialog)

        # Write the dialogs back to the file
        with open('dialogs.json', 'w') as f:
            json.dump(dialogs, f)
    else:
        print("POST request failed with status code:", response.status_code)
        
        
        
def get_dialog_by_id(dialog_id):
    # Check if the file exists
    if os.path.exists('dialogs.json'):
        # If the file exists, open it and load the data
        with open('dialogs.json', 'r') as f:
            dialogs = json.load(f)
            
            # Iterate over the dialogs
            for dialog in dialogs:
                # If the dialog_id matches, return the dialog
                if dialog['dialog_id'] == dialog_id:
                    return dialog

    # If the file doesn't exist or the dialog_id wasn't found, return None
    return None        