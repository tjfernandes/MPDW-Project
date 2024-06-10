import os
import numpy as np
import json
import torch
import search
import planllm as llm

from dialog_manager.example.dialog_factory.dialog_manager import DialogManager
from dialog_manager.example.states import *
from dialog_manager.example.events import *


from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)

relevant_intents = [
    "GreetingIntent",
    "IdentifyProcessIntent",
    "OutOfScopeIntent",
    "YesIntent",
    "NoIntent",
    "NextStepIntent",
    "StopIntent",
    "SelectIntent",
]


## Intent Detection
with open("twiz-data/all_intents.json", 'r') as all_intents_json:
    all_intents = json.load(all_intents_json)
        
id_to_intent, intent_to_id = dict(), dict()
for i, intent in enumerate(all_intents):
    id_to_intent[i] = intent
    intent_to_id[intent] = i

tokenizer_name = 'roberta-base' # try 'bert-base-uncased', 'bert-base-cased', 'bert-large-uncased'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSequenceClassification.from_pretrained("NOVA-vision-language/task-intent-detector", num_labels=len(all_intents))


def predict_intent(agent_u, user_u):
    input_encoding = tokenizer.encode_plus(agent_u, user_u, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

    with torch.no_grad():
        logits = model(**input_encoding).logits
        predicted_class_id = logits.argmax(-1).item() # grab the index of the highest scoring output

    return all_intents[predicted_class_id]


def slot_filling(question, context):
    model_name = "deepset/roberta-base-squad2"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question' : question,
        'context' : context
    }

    return nlp(QA_input)
    
def start_new_dialog(client, index_name):
    
    # create the dialog manager
    dialog_manager = DialogManager()

    # define the state manager with the needed variables
    state_manager = {"intent": "",
                    "recipe": {},
                    "candidate_recipes": [],
                    "step": 0,
                    "agent_u": "",
                    "slots": ""}
    
    intent_handlers = {
        "GreetingIntent": handle_greeting,
        "IdentifyProcessIntent": handle_identify_process,
        "SelectIntent": handle_select,
        "NextStepIntent": handle_next_step,
        "StopIntent": handle_stop,
        "YesIntent": handle_yes,
        "NoIntent": handle_no,
        "PreviousStepIntent": handle_previous_step,
    }

    handle_greeting(state_manager, client, index_name)

    
    while True:
        agent_u = state_manager["agent_u"] 
        
        # receive intent from user utterance
        user_u = input('User: ')
        
        intent = predict_intent(agent_u, user_u)
        
        
        state_manager["intent"] = intent
        print("Intent: " + state_manager["intent"])

        # turn intent into event and trigger a transition in the state machine
        event = dialog_manager.event_type(state_manager)
        print("Event: " + event.__name__)
        dialog_manager.trigger(event(), state_manager)

        # Retrieve the handler function for the current intent from the dictionary
        
        # If the intent is not found, use `handle_unknown_intent` as a default
        if intent not in relevant_intents:
            handler_function = handle_unknown_intent
        
        elif intent == "StopIntent":
            handler_function = handle_stop(state_manager, client, index_name)
            
        else:
            slots = slot_filling(agent_u, user_u)
            if slots != "":
                state_manager["slots"] = slots["answer"]
                
            handler_function = intent_handlers.get(intent)

        # Call the handler function
        handler_function(state_manager, client, index_name)
        
        
        
def handle_greeting(state_manager, client, index_name):
    bot_greeting = "BOT: Hello, there! What can I help you with today?" 
    print(bot_greeting)
    state_manager["agent_u"] = bot_greeting
    
def handle_identify_process(state_manager, client, index_name):
    slots = state_manager["slots"]
    top_3_recipes = search.text_query(client, index_name, slots)
    
    agent_u = "BOT: "
    
    # List the recipes found
    if len(top_3_recipes) == 0:
        agent_u += "I'm sorry, I couldn't find any recipes for you. Could you please try again?"
    else:
        agent_u += "I found a few recipes for you. Select one of these options:\n"
        for i, recipe in enumerate(top_3_recipes, start=1):
            state_manager["candidate_recipes"].append(recipe)
            if (recipe["_source"].get("title") is None):
                title = "No title available"
            else:
                title = recipe["_source"]["title"]
                
            if (recipe["_source"].get("description") is None):
                description = "No description available"
            else:
                description = recipe["_source"]["description"]
                
                
            agent_u += f"\t{i}. {title}:\n"
            agent_u += f"\t\tDescription: {description}\n"
            
            # Assuming you want to include ingredients in agent_u as well
            ingredients = [ingredient["name"] for ingredient in recipe["_source"]["ingredients"] if ingredient["name"] is not None]
            agent_u += "\t\tIngredients: " + ", ".join(ingredients) + "\n"
    
    print(agent_u)
    state_manager["agent_u"] = agent_u
                
    
    
def handle_select(state_manager, client, index_name):
    agent_u = "BOT: "   
    
    slots = state_manager["slots"]
    if slots == "":
        agent_u += "I'm sorry, I didn't understand that. Could you please repeat that?"
        print(agent_u)
        state_manager["agent_u"] = agent_u
        return
    
    # print("Slots: " + slots)
    
    elif any(substring in slots.lower() for substring in ["1", "2", "3", "one", "two", "three", "first", "second", "third"]):
        selection_map = {
            "1": 0, "one": 0, "first": 0,
            "2": 1, "two": 1, "second": 1,
            "3": 2, "three": 2, "third": 2
        }
        
        selected_index = selection_map.get(slots.lower())
        if (selected_index is not None and selected_index < len(state_manager["candidate_recipes"])):
            selected_recipe = state_manager["candidate_recipes"][selected_index]
            state_manager["recipe"] = selected_recipe
                
    else:
        for candidate in state_manager["candidate_recipes"]:
            if slots.lower() in candidate["_source"]["title"].lower():
                state_manager["recipe"] = candidate
                
                 
    agent_u += state_manager["recipe"]["_source"]["title"]
    agent_u += "\nAre you sure you want to learn this recipe? (Yes/No)"
    
    print(agent_u)
    state_manager["agent_u"] = agent_u
    
def handle_next_step(state_manager, client, index_name):
    agent_u = "BOT: "
    agent_u += llm.makeRequest("next")
    print(agent_u)
    state_manager["agent_u"] = agent_u
    
def handle_stop(state_manager, client, index_name):
    print("BOT: Alright, Goodbye!")
    return
    
def handle_yes(state_manager, client, index_name):
    agent_u = "BOT: "
    agent_u += llm.start_conversation(state_manager["recipe"])
    print(agent_u)
    state_manager["agent_u"] = agent_u
    
def handle_no(state_manager, client, index_name):
    print("Alright, let me know if you need anything else.")
    
def handle_previous_step(state_manager, client, index_name):
    print("Let's go back to the previous step.")
    
def handle_unknown_intent():
    print("I'm sorry, I didn't understand that. Could you please repeat that?")