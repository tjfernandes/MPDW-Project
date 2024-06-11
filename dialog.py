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

relevant_events = [
    "GreetingIntent",
    "IdentifyProcessIntent",
    "OutOfScopeIntent",
    "YesIntent",
    "NoIntent",
    "NextStepIntent",
    "StopIntent",
    "SelectIntent",
    "StartStepsIntent",
    "LastStepEvent",
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
                    "slots": "",
                    "client": client,
                    "index_name": index_name}
    
    intent_handlers = {
        "StopIntent": handle_stop,
        "NoIntent": handle_no,
        "PreviousStepIntent": handle_previous_step,
        "LastStepEvent": handle_last_step,
    }
    
    agent_u = "BOT: " + dialog_manager.launch_result["response"]
    state_manager["agent_u"] = agent_u
    print(agent_u)

    while True:
        agent_u = state_manager["agent_u"] 
        user_u = input('User: ')
        
        intent = predict_intent(agent_u, user_u)
        state_manager["intent"] = intent
        
        slots = slot_filling(agent_u, user_u)
        if slots != "":
            state_manager["slots"] = slots["answer"]

        # turn intent into event and trigger a transition in the state machine
        event = dialog_manager.event_type(state_manager)
        result = dialog_manager.trigger(event(), state_manager)
            
        agent_u = result["response"]
        print(agent_u)
        state_manager["agent_u"] = agent_u
    
def handle_stop(state_manager, client, index_name):
    print("BOT: Alright, Goodbye!")
    return
    
def handle_no(state_manager, client, index_name):
    agent_u = "BOT: "
    agent_u += "Alright, let me know if you need anything else."
    print(agent_u)
    state_manager["agent_u"] = agent_u
    
def handle_previous_step(state_manager, client, index_name):
    print("Let's go back to the previous step.")
    
def handle_last_step(state_manager, client, index_name):
    agent_u = "BOT: "
    agent_u += llm.make_request("next")
    agent_u += "Do you want to learn more recipes? (Yes/No)"
    print(agent_u)
    state_manager["agent_u"] = agent_u
    
def handle_unknown_intent():
    print("I'm sorry, I didn't understand that. Could you please repeat that?")