import os
import numpy as np # type: ignore
import json
import torch # type: ignore
import planllm as llm

from dialog_manager.example.dialog_factory.dialog_manager import DialogManager
from dialog_manager.example.states import *
from dialog_manager.example.events import *


from transformers import ( # type: ignore
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)

allowed_intents = [
    "FallbackIntent",
    "IdentifyProcessIntent",
    "MoreOptionsIntent",
    "NextStepIntent",
    "PreviousStepIntent",
    "NoIntent",
    "StopIntent",
    "YesIntent",
    "OutOfScopeIntent",
    "SelectIntent",
    "QuestionIntent",
    "MoreDetailIntent",
]



## Intent Detection setup
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
                    "current_recipe_idx": 0,
                    "step": 0,
                    "agent_u": "",
                    "user_u": "",
                    "slots": "",
                    "client": client,
                    "index_name": index_name}
    
    # Start the dialog
    agent_u = dialog_manager.launch_result["response"]
    state_manager["agent_u"] = agent_u
    print(agent_u)

    # Start the dialog loop
    while True:
        agent_u = state_manager["agent_u"] 
        user_u = input('User: ')
        if user_u == "":
            continue
        state_manager["user_u"] = user_u
        
        intent = predict_intent(agent_u, user_u)
        if intent not in allowed_intents:
            print("Sorry... I'm not sure I can do that.")
            continue
        
        if intent == "IdentifyProcessIntent":
            state_manager["candidate_recipes"] = []
            state_manager["current_recipe_idx"] = 0
        
        state_manager["intent"] = intent
        
        slots = slot_filling(agent_u, user_u)
        if slots != "":
            state_manager["slots"] = slots["answer"]

        # turn intent into event and trigger a transition in the state machine
        event = dialog_manager.event_type(state_manager)
        result = dialog_manager.trigger(event(), state_manager)
        
        agent_u = result["response"]
        print(agent_u)
        
        if result["screen"] == "fallback":
            continue
        
        state_manager["agent_u"] = agent_u
        
        # if current state is GoodbyeState, break the loop
        if dialog_manager.checkpoint[-1][1].__class__.__name__ == "GoodbyeState":
            break