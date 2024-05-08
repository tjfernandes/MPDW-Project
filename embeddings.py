from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pickle
import os
import json

# Import custom modules
import index_management as im

#Mean Pooling - Take average of all tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


#Encode text
def encode(texts):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings      
        
def get_embeddings():
    if os.path.exists('embeddings.pkl'):
        # Load embeddings from a file
        print('Embeddings file found. Loading embeddings...')
        with open('embeddings.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        print('Embeddings file not found. Generating embeddings...')
        return add_embeddings()
    
def add_embeddings():
    titles = []
    descriptions = []
    times = []
    difficulties = []
    # ingredients = []
    # instructions = []
    
    for recipe in im.recipes_data.values():
        titles.append(recipe["displayName"] if recipe["displayName"] is not None else 'None') 
        descriptions.append(recipe["description"] if recipe["description"] is not None else 'None')
        times.append(str(recipe["totalTimeMinutes"]) if recipe["totalTimeMinutes"] is not None else 'None')        
        difficulties.append(recipe["difficultyLevel"] if recipe["difficultyLevel"] is not None else 'None')
        # for ingredient in recipe["ingredients"]:
        #     ingredients.append(ingredient if ingredient is not None else 'None')
        # for instruction in recipe["instructions"]:
        #     instructions.append(instruction["stepText"] if instruction["stepText"] is not None else 'None')
        
    # Calculate embeddings
    titles_emb = encode(titles)
    descriptions_emb = encode(descriptions)
    times_emb = encode(times)
    difficulties_emb = encode(difficulties)
    # ingredients_emb = encode(ingredients)
    # instructions_emb = encode(instructions)

    # Save embeddings to a file
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump({
            'titles': titles_emb,
            'descriptions': descriptions_emb,
            'times': times_emb,
            'difficulties': difficulties_emb,
            # 'ingredients': ingredients_emb,
            # 'instructions': instructions_emb
        }, f)
        
    with open('embeddings.pkl', 'rb') as f:
        return pickle.load(f)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-base-v2")   