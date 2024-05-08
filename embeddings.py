from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pickle
import os

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


# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-base-v2")     
    
# Check if the file is not empty
if os.path.exists('embeddings.pkl'):
    # Load embeddings from a file
    with open('embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    # Access the embeddings
    titles_emb = embeddings['titles']
    descriptions_emb = embeddings['descriptions']
    time_emb = embeddings['time']
    difficulty_emb = embeddings['difficulty']
    #ingredients_emb = embeddings['ingredients']
    #instructions_emb = embeddings['instructions']
else:
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
        #     ingredients.append(ingredient["ingredient"] if ingredient["ingredient"] is not None else 'None')
        # for instruction in recipe["instructions"]:
        #     instructions.append(instruction["stepText"] if instruction["stepText"] is not None else 'None')
        
    # Calculate embeddings
    titles_emb = encode(titles)
    descriptions_emb = encode(descriptions)
    time_emb = encode(times)
    difficulty_emb = encode(difficulties)
    #ingredients_emb = encode(ingredients)
    #instructions_emb = encode(instructions)

    # Save embeddings to a file
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump({
            'titles': titles_emb,
            'descriptions': descriptions_emb,
            'time': time_emb,
            'difficulty': difficulty_emb,
            #'ingredients': ingredients_emb,
            #'instructions': instructions_emb
        }, f)