from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

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

titles = []
descriptions = []
time = []
difficulty = []
ingredients = []
instructions = []
for recipe in im.recipes_data.values():
    titles.append(recipe["displayName"] if recipe["displayName"] is not None else 'None') 
    descriptions.append(recipe["description"] if recipe["description"] is not None else 'None')
    time.append(recipe["totalTimeMinutes"] if recipe["totalTimeMinutes"] is not None else 'None') 
    difficulty.append(recipe["difficultyLevel"] if recipe["difficultyLevel"] is not None else 'None')
    ingredients.append(recipe["ingredients"] if recipe["ingredients"] is not None else 'None')
    instructions.append(recipe["instructions"] if recipe["instructions"] is not None else 'None')
     
    
titles_emb = encode(titles)
descriptions_emb = encode(descriptions)
time_emb = encode(titles)
difficulty_emb = encode(titles)
ingredients_emb = encode(titles)
instructions_emb = encode(titles)