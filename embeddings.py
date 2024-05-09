from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel

import torch
import torch.nn.functional as F
import pickle
import os
from PIL import Image
import requests
import json

# Import custom modules
import index_management as im

def decoding(sentence):
    generation_config = model.generation_config
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.max_new_tokens = 150
    
    encoded_input_ids_1 = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    
    with torch.no_grad():
        generation_output = model.generate(
            input_ids = encoded_input_ids_1,
            generation_config = generation_config,
            return_dict_in_generate = True,
            output_scores = True
        )

    for s in generation_output.sequences:
        output = tokenizer.decode(s, skip_special_tokens=True)
        print(output)
    

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

def encode_images(image_urls):
    # Load the pre-trained CLIP model and processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    
    # Load all images
    images = [Image.open(requests.get(url, stream=True).raw).resize((224, 224)) for url in image_urls]

    # Encode images
    image_inputs = processor(
        images=images,
        return_tensors="pt",
        padding=True
    ).to(device)
    with torch.no_grad():
        image_embeddings = model.get_image_features(**image_inputs)
    
    return image_embeddings

def encode_images_in_batches(images, batch_size=32):
        for i in range(0, len(images), batch_size):
            yield encode_images(images[i:i+batch_size])
        
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
    images = []
    
    for recipe in im.recipes_data.values():
        titles.append(recipe["displayName"] if recipe["displayName"] is not None else 'None') 
        descriptions.append(recipe["description"] if recipe["description"] is not None else 'None')
        for image in recipe["images"]:
            images.append(image['url'] if image['url'] is not None else 'None')
        
    # Calculate embeddings
    titles_emb = encode(titles)
    descriptions_emb = encode(descriptions)
    images_emb = encode_images(images)
    # images_emb_generator = encode_images_in_batches(images)

    # all_images_emb = []

    # Save embeddings to a file
    try:
        # for images_emb in images_emb_generator:
        #     # Append each batch of image embeddings to the list
        #     all_images_emb.extend(images_emb)
            
        with open('embeddings.pkl', 'wb') as f:
            pickle.dump({
                'titles_embedded': titles_emb,
                'titles_str': titles,
                'descriptions_embedded': descriptions_emb,
                'descriptions_str': descriptions,
                'images_embedded': images_emb,
                'images_urls': images
            }, f)
    except Exception as e:
        print(f"Error while writing embeddings to file: {e}")
        return None
            
    # Free up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Load embeddings from file
    try:
        with open('embeddings.pkl', 'rb') as f:
            return pickle.load(f)
    except EOFError:
        print("Error: The embeddings file is empty or not completely written.")
        return None
    except Exception as e:
        print(f"Error while loading embeddings from file: {e}")
        return None
    
    
    
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-base-v2").to(device) 