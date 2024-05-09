import embeddings
import pprint as pp

from PIL import Image
import requests
import torch
import pickle
import embeddings as emb
from io import BytesIO


from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

def text_query(client, index_name, query):
    query_emb = embeddings.encode(query)

    query_denc = {
        'size': 3,
        '_source': ['recipe_id', 'title', 'description', 'instructions'],
        "query": {
            "bool": {
                "must": [
                    {  
                        "knn": {
                            "title_embedding": {
                                "vector": query_emb[0].numpy(),
                                "k": 2
                            }
                        },
                    },
                    {
                        "match": {
                            "title": query
                        }
                    }
                ],
                "should": [
                    {
                        "knn": {
                            "description_embedding": {
                                "vector": query_emb[0].numpy(),
                                "k": 2
                            }
                        }
                    },
                ]
            }
        }
    }

    response = client.search(
        body = query_denc,
        index = index_name
    )

    #print('\nSearch results:')
    #pp.pprint(response)
    
    # Sort the hits by score and get the recipe_id of the hit with the highest score
    highest_score_hit = max(response['hits']['hits'], key=lambda hit: hit['_score'])
    recipe_id = highest_score_hit['_source']['recipe_id']

    return highest_score_hit


def text_to_image(client, index_name, query_txt):
    embeddings = emb.get_embeddings()
        
    images_embeddings = embeddings['images_embedded']
    
    # Encode text query
    text_inputs = processor(text=query_txt, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_inputs)

    # Calculate similarity
    similarity = text_embeddings @ images_embeddings.T
    values, indices = torch.topk(similarity, 1)

    # Return the most similar image
    images = embeddings['images_urls']
    most_similar_image = images[indices[0]]
    
    print(most_similar_image)
    
    
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import UnidentifiedImageError

# Define the image transformation
transform = Compose([
    Resize(256), 
    CenterCrop(224), 
    ToTensor(), 
    Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])
    
def image_to_text(client, index_name, img_url):
    # Load the image
    response = requests.get(img_url)
    image = Image.open(BytesIO(response.content))
    image_input = transform(image).unsqueeze(0).to(device)

    # Calculate image features
    with torch.no_grad():
        image_features = model.get_image_features(image_input)
        
    # Define a set of sentences to compare with the image
    titles = embeddings.get_embeddings()['titles_str']
    sentences = titles
    text_inputs = processor(text=sentences, return_tensors="pt", padding=True).to(device)
     # Calculate text features
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)

    # Calculate the similarity between image and text features
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = torch.topk(similarity, 1)
    
    # Return the sentences with the highest similarity scores
    most_similar_sentence = sentences[indices[0]]
    
    print("Response:\n", most_similar_sentence)
    text_query(client, index_name, most_similar_sentence)
    
    
def image_to_image(client, index_name, img_url):
    response = requests.get(img_url)
    print("url: ", img_url),
    print("response: ", response)
    image = Image.open(BytesIO(response.content))
    try:
        image = Image.open(BytesIO(response.content))
    except UnidentifiedImageError:
        print("The URL does not point to a valid image file.")
        return
    image_input = transform(image).unsqueeze(0).to(device)

    # Calculate image features
    with torch.no_grad():
        image_features = model.get_image_features(image_input)

    embeddings = emb.get_embeddings()
    images_embeddings = embeddings['images_embedded']
    images_embeddings = embeddings['images_embedded']    
    
    # Calculate similarity
    similarity = image_features @ images_embeddings.T
    values, indices = torch.topk(similarity, 1)

    # Return the most similar image
    images = embeddings['images_urls']
    most_similar_image = images[indices[0]]
    
    print("Response:\n", most_similar_image)


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the pre-trained CLIP model and processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)