import embeddings
import pprint as pp

from PIL import Image
import requests
import torch
import pickle

from transformers import CLIPProcessor, CLIPModel

def text_query(client, index_name, query):
    query_emb = embeddings.encode(query)

    query_denc = {
        'size': 3,
        '_source': ['title', 'description'],
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

    print('\nSearch results:')
    pp.pprint(response)



def get_image_from_text_query(client, index_name, query_txt):
    
    with open('embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
        
    images_embeddings = torch.tensor(embeddings['images'])
    
    print("Type of images_embeddings in search: ", type(images_embeddings))
    
    # Encode text query
    text_inputs = processor(text=query_txt, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_inputs)

    # Calculate similarity
    similarity = text_embeddings @ images_embeddings.T
    values, indices = torch.topk(similarity, 1)

    # Return the most similar image
    most_similar_image = images_embeddings[indices[0]]


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the pre-trained CLIP model and processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)