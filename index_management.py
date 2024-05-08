from config import CONFIG
from opensearchpy import OpenSearch
import json as json
import torch

import embeddings as emb


host = CONFIG["host"]
port = CONFIG["port"]
user = CONFIG["user"]
password = CONFIG["password"]
index_name = user

# Delete the index
def delete_index(client, index_name):
    if client.indices.exists(index=index_name):
        # Delete the index.
        response = client.indices.delete(
            index = index_name,
            timeout = 600
        )
        print('\nDeleting index:')
        print(response)


def create_client():
    # Create the client with SSL/TLS enabled, but hostname verification disabled.
    client = OpenSearch(
        hosts = [{'host': host, 'port': port}],
        http_compress = True, # enables gzip compression for request bodies
        http_auth = (user, password),
        url_prefix = 'opensearch',
        use_ssl = True,
        verify_certs = False,
        ssl_assert_hostname = False,
        ssl_show_warn = False
    )
        
    return client

# Check if the index exists
def index_exists(client):
    if client.indices.exists(index=index_name):
        return True
    return False


#Creates the index with proper mapping
def create_index(client):
    index_body = {
    "settings":{
        "index":{
            "number_of_replicas":0,
            "number_of_shards":4,
            "refresh_interval":"1s",
            "knn":"true"
        }
    },
    "mappings": {
            "properties": {
                "recipe_id": {"type": "keyword"},
                "title": {"type": "text"},
                "description": {"type": "text"},
                "time": {"type": "integer"},
                "difficulty": {"type": "keyword"},
                "ingredients": {
                    "type": "nested",
                    "properties": {
                        "text": {"type": "text"},
                        "name": {"type": "text"},
                        "quantity": {"type": "float"},
                        "unit": {"type": "keyword"}
                    }
                },
                "instructions": {
                    "type": "nested",
                    "properties": {
                        "stepNumber": {"type": "integer"},
                        "text": {"type": "text"},
                        "durationSeconds": {"type": "integer"}
                    }
                },
                "nutrients": {
                    "type": "object",
                    "properties": {
                        "calories": {
                            "type": "object",
                            "properties": {
                                "quantity": {"type": "float"},
                                "measurement": {"type": "keyword"}
                            }
                        },
                        "protein": {
                            "type": "object",
                            "properties": {
                                "quantity": {"type": "float"},
                                "measurement": {"type": "keyword"}
                            }
                        },
                        "fat": {
                            "type": "object",
                            "properties": {
                                "quantity": {"type": "float"},
                                "measurement": {"type": "keyword"}
                            }
                        },
                        "carbohydrates": {
                            "type": "object",
                            "properties": {
                                "quantity": {"type": "float"},
                                "measurement": {"type": "keyword"}
                            }
                        }
                    }
                },
                "title_embedding":{
                    "type":"knn_vector",
                    "dimension": 768,
                    "method":{
                        "name":"hnsw",
                        "space_type":"innerproduct",
                        "engine":"faiss",
                        "parameters":{
                        "ef_construction":256,
                        "m":48
                        }
                    }
                },
                "description_embedding":{
                    "type":"knn_vector",
                    "dimension": 768,
                    "method":{
                        "name":"hnsw",
                        "space_type":"innerproduct",
                        "engine":"faiss",
                        "parameters":{
                        "ef_construction":256,
                        "m":48
                        }
                    }
                },
                "time_embedding":{
                    "type":"knn_vector",
                    "dimension": 768,
                    "method":{
                        "name":"hnsw",
                        "space_type":"innerproduct",
                        "engine":"faiss",
                        "parameters":{
                          "ef_construction":256,
                          "m":48
                        }
                    }
                },
                "difficulty_embedding":{
                    "type":"knn_vector",
                    "dimension": 768,
                    "method":{
                        "name":"hnsw",
                        "space_type":"innerproduct",
                        "engine":"faiss",
                        "parameters":{
                          "ef_construction":256,
                          "m":48
                        }
                    }
                },
                "ingredients_embedding":{
                    "type":"knn_vector",
                    "dimension": 768,
                    "method":{
                        "name":"hnsw",
                        "space_type":"innerproduct",
                        "engine":"faiss",
                        "parameters":{
                          "ef_construction":256,
                          "m":48
                        }
                    }
                },
                "instructions_embedding":{
                    "type":"knn_vector",
                    "dimension": 768,
                    "method":{
                        "name":"hnsw",
                        "space_type":"innerproduct",
                        "engine":"faiss",
                        "parameters":{
                          "ef_construction":256,
                          "m":48
                        }
                    }
                },
                "nutrients_embedding":{
                    "type":"knn_vector",
                    "dimension": 768,
                    "method":{
                        "name":"hnsw",
                        "space_type":"innerproduct",
                        "engine":"faiss",
                        "parameters":{
                          "ef_construction":256,
                          "m":48
                        }
                    }
                }
            }
        }
    }

    return client.indices.create(index_name, body=index_body)


def add_recipes_to_index(client, recipes_data):
    embeddings = emb.get_embeddings()
    
    for recipe_id in recipes_data:
        # Extract and format ingredients data
        ingredients = []
        for ingredient_data in recipes_data[recipe_id]['ingredients']:
            ingredient = {
                "text": ingredient_data['displayText'],
                "name": ingredient_data['ingredient'],
                "quantity": ingredient_data['quantity'],
                "unit": ingredient_data['unit']
            }
            ingredients.append(ingredient)

        instructions = []
        for instruction_data in recipes_data[recipe_id]['instructions']:
            instruction = {
                "stepNumber": instruction_data['stepNumber'],
                "text": instruction_data['stepText'],
                "durationSeconds": instruction_data['stepDurationSeconds']
            }
            instructions.append(instruction)

        # Check if nutrients data is available
        nutrients_data = recipes_data[recipe_id].get('nutrition')
        if nutrients_data:
            # Check if nutrients are available
            nutrients = {}
            for nutrient_name in ['calories', 'carbohydrateContent', 'fatContent', 'proteinContent']:
                nutrient_data = nutrients_data.get('nutrients', {}).get(nutrient_name)
                if nutrient_data:
                    nutrients[nutrient_name] = {
                        "quantity": nutrient_data.get('quantity'),
                        "unit": nutrient_data.get('measurement')
                    }
        else:
            nutrients = None
        
        recipe = {
            "recipe_id": recipe_id,
            "title": recipes_data[recipe_id]['displayName'],
            "description": recipes_data[recipe_id]['description'],
            "difficulty": recipes_data[recipe_id]['difficultyLevel'],
            "ingredients": ingredients,
            "instructions": instructions,
            "nutrients": nutrients,  # Assign nutrients here
            "time": recipes_data[recipe_id]['totalTimeMinutes'],
            
            "title_embedding": embeddings['titles'][int(recipe_id)].numpy(),
            "description_embedding": embeddings['descriptions'][int(recipe_id)].numpy(),
            "time_embedding": embeddings['times'][int(recipe_id)].numpy(),
            "difficulty_embedding": embeddings['difficulties'][int(recipe_id)].numpy(),
            # "ingredients_embedding": recipes_data[recipe_id]['ingredients_embedding'],
            # "instructions_embedding": recipes_data[recipe_id]['instructions_embedding'],
        }
        
        # Add recipe to index
        client.index(index=index_name, id=int(recipe_id), body=recipe)


def delete_recipes_from_index(client, recipe_book_len):
    for i in range(0, recipe_book_len):
        client.delete(index=index_name, id=i)    
    print('Done deleting recipes')
    
    

def close_index(client):
    return client.indices.close(index = index_name, timeout=600)


    
with open("recipes_data.json", "r") as read_file:
    recipes_data = json.load(read_file)

recipe_book_len = len(recipes_data)

