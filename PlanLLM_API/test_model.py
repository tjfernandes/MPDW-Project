import os

import requests
import json


internal_url = 'http://10.10.255.202:5633'
external_url = "https://twiz.novasearch.org"

max_timeout = 10

def test_ping(url: str):
    # Make a GET request to the URL
    response = requests.get(url, timeout=max_timeout)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("GET request successful for URL:", url)
        print("Response:", response.text)
    else:
        print("GET request failed with status code:", response.status_code)


def test_raw_post_request(url: str, text: str):

    url = os.path.join(url, "raw")

    # This is just an example that you can try to play around
    #test_text = "Hi can you give the recipe?"
    # In practice it should be in a format similar to this
    # test_text = "<|prompter|> You are a taskbot tasked with helping users cook recipes or DIY projects. I will give you a recipe and I want you to help me do it step by step. You should always be empathetic, honest, and should always help me. If I ask you something that does not relate to the recipe you should politely reject the request and try too get me focused on the recipe. I am unsure how to cook something or do something related to the recipe you should help me to the best of your ability. Please use a neutral tone of voice. Recipe: Test Recipe Steps: Step 1: Preheat oven to 350 degrees Step 2: Mix ingredients together Step 3: Bake for 30 minutes <|endofturn|> <|prompter|> I haven't started cooking yet. <|endofturn|> <|assistant|> ok! <|endofturn|> <|prompter|> Hello <|endofturn|> <|assistant|>"

    data = {
        "text": text,
        "max_tokens": 100,
        "temperature": 0.0,
        "top_p": 1,
        "top_k": -1,
    }

    # Make the POST request
    response = requests.post(url, json=data, timeout=max_timeout)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("POST request successful for URL:", url)
        print("Response:", response.text)
    else:
        print("POST request failed with status code:", response.status_code)


def test_structured_post_request(url: str):

    url = os.path.join(url, "structured")

    # check this file to understand the structure of the data
    with open('example_conversation.json') as f:
        data = json.load(f)

    data = {
        "dialog": data,
        "max_tokens": 100,
        "temperature": 0.0,
        "top_p": 1,
        "top_k": -1,
    }

    # Make the POST request
    response = requests.post(url, json=data, timeout=max_timeout)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("POST request successful for URL:", url)
        print("Response:", response.text)
    else:
        print("POST request failed with status code:", response.status_code)


if __name__ == '__main__':

    # ping
    # test_ping(internal_url)
    # test_ping(external_url)

    # raw
    # test_raw_post_request(internal_url)
    # get_recipe_text = "You are a taskbot tasked with helping users cook recipes or DIY projects. I will give you a recipe and I want you to help me do it step by step. You should always be empathetic, honest, and should always help me. If I ask you something that does not relate to the recipe you should politely reject the request and try too get me focused on the recipe. I am unsure how to cook something or do something related to the recipe you should help me to the best of your ability. Please use a neutral tone of voice. Understood?"
    # test_raw_post_request(external_url, get_recipe_text)
    
    # get_next_step = "Are you familiar with the recipe of Spaggetti Carbonara?"
    # test_raw_post_request(external_url, get_next_step)
    
    # get_first_step = "Give me the first step of the recipe"
    # test_raw_post_request(external_url, get_next_step)

    # strutured
    # test_structured_post_request(internal_url)
    test_structured_post_request(external_url)
