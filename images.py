from PIL import Image
import requests
import torch

from transformers import CLIPProcessor, CLIPModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the pre-trained CLIP model and processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

image_urls = [
    "https://m.media-amazon.com/images/S/alexa-kitchen-msa-na-prod/recipes/thekitchn/016aa4923f044e1bad4cb2802f04133f7cf787b9bbf4fceb52438ecb70b28d89.jpg",
    "https://m.media-amazon.com/images/S/alexa-kitchen-msa-na-prod/recipes/thekitchn/2eba4355b22b630a2a230184619a07db3658962bae1a830cbc2c1cc9e93e86eb.jpg",
    "https://m.media-amazon.com/images/S/alexa-kitchen-msa-na-prod/recipes/thekitchn/77d73112c0675e4107d9c0e0dd5ee1038e43a8cb495ff2d8aeebc77564f6089d.jpg",
]

# Load all images
images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]

# Encode images
image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    image_embeddings = model.get_image_features(**image_inputs)

# Encode text query
text_inputs = processor(text=["a photo of pesto"], return_tensors="pt").to(device)
with torch.no_grad():
    text_embeddings = model.get_text_features(**text_inputs)
    
    
# Calculate similarity
similarity = text_embeddings @ image_embeddings.T
values, indices = torch.topk(similarity, 1)

# Return the most similar image
most_similar_image = image_urls[indices[0]]

print(most_similar_image)
