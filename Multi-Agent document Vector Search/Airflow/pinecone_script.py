import os
import torch
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

# Set up Pinecone API key
api_key = os.getenv("PINECONE_API_KEY") or "YOUR_API_KEY"
pinecone_client = Pinecone(api_key=api_key)

# Define the serverless index specification
spec = ServerlessSpec(cloud="aws", region="us-east-1")  # Use us-east-1 region for free tier
index_name = "hello-pinecone"
dimensions = 384  # Ensure this matches your embedding size for both text and images

# Check if the index exists, delete it if it does
existing_indexes = [index['name'] for index in pinecone_client.list_indexes()]
if index_name in existing_indexes:
    pinecone_client.delete_index(index_name)

# Create the index in us-east-1
pinecone_client.create_index(
    name=index_name,
    dimension=dimensions,
    metric="cosine",
    spec=spec
)

# Connect to the index
index = pinecone_client.Index(index_name)
print(f"Index {index_name} created and connected successfully!")

# Load the pre-trained language model for text embedding generation
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to convert text to embeddings
def get_text_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# Read content from conversion_results.txt for text/table embeddings
file_path = "conversion_results.txt"
with open(file_path, "r") as file:
    document_text = file.read()

# Generate and upsert embeddings for the document text
text_embedding_vector = get_text_embeddings(document_text)
text_embedding_id = "document_text_001"  # Unique ID for the text embedding
index.upsert([(text_embedding_id, text_embedding_vector)])
print(f"Text embedding with ID {text_embedding_id} upserted to Pinecone index.")

# Load the pre-trained image embedding model
image_model = resnet50(pretrained=True)
image_model.eval()
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to convert image to embeddings
def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = image_model(image_tensor).squeeze()
    return embedding.numpy()[:dimensions]  # Ensure it matches the Pinecone index dimensions

# Directory containing the images
image_dir = "/Users/shreyabage/Documents/Airflow/extracted_images"
image_embeddings = {}

# Generate and upsert embeddings for each image in the directory
for image_filename in os.listdir(image_dir):
    if image_filename.endswith((".png", ".jpg", ".jpeg")):  # Only process image files
        image_path = os.path.join(image_dir, image_filename)
        image_embedding_vector = get_image_embedding(image_path)
        image_id = image_filename  # Use filename as the ID for each image
        index.upsert([(image_id, image_embedding_vector)])
        print(f"Image embedding with ID {image_id} upserted to Pinecone index.")

print("All embeddings upserted successfully!")
