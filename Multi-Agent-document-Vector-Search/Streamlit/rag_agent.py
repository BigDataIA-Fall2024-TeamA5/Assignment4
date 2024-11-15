import pinecone
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import boto3

# Load environment variables
load_dotenv()

# Load API keys and configuration from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "ai-and-big-data-in-investments-index"
PINECONE_ENVIRONMENT = "us-east-1"

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = "us-east-2"

# Validate required environment variables
if not all([PINECONE_API_KEY, AWS_ACCESS_KEY, AWS_SECRET_KEY]):
    raise ValueError("Missing required environment variables. Check your .env file.")

# Initialize Pinecone client
print("Initializing Pinecone client...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists
if INDEX_NAME in [index.name for index in pc.list_indexes()]:
    pinecone_index = pc.Index(INDEX_NAME)
    print(f"Connected to Pinecone index: {INDEX_NAME}")
else:
    print(f"Error: Pinecone index '{INDEX_NAME}' not found.")
    exit(1)

# Initialize the embedding model
print("Loading embedding model...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded successfully.")

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# S3 bucket name
bucket_name = "doclingairflow"

def query_pinecone(user_query, top_k=3):
    """Query Pinecone to find the most relevant embeddings and fetch metadata from S3 if needed."""
    print("Encoding user query...")
    query_embedding = sentence_model.encode(user_query).tolist()
    print("Query encoded successfully.")
    
    # Query Pinecone
    print("Querying Pinecone...")
    response = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True  # Include metadata for context
    )
    
    results = []
    if 'matches' in response:
        print(f"Found {len(response['matches'])} matches.")
        for match in response['matches']:
            metadata = match['metadata']
            text = metadata.get("text", None)
            
            # If text is missing, retrieve it from S3 using the s3_key
            if not text:
                s3_key = metadata.get("s3_key")
                if s3_key:
                    print(f"Fetching metadata from S3 for key: {s3_key}")
                    s3_response = s3.get_object(Bucket=bucket_name, Key=s3_key)
                    text = s3_response["Body"].read().decode("utf-8")
                else:
                    text = "No text available"

            score = match['score']
            results.append({"text": text, "score": score})
    else:
        print("No matches found.")
        return "No matches found."

    return results

# Prompt the user for a query and check if it's empty
while True:
    user_query = input("Please enter your query: ").strip()
    if user_query:
        break
    print("No query entered. Please enter a valid query.")

# Run the query function
results = query_pinecone(user_query)

# Display results
if results != "No matches found.":
    for idx, result in enumerate(results):
        print(f"\nMatch {idx + 1}:")
        print(f"Score: {result['score']}")
        print(f"Text: {result['text']}")
else:
    print(results)
