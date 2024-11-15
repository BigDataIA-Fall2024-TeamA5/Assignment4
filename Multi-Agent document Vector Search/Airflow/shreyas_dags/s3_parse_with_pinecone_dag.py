from airflow import DAG
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import torch
from docling.document_converter import DocumentConverter
import pdfplumber
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

# Configuration
BUCKET_NAME = "multimodalrag1"  # replace with your S3 bucket name
PDF_KEY = "testpdf.pdf"  # the key (filename) in the S3 bucket
LOCAL_PATH = "/Users/shreyabage/Documents/Airflow/docs/testpdf.pdf"  # local path to save the downloaded PDF
PARSED_TEXT_PATH = "/Users/shreyabage/Documents/Airflow/docs/parsed_text_and_tables.md"
IMAGE_DIR = "/Users/shreyabage/Documents/Airflow/docs/images"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "YOUR_API_KEY"
INDEX_NAME = "hello-pinecone"
DIMENSIONS = 384



def setup_pinecone():
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    existing_indexes = [index['name'] for index in pinecone_client.list_indexes()]
    if INDEX_NAME in existing_indexes:
        pinecone_client.delete_index(INDEX_NAME)
    pinecone_client.create_index(name=INDEX_NAME, dimension=DIMENSIONS, metric="cosine", spec=spec)
    return pinecone_client.Index(INDEX_NAME)

def download_from_s3():
    s3_hook = S3Hook(aws_conn_id="aws_s3_default")
    s3_hook.download_file(bucket_name=BUCKET_NAME, key=PDF_KEY, local_path=LOCAL_PATH)
    print(f"Downloaded {PDF_KEY} from S3 to {LOCAL_PATH}")

def parse_pdf():
    converter = DocumentConverter()
    docling_result = converter.convert(LOCAL_PATH)
    markdown_content = docling_result.document.export_to_markdown()
    with open(PARSED_TEXT_PATH, "w", encoding="utf-8") as text_file:
        text_file.write(markdown_content)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    with pdfplumber.open(LOCAL_PATH) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for image_idx, img in enumerate(page.images):
                img_bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                page_width, page_height = page.width, page.height
                if (0 <= img_bbox[0] <= page_width and 0 <= img_bbox[1] <= page_height and
                    0 <= img_bbox[2] <= page_width and 0 <= img_bbox[3] <= page_height):
                    cropped_image = page.within_bbox(img_bbox).to_image()
                    image_filename = f"{IMAGE_DIR}/image_{page_num}_{image_idx + 1}.png"
                    cropped_image.save(image_filename)
                    print(f"Image saved: {image_filename}")
                else:
                    print(f"Skipped out-of-bounds image on page {page_num}")

def generate_and_store_embeddings():
    index = setup_pinecone()
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    with open(PARSED_TEXT_PATH, "r") as file:
        document_text = file.read()
    inputs = tokenizer(document_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        text_embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    index.upsert([("document_text_001", text_embeddings)])
    print("Text embeddings upserted to Pinecone.")

    image_model = resnet50(pretrained=True)
    image_model.eval()
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for image_filename in os.listdir(IMAGE_DIR):
        if image_filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(IMAGE_DIR, image_filename)
            image = Image.open(image_path).convert("RGB")
            image_tensor = image_transform(image).unsqueeze(0)
            with torch.no_grad():
                image_embedding = image_model(image_tensor).squeeze().numpy()[:DIMENSIONS]
            index.upsert([(image_filename, image_embedding)])
            print(f"Image embedding for {image_filename} upserted to Pinecone.")

with DAG(
    dag_id="s3_parse_with_pinecone_dag",
    start_date=datetime(2024, 11, 13),
    schedule_interval="@daily",
    catchup=False,
) as dag:
    
    download_task = PythonOperator(
        task_id="download_from_s3",
        python_callable=download_from_s3,
    )

    parse_task = PythonOperator(
        task_id="parse_pdf",
        python_callable=parse_pdf,
    )

    embedding_task = PythonOperator(
        task_id="generate_and_store_embeddings",
        python_callable=generate_and_store_embeddings,
    )

    download_task >> parse_task >> embedding_task
