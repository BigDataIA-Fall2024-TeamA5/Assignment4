import logging
import os
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv
import boto3
from pinecone import Pinecone, ServerlessSpec
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from sentence_transformers import SentenceTransformer
import hashlib


# Initialize the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# AWS S3 and Pinecone configuration
s3_bucket_name = os.getenv("S3_BUCKET_NAME")
s3_pdf_folder = os.getenv("S3_PDF_FOLDER", "pdfs/")
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

# Initialize AWS S3 client
logger.info("Initializing S3 client...")
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)

# Initialize Pinecone client
logger.info("Initializing Pinecone client...")
pinecone_client = Pinecone(api_key=pinecone_api_key)

# Load the SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Utility Functions
def sanitize_name(name):
    """Converts name to lowercase and replaces invalid characters with hyphens."""
    return name.lower().replace("_", "-").replace(" ", "-")

def split_text_into_chunks(text, chunk_size=300):
    """Splits text into smaller chunks of specified size."""
    lines = text.splitlines()
    chunks = []
    current_chunk = []
    current_length = 0

    for line in lines:
        line_length = len(line.split())
        if current_length + line_length > chunk_size:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(line)
        current_length += line_length

    if current_chunk:  # Add the last chunk
        chunks.append("\n".join(current_chunk))

    return chunks

def list_pdfs_in_s3_folder():
    """Lists PDF files in the specified S3 folder."""
    logger.info(f"Listing PDFs in S3 folder: {s3_pdf_folder}")
    response = s3.list_objects_v2(Bucket=s3_bucket_name, Prefix=s3_pdf_folder)
    pdf_files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.pdf')]
    return pdf_files

def process_pdf_and_upload(pdf_key):
    """Downloads a PDF from S3, extracts images and text, then uploads to S3."""
    pdf_name = pdf_key.split('/')[-1].replace('.pdf', '')
    sanitized_pdf_name = sanitize_name(pdf_name)
    s3_output_folder = f"{sanitized_pdf_name}/"
    logger.info(f"Processing PDF: {pdf_key}")

    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_table_images = True
        pipeline_options.generate_picture_images = True
        doc_converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / sanitized_pdf_name
            output_dir.mkdir(parents=True, exist_ok=True)

            local_pdf_path = tmp_path / pdf_key.split('/')[-1]
            s3.download_file(s3_bucket_name, pdf_key, str(local_pdf_path))

            conv_res = doc_converter.convert(local_pdf_path)
            logger.info("Saving images and markdown...")

            content_md = conv_res.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
            md_filename = output_dir / f"{sanitized_pdf_name}.md"
            with md_filename.open("w") as fp:
                fp.write(content_md)
            s3_key_md = f"{s3_output_folder}{md_filename.name}"
            s3.upload_file(str(md_filename), s3_bucket_name, s3_key_md)

            return sanitized_pdf_name, s3_key_md

    except Exception as e:
        logger.error(f"Error processing PDF {pdf_key}: {e}")
    return None, None

def create_index_for_pdf(pdf_name):
    """Creates a Pinecone index for the PDF if it doesn't exist."""
    index_name = f"{pdf_name.replace('_', '-').lower()}-index"
    try:
        if index_name in [index.name for index in pinecone_client.list_indexes()]:
            logger.info(f"Index {index_name} already exists. Skipping creation.")
        else:
            logger.info(f"Creating Pinecone index for {pdf_name}")
            pinecone_client.create_index(
                name=index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region=pinecone_env)
            )
    except Exception as e:
        logger.error(f"Failed to create or verify index {index_name}: {e}")
    return index_name

def generate_and_store_embeddings(index_name, pdf_name, md_s3_key):
    """Generates embeddings from text chunks, stores metadata in S3, and references in Pinecone."""
    logger.info(f"Generating and storing embeddings for {pdf_name} in Pinecone index: {index_name}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f"{pdf_name}.md"
        s3.download_file(s3_bucket_name, md_s3_key, str(tmp_path))

        with open(tmp_path, 'r') as md_file:
            text = md_file.read()

        text_chunks = split_text_into_chunks(text)
        pinecone_index = pinecone_client.Index(index_name)

        for idx, chunk in enumerate(text_chunks):
            embedding = model.encode(chunk, convert_to_tensor=False).tolist()

            chunk_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()
            s3_key = f"{pdf_name}/metadata/{chunk_hash}.txt"

            try:
                s3.put_object(Bucket=s3_bucket_name, Key=s3_key, Body=chunk)
                logger.info(f"Uploaded metadata for chunk {idx} to S3 with key: {s3_key}")
            except Exception as e:
                logger.error(f"Failed to upload metadata for chunk {idx} to S3: {e}")
                continue

            metadata = {"s3_key": s3_key, "pdf_name": pdf_name}

            try:
                pinecone_index.upsert([{"id": f"{pdf_name}_{idx}", "values": embedding, "metadata": metadata}])
                logger.info(f"Stored embedding for chunk {idx} of {pdf_name} in Pinecone index: {index_name}")
            except Exception as e:
                logger.error(f"Failed to store embedding for chunk {idx} of {pdf_name}: {e}")

if __name__ == "__main__":
    pdf_files = list_pdfs_in_s3_folder()

    for pdf_key in pdf_files:
        pdf_name, md_s3_key = process_pdf_and_upload(pdf_key)

        if pdf_name and md_s3_key:
            index_name = create_index_for_pdf(pdf_name)
            generate_and_store_embeddings(index_name, pdf_name, md_s3_key)
