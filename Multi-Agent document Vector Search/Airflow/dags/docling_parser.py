import logging
import time
import os
from datetime import datetime
from pathlib import Path
import boto3
import tempfile
from dotenv import load_dotenv
import pinecone
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from docling_core.types.doc import ImageRefMode
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling_core.types import base

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# AWS S3 and Pinecone settings from .env file
s3_bucket_name = os.getenv("S3_BUCKET_NAME")
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)

# Pinecone initialization
from pinecone import Pinecone, ServerlessSpec

# Initialize the Pinecone client
pinecone_client = Pinecone(
    api_key=pinecone_api_key
)

# Check if the index exists, and create it if not
if pinecone_index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=pinecone_index_name,
        dimension=1536,  # Set this to the correct dimension of your embeddings
        metric='cosine',  # Use the appropriate metric for your use case
        spec=ServerlessSpec(
            cloud='aws',
            region=pinecone_env  # Set this to your Pinecone environment region
        )
    )

# Configuration for image resolution scaling
IMAGE_RESOLUTION_SCALE = float(os.getenv("IMAGE_RESOLUTION_SCALE", 2.0))

def download_pdf_from_s3(pdf_key, local_dir):
    """Download a PDF file from S3 to a local directory."""
    local_pdf_path = local_dir / pdf_key.split('/')[-1]
    s3.download_file(s3_bucket_name, pdf_key, str(local_pdf_path))
    return local_pdf_path


def upload_to_s3(local_path, s3_key):
    """Upload a local file to S3."""
    s3.upload_file(str(local_path), s3_bucket_name, s3_key)


def process_pdf(**context):
    """Process the PDF to extract images and markdown content."""
    pdf_key = context['dag_run'].conf.get('pdf_key')
    pdf_key_base = pdf_key.split('/')[-1].replace('.pdf', '')

    # Set up pipeline options for Docling
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = False
    pipeline_options.generate_table_images = True
    pipeline_options.generate_picture_images = True

    # Initialize document converter
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        output_dir = tmp_path / pdf_key_base
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download PDF from S3
        local_pdf_path = download_pdf_from_s3(pdf_key, tmp_path)
        conv_res = doc_converter.convert(local_pdf_path)

        # Save images and markdown
        for element, _level in conv_res.document.iterate_items():
            if isinstance(element, PictureItem) or isinstance(element, TableItem):
                image_filename = output_dir / f"{pdf_key_base}-element-{element.id}.png"
                element.image.pil_image.save(image_filename, "PNG")
                upload_to_s3(image_filename, f"{pdf_key_base}/images/{image_filename.name}")

        # Save markdown content
        content_md = conv_res.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
        md_filename = output_dir / f"{pdf_key_base}.md"
        with md_filename.open("w") as fp:
            fp.write(content_md)
        upload_to_s3(md_filename, f"{pdf_key_base}/{md_filename.name}")

    # Pass the pdf_key_base for the next tasks
    context['task_instance'].xcom_push(key="pdf_key_base", value=pdf_key_base)


def create_embeddings(**context):
    """Generate embeddings for each text chunk in the processed PDF."""
    pinecone_index = pinecone_client.Index(pinecone_index_name)
    pdf_key_base = context['task_instance'].xcom_pull(key="pdf_key_base")
    output_dir = Path(tempfile.gettempdir()) / pdf_key_base
    embeddings = []

    # Assuming text chunks have been saved in markdown files or similar
    md_filepath = output_dir / f"{pdf_key_base}.md"
    with open(md_filepath, 'r') as md_file:
        for idx, chunk in enumerate(md_file.readlines()):
            embedding = pinecone_index.upsert([
                {
                    "id": f"{pdf_key_base}_{idx}",
                    "values": [float(x) for x in chunk.split()],
                    "metadata": {"pdf_name": pdf_key_base}
                }
            ])
            embeddings.append(embedding)
    _log.info(f"Generated and stored {len(embeddings)} embeddings for {pdf_key_base}")


def store_embeddings_in_pinecone(**context):
    """Store generated embeddings into Pinecone with metadata."""
    pdf_key_base = context['task_instance'].xcom_pull(key="pdf_key_base")
    output_dir = Path(tempfile.gettempdir()) / pdf_key_base
    pinecone_index = pinecone_client.Index(pinecone_index_name)
    md_filepath = output_dir / f"{pdf_key_base}.md"
    with open(md_filepath, 'r') as md_file:
        for idx, chunk in enumerate(md_file.readlines()):
            pinecone_index.upsert([
                {
                    "id": f"{pdf_key_base}_{idx}",
                    "values": [float(x) for x in chunk.split()],
                    "metadata": {"pdf_name": pdf_key_base}
                }
            ])

with DAG(
    "pdf_embedding_pipeline_1",
    start_date=datetime(2023, 1, 1),
    schedule='@daily',
    catchup=False,
) as dag:

    process_task = PythonOperator(
        task_id="process_pdf_task",
        python_callable=process_pdf
    )

    embedding_task = PythonOperator(
        task_id="embedding_task",
        python_callable=create_embeddings
    )

    store_task = PythonOperator(
        task_id="store_embeddings_task",
        python_callable=store_embeddings_in_pinecone
    )

    # Task dependencies
    process_task >> embedding_task >> store_task