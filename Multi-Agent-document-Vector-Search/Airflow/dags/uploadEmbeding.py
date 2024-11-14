from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import sys

# Add your project directory to the Python path
sys.path.append('/Users/aniketpatole/Documents/GitHub/New/Projects/BigData/Assignment4/Multi-Agent-document-Vector-Search')

# Import your functions
from Airflow.scripts.docling_parser import (
    list_pdfs_in_s3_folder,
    process_pdf_and_upload,
    create_index_for_pdf,
    generate_and_store_embeddings
)

# Default arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),  # Timeout for task execution
}

# DAG definition
with DAG(
    "process_upload_embed_dag",
    default_args=default_args,
    description="A DAG to process PDFs, upload metadata to S3, and store embeddings in Pinecone",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # Process PDFs task
    def process_pdfs():
        try:
            logging.info("Starting to process PDFs...")
            pdf_files = list_pdfs_in_s3_folder()
            logging.info(f"Found {len(pdf_files)} PDF files.")
            for pdf_key in pdf_files:
                logging.info(f"Processing PDF: {pdf_key}")
                process_pdf_and_upload(pdf_key)
            logging.info("Completed processing PDFs.")
        except Exception as e:
            logging.error(f"Error during PDF processing: {e}")
            raise

    # Upload metadata task
    def upload_metadata():
        try:
            logging.info("Starting metadata upload...")
            pdf_files = list_pdfs_in_s3_folder()
            for pdf_key in pdf_files:
                logging.info(f"Uploading metadata for PDF: {pdf_key}")
                pdf_name, md_s3_key = process_pdf_and_upload(pdf_key)
                if not pdf_name or not md_s3_key:
                    logging.warning(f"Metadata upload skipped for {pdf_key}")
            logging.info("Completed metadata upload.")
        except Exception as e:
            logging.error(f"Error during metadata upload: {e}")
            raise

    # Generate embeddings task
    def generate_embeddings():
        try:
            logging.info("Starting to generate embeddings...")
            pdf_files = list_pdfs_in_s3_folder()
            logging.info(f"Found {len(pdf_files)} PDF files for embedding generation.")
            for pdf_key in pdf_files:
                logging.info(f"Processing embeddings for PDF: {pdf_key}")
                pdf_name, md_s3_key = process_pdf_and_upload(pdf_key)
                if pdf_name and md_s3_key:
                    index_name = create_index_for_pdf(pdf_name)
                    logging.info(f"Index created: {index_name} for PDF: {pdf_name}")
                    generate_and_store_embeddings(index_name, pdf_name, md_s3_key)
                    logging.info(f"Successfully generated embeddings for {pdf_name}.")
                else:
                    logging.warning(f"Skipping embeddings for {pdf_key} due to missing metadata.")
            logging.info("Completed embedding generation.")
        except Exception as e:
            logging.error(f"Error during embedding generation: {e}")
            raise

    # Define tasks
    process_task = PythonOperator(
        task_id="process_pdfs_task",
        python_callable=process_pdfs,
    )

    upload_task = PythonOperator(
        task_id="upload_metadata_task",
        python_callable=upload_metadata,
    )

    generate_task = PythonOperator(
        task_id="generate_embeddings_task",
        python_callable=generate_embeddings,
    )

    # Set task dependencies
    process_task >> upload_task >> generate_task