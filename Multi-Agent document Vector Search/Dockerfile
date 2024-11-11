# Start with the base Airflow image
FROM apache/airflow:2.10.3

# Install required Python packages
RUN pip install \
    docling==2.0.0 \
    pinecone-client \
    sentence-transformers \
    docling_core

# Set environment variables for Airflow
ENV AIRFLOW_HOME=/opt/airflow