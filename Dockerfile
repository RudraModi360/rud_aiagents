# Use a lightweight Python base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install python-multipart fastapi uvicorn jupyter-client nbformat ipykernel 
RUN python3 -m ipykernel install --user
RUN pip install pandas numpy matplotlib scipy seaborn scikit-learn pyarrow tabulate openpyxl xlrd

# Create necessary directories
RUN mkdir -p /mnt/data /mnt/jupyter_sessions /workspace


# Set environment variables for mounted volumes
ENV DATA_DIR=/mnt/data
