FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages globally (not in user space)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Verify installations
RUN python -c "import flask; import tensorflow; print('Packages loaded successfully')"

EXPOSE 5000
# Run the application directly from C:\Users\HP\Desktop\module 7 part 2
CMD ["python", "app.py"]