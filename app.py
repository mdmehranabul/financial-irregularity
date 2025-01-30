FROM python:3.10-slim
# Set the working directory
WORKDIR /app

# Copy only requirements first (better caching)
COPY requirements.txt .

# Install dependencies efficiently
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 5000 (if running a web app)
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
