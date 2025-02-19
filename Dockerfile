# Use the official lightweight Python image.
FROM python:3.10-slim

# Set the working directory in the container.
WORKDIR /app

RUN pip install --upgrade pip
# Copy requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose the server port
EXPOSE 8080

# Command to start the server
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
