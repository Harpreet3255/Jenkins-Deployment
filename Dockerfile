# Use an official Python image as a base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install Ansible and other dependencies
RUN apt-get update && apt-get install -y \
    ansible \
    git \
    ssh \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file to install Python dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the Ansible playbook
COPY ansible playbook.yml
COPY tests/test_model.py /app/tests/test_model.py


# Expose the port the app will run on (optional, for if you want to run a web app inside the container)
# EXPOSE 8080

# Set the entrypoint to run Jenkins-related commands, this can be customized
ENTRYPOINT ["tail", "-f", "/dev/null"]
