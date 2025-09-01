# Use official Python image
FROM python:3.9-slim

# Create a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the textModel folder contents into /app
COPY --chown=user ./textModel/ . 

# Expose the port your Flask app will run on
EXPOSE 7860

# Command to run Flask app in Hugging Face Spaces
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=7860

CMD ["flask", "run"]

