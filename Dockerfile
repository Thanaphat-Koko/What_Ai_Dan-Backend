ARG PYTHON_VERSION=3.10.5
FROM python:${PYTHON_VERSION}-slim as base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx 

# Copy the source code into the container.
COPY . .

RUN python3 -m venv env && \
    chmod 744 env/bin/activate && \
    ./env/bin/activate && \
    # pip install opencv-python
    pip install -r requirements.txt


# Expose the port that the application listens on.
EXPOSE 5000

# Run the application.
CMD ["python3", "app.py"]
