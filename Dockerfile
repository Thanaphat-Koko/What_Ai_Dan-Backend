# start by pulling the python image
FROM ubuntu:20.04

# Set the maintainer label
LABEL maintainer="backend@WhatAiDan.com"

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install necessary packages
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean

# switch working directory
WORKDIR /app
    
# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# install the dependencies and packages in the requirements file
RUN apt-get install python3-opencv -y && \
pip3 install -r requirements.txt 

# copy every content from the local file to the image
COPY . /app

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the Flask application
CMD ["python3", "app.py"]