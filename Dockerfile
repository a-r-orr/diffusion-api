FROM python:3.13-slim-bullseye

WORKDIR /usr/src/app

RUN apt-get -qqy update && apt-get install -qqy

# Install dependencies
COPY requirements.txt ./
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir -r requirements.txt

# Copy Model Cache
COPY local_model_cache /usr/src/app/local_model_cache

# Copy application code
COPY main.py /usr/src/app/main.py

# Get start up file in place
COPY helper_script.sh /usr/src/app/helper_script.sh
RUN chmod +x helper_script.sh

# Run start up file
CMD ["/usr/src/app/helper_script.sh"]