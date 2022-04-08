FROM ghcr.io/osai-ai/dokai:22.03-pytorch

ENV TORCH_HOME /workdir/data/.torch

# Predict requirements
COPY ./requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt