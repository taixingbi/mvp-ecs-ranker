FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      unzip \
      ca-certificates \
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscliv2.zip /var/lib/apt/lists/*

WORKDIR /workspace

COPY app/requirements.txt /workspace/app/requirements.txt
RUN pip install --no-cache-dir -r /workspace/app/requirements.txt

COPY app /workspace/app
COPY scripts/entrypoint.sh /workspace/scripts/entrypoint.sh
RUN chmod +x /workspace/scripts/entrypoint.sh

ENV MODEL_S3_URI=s3://reranker-models-646821141010/bge-reranker-v2-m3/ \
    MODEL_PATH=/models/bge-reranker-v2-m3 \
    MODEL_ID=BAAI/bge-reranker-v2-m3 \
    ADAPTER_PORT=8080 \
    AWS_REGION=us-east-1

EXPOSE 8080

ENTRYPOINT ["/workspace/scripts/entrypoint.sh"]
