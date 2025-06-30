FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

WORKDIR /app
COPY . .
RUN apt-get install -y python3 python3-pip libopencv-dev && \
    pip install -r ./requirements.txt

EXPOSE 25629
CMD ["python3", "server.py"]
