FROM tensorflow/tensorflow:1.13.1-gpu-py3

COPY main.py ./main.py
COPY human_ISH_config.py ./human_ISH_config.py
COPY crop_and_rotate.py ./crop_and_rotate.py
COPY evaluate_embeddings.py ./evaluate_embeddings.py
COPY extract_data.py ./extract_data.py
COPY ISH_segmentation.py ./ISH_segmentation.py
COPY process.py ./process.py

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN rm ./requirements.txt


RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN mkdir ./triplet-reid
RUN git clone https://github.com/PegahA/triplet-reid.git ./triplet-reid


ENTRYPOINT ["python3" , "main.py"]
