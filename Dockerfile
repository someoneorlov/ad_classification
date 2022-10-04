# FROM python:3.8.6-buster

FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get update && \
	apt-get install -y curl python3.8 python3.8-distutils && \
	ln -s /usr/bin/python3.8 /usr/bin/python && \
	rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    python -m pip install -U pip==20.3.3

ENV PROJECT_ROOT /app

ENV DATA_ROOT /data
ENV TEST_DATA_ROOT /test_data

RUN mkdir $PROJECT_ROOT $DATA_ROOT

COPY . $PROJECT_ROOT

WORKDIR $PROJECT_ROOT

ADD lib/pretrained_models/catboost_model/ lib/pretrained_models/catboost_model/
RUN tar -xf lib/pretrained_models/catboost_model/catboost_model.cbm.tar.xz -C lib/pretrained_models/catboost_model/

RUN pip install -r requirements.txt

RUN python -c "from transformers import BertTokenizer; tokenizer = BertTokenizer.from_pretrained('someoneorlov/rubert_contact'); tokenizer.save_pretrained('lib/pretrained_models/bert_model/')"

RUN python -c "from transformers import BertForSequenceClassification; model = BertForSequenceClassification.from_pretrained('someoneorlov/rubert_contact'); model.save_pretrained('lib/pretrained_models/bert_model/')"


CMD python lib/run.py
