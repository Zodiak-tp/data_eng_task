# Context expected to be parent folder
FROM jupyter/datascience-notebook:dc57157d6316

USER root

WORKDIR /build

RUN apt-get update && \
    apt-get install -y libpq-dev

RUN pip install pipenv

WORKDIR /usr/src/app

COPY jupyter/Pipfile* ./

RUN pipenv install --deploy --system

COPY src/data_engineering data_engineering
COPY pos_data pos_data

ENV PYTHONPATH "/usr/src/app:${PYTHONPATH}"

# Enable jupyter-notebook extensions, such as codefolding
RUN jupyter contrib nbextension install --system
RUN jupyter nbextension enable codefolding/main


ENV NB_PREFIX /
CMD ["sh","-c", "jupyter notebook --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=3000 --NotebookApp.token='data' --NotebookApp.password='data' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]

