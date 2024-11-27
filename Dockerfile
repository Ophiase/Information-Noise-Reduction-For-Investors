FROM continuumio/miniconda3
WORKDIR /app
COPY . /app
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "information-noise-reduction-for-investors", "/bin/bash", "-c"]

EXPOSE 8888

CMD ["conda", "run", "-n", "information-noise-reduction-for-investors", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]