FROM continuumio/miniconda3
WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "information-noise-reduction-for-investors", "/bin/bash", "-c"]
COPY . .
# TODO :
ENTRYPOINT ["conda", "run", "-n", "information-noise-reduction-for-investors", "python", "cli.py"]
