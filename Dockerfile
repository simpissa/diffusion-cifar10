FROM continuumio/miniconda3

# Set working directory
WORKDIR /workspace

# Copy Conda env file and source code
COPY environment.yml .
COPY . .

# Create conda environment
RUN conda env create -f environment.yml

# Use the conda environment
SHELL ["conda", "run", "-n", "GVC", "/bin/bash", "-c"]

# Run training by default
CMD ["conda", "run", "--no-capture-output", "-n", "GVC", "python", "train.py"]
