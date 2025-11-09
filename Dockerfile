# Minimal reproducible Dockerfile for the MSc IDS repo
# Uses micromamba base image (small, fast conda-style env manager)
FROM mambaorg/micromamba:1.4.0

# Set workdir to repository root inside container
WORKDIR /workspace

# Copy environment specification into image
COPY environment.yml /workspace/environment.yml

# Create the conda env from environment.yml and clean caches
RUN micromamba env create -f /workspace/environment.yml -n msc-ids && \
    micromamba clean --all --yes

# Use bash as default shell and pre-load environment
USER root
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate msc-ids" >> /root/.bashrc
USER mambauser

# Copy repository files into the image (for build-time usage)
# Note: for development we will mount the repo at runtime instead of using the image copy
COPY . /workspace

# Default command: start a bash session (you can override to run jupyter)
CMD ["bash"]
