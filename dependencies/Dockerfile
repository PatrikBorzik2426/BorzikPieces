FROM ghcr.io/tauffer-consulting/domino-base-piece:latest

# Default Dockerfile - redirects to Dockerfile_base
# This is kept for backward compatibility

RUN pip install --no-cache-dir \
    "numpy>=1.24.0,<2.0.0" \
    scipy==1.11.4 \
    pandas==2.0.3 \
    plotly==5.18.0 \
    Pillow==10.1.0 \
    nibabel==5.2.0 \
    matplotlib==3.7.5 \
    seaborn==0.12.2 \
    tqdm==4.66.1

# Copy the piece source code and config
COPY config.toml domino/pieces_repository/
COPY pieces domino/pieces_repository/pieces
COPY .domino domino/pieces_repository/.domino

# Copy sample data for medical imaging pieces
COPY data /data


# Copy the piece source code and config
COPY config.toml domino/pieces_repository/
COPY pieces domino/pieces_repository/pieces
COPY .domino domino/pieces_repository/.domino

# Copy sample data for medical imaging pieces
COPY data /data