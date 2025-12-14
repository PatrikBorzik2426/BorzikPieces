FROM ghcr.io/tauffer-consulting/domino-base-piece:latest

# Install base scientific packages
RUN pip install --no-cache-dir "numpy>=1.24.0,<2.0.0" scipy==1.11.4 pandas==2.0.3

# Install visualization packages
RUN pip install --no-cache-dir plotly==5.18.0 Pillow==10.1.0 matplotlib==3.7.5 seaborn==0.12.2

# Install medical imaging packages
RUN pip install --no-cache-dir nibabel==5.2.0

# Install medical AI packages (MONAI will install PyTorch as dependency)
RUN pip install --no-cache-dir monai==1.3.0 einops==0.7.0

# Copy the piece source code and config
COPY config.toml domino/pieces_repository/
COPY pieces domino/pieces_repository/pieces
COPY .domino domino/pieces_repository/.domino

# Copy sample data for medical imaging pieces
COPY data /data