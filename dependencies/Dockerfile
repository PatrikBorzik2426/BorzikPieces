FROM ghcr.io/tauffer-consulting/domino-base-piece:latest

# Install specific requirements for all pieces
RUN pip install --no-cache-dir numpy==1.23.5 plotly==5.18.0 Pillow==10.1.0 nibabel==5.2.0 scipy==1.11.4 matplotlib==3.7.5

# Copy the piece source code and config
COPY config.toml domino/pieces_repository/
COPY pieces domino/pieces_repository/pieces
COPY .domino domino/pieces_repository/.domino

# Copy sample data for medical imaging pieces
COPY data /data