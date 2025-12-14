# Dependencies Structure

This repository uses **Dockerfile-based dependencies** for all pieces. Dependencies are split into two groups:

## Dockerfile Groups

### 1. Dockerfile_base
**Used by:** Basic data processing and visualization pieces
- NiftiDataLoaderPiece
- NiftiVisualizationPiece
- NiftiEDAPiece
- NiftiPreprocessingPiece
- DataSplitPiece
- PituitaryDatasetPiece
- HelloWorldPiece
- GenerativeShapesPiece

**Dependencies:**
- numpy, scipy, pandas
- matplotlib, seaborn, plotly
- nibabel (medical imaging)
- Pillow (image processing)

### 2. Dockerfile_torch
**Used by:** Deep learning pieces requiring PyTorch and MONAI
- ModelTrainingPiece
- ModelInferencePiece

**Dependencies:**
- All dependencies from Dockerfile_base
- PyTorch (latest version)
- MONAI (medical AI library)
- einops (tensor operations)

## Requirements Files

- `requirements.txt` - Basic dependencies (matches Dockerfile_base)
- `requirements_torch.txt` - Deep learning dependencies (matches Dockerfile_torch)

## Main Dockerfile

The main `Dockerfile` is kept for backward compatibility and mirrors `Dockerfile_base`.

## How It Works

Each piece's `metadata.json` specifies which Dockerfile to use:

```json
{
  "dependency": {
    "dockerfile": "Dockerfile_base"  // or "Dockerfile_torch"
  }
}
```

This approach ensures:
- ✅ Faster builds for non-ML pieces (no PyTorch installation)
- ✅ Smaller Docker images for simple pieces
- ✅ Proper dependency isolation
- ✅ Clear separation of concerns
