import pytest
import os
import tempfile
import numpy as np
from pieces.NiftiPreprocessingPiece.piece import NiftiPreprocessingPiece
from pieces.NiftiPreprocessingPiece.models import InputModel, SubjectInfo, NormalizationMethod


class TestNiftiPreprocessingPiece:
    """Tests for NiftiPreprocessingPiece"""

    def setup_method(self):
        """Set up test fixtures"""
        self.piece = NiftiPreprocessingPiece()

    def test_zscore_normalization(self):
        """Test z-score normalization"""
        volume = np.random.randn(10, 10, 10).astype(np.float32) * 100 + 500
        
        normalized = self.piece.normalize_volume(volume, NormalizationMethod.ZSCORE)
        
        # Z-score should have approximately mean=0 and std=1
        assert abs(normalized.mean()) < 0.01
        assert abs(normalized.std() - 1.0) < 0.01

    def test_minmax_normalization(self):
        """Test min-max normalization"""
        volume = np.random.randn(10, 10, 10).astype(np.float32) * 100 + 500
        
        normalized = self.piece.normalize_volume(volume, NormalizationMethod.MINMAX)
        
        # Min-max should scale to [0, 1]
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_percentile_normalization(self):
        """Test percentile-based normalization"""
        volume = np.random.randn(10, 10, 10).astype(np.float32) * 100 + 500
        # Add some outliers
        volume[0, 0, 0] = 10000
        volume[1, 1, 1] = -10000
        
        normalized = self.piece.normalize_volume(
            volume, 
            NormalizationMethod.PERCENTILE,
            lower_pct=5.0,
            upper_pct=95.0
        )
        
        # Should clip outliers and scale
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_no_normalization(self):
        """Test pass-through (no normalization)"""
        volume = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = self.piece.normalize_volume(volume, NormalizationMethod.NONE)
        
        np.testing.assert_array_equal(volume, result)
