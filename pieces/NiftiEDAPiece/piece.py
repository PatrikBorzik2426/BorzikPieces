from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, EDAStatistics
import os
import json
import base64
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict


class NiftiEDAPiece(BasePiece):
    """
    A piece that performs comprehensive Exploratory Data Analysis (EDA) on NIfTI medical imaging datasets.
    
    This piece generates various visualizations and statistics including:
    - Volume shape distributions
    - Intensity histograms and distributions
    - Class distribution in masks
    - Slice-wise analysis
    - Mask coverage statistics
    - Correlations and outlier detection
    """

    def _generate_html_gallery(self, output_dir: str, viz_count: int, analysis_summary: str) -> str:
        """Generate an HTML gallery showing all visualizations"""
        
        # List of expected visualization files
        viz_files = [
            ('volume_shapes.png', 'Volume Shape Distributions'),
            ('intensity_boxplots.png', 'Intensity Statistics (Mean, Std, Median, Max)'),
            ('intensity_distribution.png', 'Overall Intensity Distribution'),
            ('class_distribution.png', 'Class Distribution'),
            ('mask_coverage.png', 'Mask Coverage Distribution'),
            ('intensity_correlations.png', 'Intensity Metrics Correlation Matrix'),
            ('per_subject_intensity.png', 'Per-Subject Intensity Comparison')
        ]
        
        html_parts = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '<meta charset="UTF-8">',
            '<title>NIfTI EDA Analysis Report</title>',
            '<style>',
            'body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }',
            'h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }',
            'h2 { color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 5px; }',
            '.summary { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
            '.summary pre { background: #f9f9f9; padding: 15px; border-left: 4px solid #4CAF50; overflow-x: auto; }',
            '.viz-container { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
            '.viz-title { font-size: 18px; font-weight: bold; color: #444; margin-bottom: 15px; }',
            '.viz-image { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }',
            '.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }',
            '.stat-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #4CAF50; }',
            '.stat-label { font-size: 12px; color: #888; text-transform: uppercase; }',
            '.stat-value { font-size: 24px; font-weight: bold; color: #333; margin-top: 5px; }',
            '</style>',
            '</head>',
            '<body>',
            '<h1>📊 NIfTI Medical Imaging EDA Report</h1>',
            '<div class="summary">',
            '<h2>Analysis Summary</h2>',
            f'<pre>{analysis_summary}</pre>',
            '</div>'
        ]
        
        # Add each visualization
        for viz_file, viz_title in viz_files:
            viz_path = os.path.join(output_dir, viz_file)
            if os.path.exists(viz_path):
                try:
                    with open(viz_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    html_parts.extend([
                        '<div class="viz-container">',
                        f'<div class="viz-title">{viz_title}</div>',
                        f'<img src="data:image/png;base64,{img_data}" class="viz-image" alt="{viz_title}">',
                        '</div>'
                    ])
                except Exception as e:
                    self.logger.warning(f"Could not load {viz_file}: {e}")
        
        html_parts.extend([
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(html_parts)

    def piece_function(self, input_data: InputModel) -> OutputModel:
        try:
            self.logger.info("=" * 60)
            self.logger.info("Starting NiftiEDAPiece execution")
            self.logger.info("=" * 60)
            
            # Create output directory
            output_dir = input_data.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Limit subjects for performance
            subjects = input_data.subjects[:input_data.max_subjects]
            self.logger.info(f"Analyzing {len(subjects)} subjects (max: {input_data.max_subjects})")
            
            # Initialize data collectors
            volume_shapes = []
            intensity_stats = []
            mask_coverages = []
            class_counts = defaultdict(int)
            slice_intensities = []
            
            # Analyze each subject
            for idx, subject in enumerate(subjects):
                try:
                    self.logger.info(f"Processing subject {idx+1}/{len(subjects)}: {subject.subject_id}")
                    
                    # Load image
                    img_nifti = nib.load(subject.image_path)
                    img_data = img_nifti.get_fdata().astype(np.float32)
                    
                    # Collect shape statistics
                    volume_shapes.append(list(img_data.shape))
                    
                    # Collect intensity statistics
                    p1, p99 = np.percentile(img_data, [1, 99])
                    intensity_stats.append({
                        'subject_id': subject.subject_id,
                        'mean': float(img_data.mean()),
                        'std': float(img_data.std()),
                        'min': float(img_data.min()),
                        'max': float(img_data.max()),
                        'p1': float(p1),
                        'p99': float(p99),
                        'median': float(np.median(img_data))
                    })
                    
                    # Sample random slices for intensity distribution
                    if input_data.num_sample_slices > 0:
                        num_slices = min(input_data.num_sample_slices, img_data.shape[1])
                        slice_indices = np.random.choice(img_data.shape[1], num_slices, replace=False)
                        for slice_idx in slice_indices:
                            slice_data = img_data[:, slice_idx, :].flatten()
                            slice_intensities.extend(slice_data[::100])  # Subsample for memory
                    
                    # Load and analyze mask if available
                    if subject.mask_path:
                        mask_nifti = nib.load(subject.mask_path)
                        mask_data = mask_nifti.get_fdata().astype(np.int64)
                        
                        # Calculate mask coverage
                        total_voxels = mask_data.size
                        foreground_voxels = np.sum(mask_data > 0)
                        coverage = foreground_voxels / total_voxels
                        mask_coverages.append(coverage)
                        
                        # Count classes
                        unique_classes, counts = np.unique(mask_data, return_counts=True)
                        for cls, count in zip(unique_classes, counts):
                            class_counts[int(cls)] += int(count)
                    
                except Exception as e:
                    self.logger.error(f"Error processing subject {subject.subject_id}: {str(e)}")
                    continue
            
            # Generate visualizations
            viz_count = 0
            
            # 1. Volume Shape Distribution
            self.logger.info("Generating volume shape distribution plot...")
            shapes_df = pd.DataFrame(volume_shapes, columns=['X', 'Y', 'Z'])
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for idx, dim in enumerate(['X', 'Y', 'Z']):
                axes[idx].hist(shapes_df[dim], bins=20, color='skyblue', edgecolor='black')
                axes[idx].set_xlabel(f'{dim} dimension')
                axes[idx].set_ylabel('Frequency')
                axes[idx].set_title(f'{dim} Dimension Distribution')
                axes[idx].axvline(shapes_df[dim].mean(), color='red', linestyle='--', label=f'Mean: {shapes_df[dim].mean():.1f}')
                axes[idx].legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'volume_shapes.png'), dpi=150, bbox_inches='tight')
            plt.close()
            viz_count += 1
            
            # 2. Intensity Statistics Boxplot
            self.logger.info("Generating intensity statistics boxplot...")
            stats_df = pd.DataFrame(intensity_stats)
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            metrics = ['mean', 'std', 'median', 'max']
            for idx, metric in enumerate(metrics):
                row, col = idx // 2, idx % 2
                axes[row, col].boxplot(stats_df[metric], vert=True)
                axes[row, col].set_ylabel('Intensity')
                axes[row, col].set_title(f'Intensity {metric.capitalize()} Distribution')
                axes[row, col].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'intensity_boxplots.png'), dpi=150, bbox_inches='tight')
            plt.close()
            viz_count += 1
            
            # 3. Overall Intensity Distribution
            self.logger.info("Generating overall intensity distribution...")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(slice_intensities, bins=100, color='teal', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Intensity Value')
            ax.set_ylabel('Frequency (log scale)')
            ax.set_yscale('log')
            ax.set_title('Overall Intensity Distribution (Sampled Slices)')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'intensity_distribution.png'), dpi=150, bbox_inches='tight')
            plt.close()
            viz_count += 1
            
            # 4. Class Distribution
            if class_counts:
                self.logger.info("Generating class distribution plot...")
                classes = sorted(class_counts.keys())
                counts = [class_counts[c] for c in classes]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar([f'Class {c}' for c in classes], counts, color='coral', edgecolor='black')
                ax.set_xlabel('Class')
                ax.set_ylabel('Total Voxel Count')
                ax.set_title('Class Distribution Across All Subjects')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add percentage labels
                total = sum(counts)
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    percentage = (count / total) * 100
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{percentage:.2f}%',
                           ha='center', va='bottom', fontsize=9)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150, bbox_inches='tight')
                plt.close()
                viz_count += 1
            
            # 5. Mask Coverage Distribution
            if mask_coverages:
                self.logger.info("Generating mask coverage distribution...")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(mask_coverages, bins=30, color='mediumseagreen', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Foreground Coverage Ratio')
                ax.set_ylabel('Number of Subjects')
                ax.set_title('Mask Coverage Distribution')
                ax.axvline(np.mean(mask_coverages), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(mask_coverages):.4f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'mask_coverage.png'), dpi=150, bbox_inches='tight')
                plt.close()
                viz_count += 1
            
            # 6. Intensity Statistics Heatmap
            self.logger.info("Generating intensity statistics correlation heatmap...")
            corr_metrics = ['mean', 'std', 'median', 'max', 'min', 'p1', 'p99']
            corr_df = stats_df[corr_metrics]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title('Intensity Metrics Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'intensity_correlations.png'), dpi=150, bbox_inches='tight')
            plt.close()
            viz_count += 1
            
            # 7. Per-subject intensity comparison
            self.logger.info("Generating per-subject intensity comparison...")
            fig, ax = plt.subplots(figsize=(max(12, len(subjects)*0.4), 6))
            x_pos = np.arange(len(stats_df))
            ax.errorbar(x_pos, stats_df['mean'], yerr=stats_df['std'], 
                       fmt='o', capsize=5, capthick=2, markersize=4)
            ax.set_xlabel('Subject Index')
            ax.set_ylabel('Mean Intensity ± Std')
            ax.set_title('Intensity Mean and Std per Subject')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'per_subject_intensity.png'), dpi=150, bbox_inches='tight')
            plt.close()
            viz_count += 1
            
            # Calculate summary statistics
            shape_mean = [float(shapes_df[dim].mean()) for dim in ['X', 'Y', 'Z']]
            shape_std = [float(shapes_df[dim].std()) for dim in ['X', 'Y', 'Z']]
            
            statistics = EDAStatistics(
                total_subjects=len(input_data.subjects),
                analyzed_subjects=len(subjects),
                volume_shape_mean=shape_mean,
                volume_shape_std=shape_std,
                intensity_mean=float(stats_df['mean'].mean()),
                intensity_std=float(stats_df['std'].mean()),
                mask_coverage_mean=float(np.mean(mask_coverages)) if mask_coverages else 0.0,
                class_distribution={str(k): int(v) for k, v in class_counts.items()}
            )
            
            # Generate analysis summary
            analysis_summary = f"""
EDA Summary Report
==================
Total Subjects: {statistics.total_subjects}
Analyzed Subjects: {statistics.analyzed_subjects}

Volume Shape Statistics:
- Mean Shape: {shape_mean[0]:.1f} × {shape_mean[1]:.1f} × {shape_mean[2]:.1f}
- Std Dev: {shape_std[0]:.1f} × {shape_std[1]:.1f} × {shape_std[2]:.1f}

Intensity Statistics:
- Mean Intensity (across subjects): {statistics.intensity_mean:.2f}
- Mean Std Dev (across subjects): {statistics.intensity_std:.2f}

Mask Statistics:
- Mean Foreground Coverage: {statistics.mask_coverage_mean:.4f} ({statistics.mask_coverage_mean*100:.2f}%)
- Class Distribution: {len(statistics.class_distribution)} classes detected

Visualizations Generated: {viz_count}
"""
            
            # Save report
            report_path = os.path.join(output_dir, 'eda_report.txt')
            with open(report_path, 'w') as f:
                f.write(analysis_summary)
            
            # Save detailed statistics as JSON
            stats_json_path = os.path.join(output_dir, 'eda_statistics.json')
            with open(stats_json_path, 'w') as f:
                json.dump({
                    'statistics': statistics.model_dump(),
                    'detailed_stats': intensity_stats,
                    'volume_shapes': volume_shapes,
                    'mask_coverages': mask_coverages,
                }, f, indent=2)
            
            self.logger.info(analysis_summary)
            self.logger.info("=" * 60)
            self.logger.info(f"EDA completed successfully. Generated {viz_count} visualizations.")
            self.logger.info(f"Results saved to: {output_dir}")
            self.logger.info("=" * 60)
            
            # Create HTML gallery for display_result
            self.logger.info("Generating HTML visualization gallery...")
            html_content = self._generate_html_gallery(output_dir, viz_count, analysis_summary)
            
            self.display_result = {
                "file_type": "html",
                "value": html_content
            }
            
            return OutputModel(
                statistics=statistics,
                report_path=report_path,
                visualization_dir=output_dir,
                num_visualizations=viz_count,
                analysis_summary=analysis_summary.strip()
            )
            
        except Exception as e:
            self.logger.error(f"Error in NiftiEDAPiece: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
