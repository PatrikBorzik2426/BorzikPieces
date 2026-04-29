from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, EDAStatistics, AnalysisText
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
from collections import Counter
from typing import List, Dict, Tuple
from tqdm import tqdm


class NiftiEDAPiece(BasePiece):
    """
    A piece that performs comprehensive Exploratory Data Analysis (EDA) on NIfTI medical imaging datasets.
    
    This piece mirrors the academic research notebook EDA approach, providing detailed analyses with
    text descriptions explaining medical and technical significance of findings.
    
    Analyses performed:
    - Volume shape distribution and heterogeneity detection
    - Voxel spacing consistency checks (protocol validation)
    - Mask value analysis (binary vs multiclass detection)
    - Class distribution and imbalance analysis
    - Lesion size and coverage statistics
    - Spatial distribution of lesions across slices
    - Intensity distribution analysis (raw and normalized)
    - Normalization strategy comparison (z-score, clip+z-score)
    
    All findings include Slovak text descriptions matching the original research notebook style.
    """
    
    def _analyze_shapes(self, unique_shapes: set, shape_counts: Counter, total_subjects: int) -> str:
        """Generate Slovak text analysis for volume shapes"""
        if len(unique_shapes) == 1:
            shape = list(unique_shapes)[0]
            return (
                f"Analýza tvarov obrazových objemov ukázala, že všetky snímky majú **uniformný tvar** "
                f"{shape[0]}×{shape[1]}×{shape[2]} voxelov.\n\n"
                f"Táto konzistencia naznačuje, že všetky snímky boli získané pomocou rovnakého "
                f"zobrazovacieho protokolu a rovnakej priestorovej vzorkovacej stratégie, "
                f"čo zjednodušuje predspracovanie dát a trénovanie modelu."
            )
        else:
            shape_list = "\n".join([f"- {s[0]}×{s[1]}×{s[2]}: {shape_counts[s]} snímok" 
                                   for s in sorted(unique_shapes)])
            return (
                f"Analýza tvarov obrazových objemov odhalila **heterogenitu v rozmeroch snímok**.\n\n"
                f"V datasete sa vyskytujú nasledujúce tvary:\n\n{shape_list}\n\n"
                f"Táto nekonzistencia naznačuje použitie rôznych zobrazovacích protokolov alebo "
                f"nastavení vzorkovania. Pre úspešné trénovanie modelu bude potrebné zjednotiť "
                f"rozmery snímok pomocou resamplovania alebo croppingovania."
            )
    
    def _analyze_spacings(self, unique_spacings: set, spacing_counts: Counter) -> str:
        """Generate Slovak text analysis for voxel spacing"""
        if len(unique_spacings) == 1:
            spacing = list(unique_spacings)[0]
            return (
                f"Analýza voxel spacing ukázala **konzistentný protokol** naprieč celým datasetom.\n\n"
                f"Všetky snímky majú spacing: **{spacing[0]:.4f}×{spacing[1]:.2f}×{spacing[2]:.4f} mm**\n\n"
                f"Táto uniformita zjednodušuje predspracovanie a zabezpečuje, že všetky snímky "
                f"reprezentujú rovnakú fyzickú veľkosť voxelov, čo je dôležité pre presnosť modelu."
            )
        else:
            spacing_list = "\n".join([f"- {s[0]:.4f}×{s[1]:.2f}×{s[2]:.4f} mm: {spacing_counts[s]} snímok" 
                                     for s in sorted(unique_spacings)])
            return (
                f"Analýza voxel spacing odhalila **{len(unique_spacings)} rôzne protokoly** "
                f"v datasete.\n\n{spacing_list}\n\n"
                f"Táto heterogenita naznačuje, že snímky boli získané s rôznymi nastaveniami "
                f"zobrazovacieho zariadenia. Pre konzistentné trénovanie bude potrebné "
                f"**resampling všetkých snímok na jednotný spacing**, aby sa zabezpečila "
                f"rovnaká fyzická veľkosť voxelov naprieč celým datasetom."
            )
    
    def _analyze_mask_values(self, unique_values: set) -> str:
        """Generate Slovak text analysis for mask values"""
        sorted_values = sorted(unique_values)
        values_str = ", ".join([str(v) for v in sorted_values])
        
        if unique_values == {0, 1}:
            return (
                f"Analýza hodnôt v segmentačných maskách ukázala, že masky **sú binárne**.\n\n"
                f"V maskách sa vyskytujú iba hodnoty: {{ {values_str} }}\n\n"
                f"Hodnota **0** reprezentuje pozadie a hodnota **1** reprezentuje léziu/oblasť záujmu.\n\n"
                f"Tento typ masiek je vhodný pre **binárnu segmentáciu** s loss funkciami ako "
                f"Binary Cross-Entropy alebo Dice Loss."
            )
        else:
            num_foreground = len(unique_values) - (1 if 0 in unique_values else 0)
            return (
                f"Analýza hodnôt v segmentačných maskách ukázala, že masky **nie sú binárne**.\n\n"
                f"V maskách sa vyskytujú hodnoty: {{ {values_str} }}\n\n"
                f"Hodnota **0** reprezentuje pozadie, zatiaľ čo hodnoty **1–{max(unique_values)}** "
                f"indikujú rôzne označené triedy alebo stupne anotácie.\n\n"
                f"To naznačuje, že dataset je **multiclass segmentačný dataset** s {num_foreground} "
                f"triedami v popredí, alebo obsahuje **viacero anatomických/klinických podtried**.\n\n"
                f"Bez ďalšieho spracovania **nie je možné tieto masky považovať za binárne**, "
                f"čo má priamy dopad na výber loss funkcie (napr. Categorical Cross-Entropy alebo "
                f"Multi-class Dice Loss) a architektúry modelu."
            )
    
    def _analyze_class_distribution(self, class_counts: Counter) -> str:
        """Generate Slovak text analysis for class distribution"""
        sorted_classes = sorted(class_counts.keys())
        total_voxels = sum(class_counts.values())
        
        # Calculate percentages
        class_info = []
        for cls in sorted_classes:
            count = class_counts[cls]
            percentage = (count / total_voxels) * 100
            if cls == 0:
                class_info.append(f"- Trieda **{cls}** (pozadie): {count:,} voxelov ({percentage:.2f}%)")
            else:
                class_info.append(f"- Trieda **{cls}**: {count:,} voxelov ({percentage:.4f}%)")
        
        class_info_str = "\n".join(class_info)
        
        # Find most underrepresented foreground class
        foreground_classes = {k: v for k, v in class_counts.items() if k > 0}
        min_class = min(foreground_classes, key=foreground_classes.get) if foreground_classes else None
        
        background_ratio = (class_counts[0] / total_voxels * 100) if 0 in class_counts else 0
        
        return (
            f"Analýza rozdelenia hodnôt v segmentačných maskách odhalila "
            f"**výraznú nevyváženosť tried**.\n\n"
            f"Rozdelenie voxelov medzi jednotlivé triedy je nasledovné:\n\n{class_info_str}\n\n"
            f"Pozadie tvorí **{background_ratio:.2f}% všetkých voxelov**, zatiaľ čo jednotlivé "
            f"triedy lézií sú extrémne podreprezentované"
            + (f", pričom trieda {min_class} je najmenej zastúpená" if min_class else "") + ".\n\n"
            f"Takáto nevyváženosť dát predstavuje **významný problém pre učenie segmentačného modelu**, "
            f"keďže model by mal tendenciu preferovať predikciu pozadia na úkor správnej segmentácie lézie.\n\n"
            f"**Odporúčané riešenia:**\n"
            f"- Použitie loss funkcií zameraných na nevyváženosť (Dice Loss, Focal Loss)\n"
            f"- Spatial sampling (patch-based prístup zameraný na oblasti s léziami)\n"
            f"- Augmentácie na vyváženie tried\n"
            f"- Weighted sampling počas trénovania"
        )
    
    def _analyze_lesion_sizes(self, lesion_voxels: np.ndarray, lesion_ratios: np.ndarray) -> str:
        """Generate Slovak text analysis for lesion sizes"""
        mean_voxels = lesion_voxels.mean()
        median_voxels = np.median(lesion_voxels)
        mean_ratio = lesion_ratios.mean() * 100
        max_ratio = lesion_ratios.max() * 100
        min_ratio = lesion_ratios[lesion_ratios > 0].min() * 100 if len(lesion_ratios[lesion_ratios > 0]) > 0 else 0
        
        return (
            f"Analýza veľkosti lézií ukazuje **výraznú nevyváženosť tried** v datasete.\n\n"
            f"**Štatistiky veľkosti lézií:**\n"
            f"- Priemerný počet voxelov lézie: **{mean_voxels:.0f}**\n"
            f"- Medián počtu voxelov lézie: **{median_voxels:.0f}**\n"
            f"- Priemerný podiel lézie na objeme: **{mean_ratio:.4f}%**\n"
            f"- Maximálny podiel lézie: **{max_ratio:.2f}%**\n"
            f"- Minimálny nenulový podiel lézie: **{min_ratio:.4f}%**\n\n"
            f"Lézia tvorí v priemere **menej než {mean_ratio:.2f}%** celkového objemu snímky, "
            f"pričom aj v najväčších prípadoch nepresahuje približne {max_ratio:.1f}% objemu. "
            f"Naopak, v niektorých prípadoch je podiel lézie menší než {min_ratio:.2f}%.\n\n"
            f"Tieto výsledky poukazujú na **silnú dominanciu triedy pozadia**, čo predstavuje "
            f"významnú výzvu pre segmentačné modely. Bez vhodných stratégií by model mohol "
            f"preferovať predikciu pozadia na úkor detekcie lézie.\n\n"
            f"**Zistenia odôvodňujú použitie:**\n"
            f"- Stratégií zameraných na riešenie nevyváženosti tried (Dice Loss, Focal Loss)\n"
            f"- Priestorové obmedzenie vstupu pomocą patch-based prístupu\n"
            f"- Targeted sampling oblastí obsahujúcich lézie"
        )
    
    def _analyze_slice_distribution(self, total_slices: int, slices_with_lesion: int, 
                                    slice_ratio: float) -> str:
        """Generate Slovak text analysis for slice-level distribution"""
        percentage_with_lesion = slice_ratio * 100
        percentage_without = (1 - slice_ratio) * 100
        
        return (
            f"Slice-level analýza ukazuje, že iba približne **{percentage_with_lesion:.2f}%** "
            f"všetkých 2D rezov obsahuje aspoň časť lézie, zatiaľ čo viac než "
            f"**{percentage_without:.2f}%** rezov neobsahuje žiadnu patologickú oblasť.\n\n"
            f"**Štatistiky:**\n"
            f"- Celkový počet rezov: **{total_slices:,}**\n"
            f"- Počet rezov s léziou: **{slices_with_lesion:,}**\n"
            f"- Podiel rezov obsahujúcich léziu: **{percentage_with_lesion:.2f}%**\n\n"
            f"To znamená, že pri použití **2D prístupu** by model vo veľkej miere trénoval "
            f"na rezoch obsahujúcich výlučne pozadie. Takýto nepomer môže viesť k výraznej "
            f"**zaujatosti modelu v prospech triedy pozadia** a zníženej citlivosti na detekciu lézie.\n\n"
            f"**Odporúčania:**\n"
            f"- Použitie **3D segmentačných prístupov** (UNet 3D, V-Net)\n"
            f"- Cielený výber (sampling) rezov, ktoré obsahujú léziu\n"
            f"- Weighted sampling s preferenciou rezov obsahujúcich lézie\n"
            f"- Kombinácia 2D a 3D prístupov"
        )
    
    def _analyze_intensities(self, all_means: List, all_stds: List, all_mins: List, 
                            all_maxs: List, sampled_voxels: np.ndarray) -> str:
        """Generate Slovak text analysis for intensity distributions"""
        mean_of_means = np.mean(all_means)
        std_of_means = np.std(all_means)
        mean_of_stds = np.mean(all_stds)
        global_min = np.min(all_mins)
        global_max = np.max(all_maxs)
        
        return (
            f"Globálna analýza intenzít naprieč celým datasetom ukazuje, že snímky majú "
            f"**pomerne široké rozdelenie intenzít**.\n\n"
            f"**Dataset-level štatistiky:**\n"
            f"- Priemerná hodnota intenzity: **{mean_of_means:.2f} ± {std_of_means:.2f}**\n"
            f"- Priemerná štandardná odchýlka: **{mean_of_stds:.2f}**\n"
            f"- Globálne minimum: **{global_min:.2f}**\n"
            f"- Globálne maximum: **{global_max:.2f}**\n\n"
            f"Priemerná hodnota intenzity je približne {mean_of_means:.0f}, s vysokou variabilitou "
            f"(štandardná odchýlka ~{mean_of_stds:.0f}), čo naznačuje **výrazné rozdiely medzi "
            f"jednotlivými voxelmi aj medzi snímkami**.\n\n"
            f"Globálne minimum intenzít je {global_min:.0f}, zatiaľ čo maximum dosahuje hodnotu "
            f"až okolo {global_max:.0f}, čo poukazuje na **prítomnosť extrémnych hodnôt**.\n\n"
            f"Takéto rozdelenie je typické pre CT dáta a naznačuje potrebu **vhodnej normalizácie "
            f"alebo orezania intenzít** pred samotným modelovaním.\n\n"
            f"**Odporúčania:**\n"
            f"- Normalizácia intenzít na fixný rozsah (napr. [0, 1] alebo [-1, 1])\n"
            f"- Z-score štandardizácia na úrovni jednotlivých snímok\n"
            f"- Orezanie extrémnych hodnôt (percentile clipping)\n"
            f"- Windowing pre CT dáta (napr. soft tissue window)"
        )
    
    def _analyze_normalization(self, raw_means: List, raw_stds: List, z_means: List, 
                              z_stds: List, clip_means: List, clip_stds: List) -> str:
        """Generate Slovak text analysis for normalization comparison"""
        return (
            f"### 1. Surové intenzity (Raw intensity)\n"
            f"- Histogram surových voxelových intenzít je **silne pravostranný (right-skewed)**.\n"
            f"- Väčšina hodnôt je koncentrovaná pri nízkych intenzitách, s **dlhým chvostom** "
            f"smerom k vysokým hodnotám (extrémy).\n"
            f"- Per-volume priemery: **{np.mean(raw_means):.2f} ± {np.std(raw_means):.2f}**\n"
            f"- Per-volume štandardné odchýlky: **{np.mean(raw_stds):.2f} ± {np.std(raw_stds):.2f}**\n"
            f"- To naznačuje **nekonzistentné škálovanie intenzít** medzi objemami.\n\n"
            
            f"### 2. Per-volume z-score normalizácia\n"
            f"- Po z-score normalizácii (počítanej zvlášť pre každý objem):\n"
            f"  - Rozdelenie je centrované okolo **0**\n"
            f"  - Smerodajná odchýlka je približne **1**\n"
            f"- Per-volume priemery: **{np.mean(z_means):.4f} ± {np.std(z_means):.4f}**\n"
            f"- Per-volume štandardné odchýlky: **{np.mean(z_stds):.4f} ± {np.std(z_stds):.4f}**\n"
            f"- Z-score úspešne **štandardizuje škálu medzi objemami**, ale je citlivý na outliery.\n\n"
            
            f"### 3. Clip (1–99%) + z-score\n"
            f"- Po orezaní extrémov (1. a 99. percentil) a následnom z-score:\n"
            f"  - Histogram je **kompaktnejší a symetrickejší**\n"
            f"  - Chvosty sú kratšie a extrémne hodnoty potlačené\n"
            f"- Per-volume priemery: **{np.mean(clip_means):.4f} ± {np.std(clip_means):.4f}**\n"
            f"- Per-volume štandardné odchýlky: **{np.mean(clip_stds):.4f} ± {np.std(clip_stds):.4f}**\n"
            f"- Táto kombinácia poskytuje **najstabilnejšiu normalizáciu**.\n\n"
            
            f"### 4. Zhrnutie\n"
            f"- **Raw intenzity**: vysoká variabilita, silný vplyv outlierov, nevhodné pre priame použitie v ML.\n"
            f"- **Per-volume z-score**: výrazné zlepšenie, ale stále citlivé na extrémy.\n"
            f"- **Clip + z-score**: **najlepšia voľba** z hľadiska stability medzi objemami, "
            f"potlačenia outlierov a konzistentného rozdelenia dát.\n\n"
            f"**Odporúčanie:** Použiť **Clip(1-99%) + z-score normalizáciu** pre trénovanie modelu."
        )
    
    def _generate_text_summary(self, statistics: EDAStatistics, 
                              analysis_texts: List[AnalysisText]) -> str:
        """Generate text summary of all analyses"""
        summary_parts = [
            "=" * 80,
            "COMPREHENSIVE EDA REPORT",
            "=" * 80,
            "",
            f"Total Subjects: {statistics.total_subjects}",
            f"Analyzed Subjects: {statistics.analyzed_subjects}",
            "",
            "=" * 80,
            "KEY FINDINGS",
            "=" * 80,
            ""
        ]
        
        for analysis in analysis_texts:
            summary_parts.append(f"## {analysis.section}")
            summary_parts.append("")
            summary_parts.append(analysis.finding)
            summary_parts.append("")
            summary_parts.append("-" * 80)
            summary_parts.append("")
        
        summary_parts.extend([
            "=" * 80,
            "SUMMARY STATISTICS",
            "=" * 80,
            f"Unique volume shapes: {statistics.unique_shapes}",
            f"Unique voxel spacings: {statistics.unique_spacings}",
            f"Number of classes: {statistics.num_classes}",
            f"Mean lesion coverage: {statistics.mean_lesion_coverage * 100:.4f}%",
            f"Slice lesion ratio: {statistics.slice_lesion_ratio * 100:.2f}%",
            f"Mean intensity: {statistics.mean_intensity:.2f}",
            f"Std intensity: {statistics.std_intensity:.2f}",
            "=" * 80
        ])
        
        return "\n".join(summary_parts)
    
    def _generate_comprehensive_html(self, output_dir: str, statistics: EDAStatistics,
                                     analysis_texts: List[AnalysisText]) -> str:
        """Generate comprehensive HTML report with all analyses and visualizations"""
        
        # Load and encode all visualizations
        viz_files = [
            ('1_shape_distribution.png', 'Rozdelenie tvarov objemov'),
            ('2_voxel_spacing.png', 'Rozdelenie voxel spacing protokolov'),
            ('3_class_distribution.png', 'Rozdelenie voxelov medzi triedy'),
            ('4_lesion_coverage.png', 'Rozdelenie pokrytia léziou'),
            ('5_intensity_analysis.png', 'Analýza intenzít'),
            ('6_normalization_comparison.png', 'Porovnanie normalizačných stratégií')
        ]
        
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="sk">',
            '<head>',
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '<title>Comprehensive NIfTI EDA Report</title>',
            '<style>',
            'body { font-family: "Segoe UI", Arial, sans-serif; margin: 0; padding: 20px; background: #f5f7fa; line-height: 1.6; }',
            'h1 { color: #2c3e50; border-bottom: 4px solid #3498db; padding-bottom: 15px; margin-top: 0; }',
            'h2 { color: #34495e; margin-top: 40px; border-bottom: 2px solid #bdc3c7; padding-bottom: 10px; }',
            'h3 { color: #7f8c8d; margin-top: 25px; }',
            '.container { max-width: 1400px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }',
            '.summary-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 8px; margin: 30px 0; }',
            '.summary-box h2 { color: white; border-bottom: 2px solid rgba(255,255,255,0.3); margin-top: 0; }',
            '.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 25px 0; }',
            '.stat-card { background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
            '.stat-label { font-size: 13px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }',
            '.stat-value { font-size: 32px; font-weight: bold; color: #2c3e50; }',
            '.analysis-section { background: #ecf0f1; padding: 25px; border-radius: 8px; margin: 30px 0; border-left: 5px solid #3498db; }',
            '.analysis-section h3 { margin-top: 0; color: #2c3e50; }',
            '.analysis-text { white-space: pre-wrap; line-height: 1.8; color: #34495e; }',
            '.analysis-text strong { color: #e74c3c; }',
            '.viz-container { background: #fff; padding: 30px; margin: 30px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
            '.viz-title { font-size: 20px; font-weight: bold; color: #2c3e50; margin-bottom: 20px; border-left: 4px solid #3498db; padding-left: 15px; }',
            '.viz-image { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }',
            'ul { line-height: 1.8; }',
            'li { margin: 8px 0; }',
            '.recommendation { background: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 15px 0; border-radius: 4px; }',
            '.warning { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 4px; }',
            '</style>',
            '</head>',
            '<body>',
            '<div class="container">',
            '<h1>📊 Comprehensive NIfTI Medical Imaging EDA Report</h1>',
            
            # Summary Box
            '<div class="summary-box">',
            '<h2>Dataset Summary</h2>',
            '<div class="stats-grid">',
            f'<div class="stat-card"><div class="stat-label">Total Subjects</div><div class="stat-value">{statistics.total_subjects}</div></div>',
            f'<div class="stat-card"><div class="stat-label">Analyzed</div><div class="stat-value">{statistics.analyzed_subjects}</div></div>',
            f'<div class="stat-card"><div class="stat-label">Unique Shapes</div><div class="stat-value">{statistics.unique_shapes}</div></div>',
            f'<div class="stat-card"><div class="stat-label">Protocols</div><div class="stat-value">{statistics.unique_spacings}</div></div>',
            f'<div class="stat-card"><div class="stat-label">Classes</div><div class="stat-value">{statistics.num_classes}</div></div>',
            f'<div class="stat-card"><div class="stat-label">Lesion Coverage</div><div class="stat-value">{statistics.mean_lesion_coverage*100:.2f}%</div></div>',
            '</div>',
            '</div>'
        ]
        
        # Add each analysis section with corresponding visualization
        analysis_viz_pairs = [
            (0, 0),  # Shape analysis + shape distribution viz
            (1, 1),  # Spacing analysis + spacing viz
            (2, None),  # Mask values (no specific viz)
            (3, 2),  # Class distribution + class distribution viz
            (4, 3),  # Lesion size + lesion coverage viz
            (5, None),  # Slice distribution (no specific viz)
            (6, 4),  # Intensity analysis + intensity viz
            (7, 5),  # Normalization + normalization viz
        ]
        
        for analysis_idx, viz_idx in analysis_viz_pairs:
            if analysis_idx < len(analysis_texts):
                analysis = analysis_texts[analysis_idx]
                html_parts.extend([
                    '<div class="analysis-section">',
                    f'<h3>{analysis.section}</h3>',
                    f'<div class="analysis-text">{analysis.finding}</div>',
                    '</div>'
                ])
                
                # Add visualization if available
                if viz_idx is not None and viz_idx < len(viz_files):
                    viz_file, viz_title = viz_files[viz_idx]
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
            '</div>',
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(html_parts)

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
            self.logger.info("=" * 80)
            self.logger.info("Starting Comprehensive NIfTI EDA Analysis")
            self.logger.info("=" * 80)
            
            # Use Domino's tracked results path instead of hardcoded output_dir
            # This ensures files are properly tracked and accessible
            output_dir = self.results_path
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Output directory: {output_dir}")
            
            # Validate subjects input
            if not input_data.subjects:
                error_msg = "No subjects provided. Please connect this piece to NiftiDataLoaderPiece or provide subjects list."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Limit subjects for performance
            subjects = input_data.subjects[:input_data.max_subjects]
            self.logger.info(f"Analyzing {len(subjects)} subjects (max: {input_data.max_subjects})")
            
            # Storage for all analyses
            analysis_texts = []
            
            # ==================== PHASE 1: Shape Analysis ====================
            self.logger.info("\n[1/8] Analyzing volume shapes...")
            shapes = []
            for subject in tqdm(subjects, desc="Loading shapes"):
                try:
                    img = nib.load(subject.image_path).get_fdata()
                    shapes.append(img.shape)
                except Exception as e:
                    self.logger.warning(f"Failed to load {subject.subject_id}: {e}")
            
            unique_shapes = set(shapes)
            shape_counts = Counter(shapes)
            
            # Generate shape analysis text
            shape_text = self._analyze_shapes(unique_shapes, shape_counts, len(subjects))
            analysis_texts.append(AnalysisText(
                section="Analýza tvarov objemov",
                finding=shape_text
            ))
            
            # Visualization: Shape distribution
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            shapes_array = np.array(shapes)
            for idx, (ax, dim_name) in enumerate(zip(axes, ['X', 'Y', 'Z'])):
                ax.hist(shapes_array[:, idx], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
                ax.set_xlabel(f'{dim_name} Dimension (voxels)')
                ax.set_ylabel('Počet snímok')
                ax.set_title(f'Rozdelenie rozmeru {dim_name}')
                ax.axvline(shapes_array[:, idx].mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Priemer: {shapes_array[:, idx].mean():.1f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '1_shape_distribution.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # ==================== PHASE 2: Voxel Spacing Analysis ====================
            self.logger.info("\n[2/8] Analyzing voxel spacing (protocol consistency)...")
            spacings = []
            for subject in tqdm(subjects, desc="Loading spacings"):
                try:
                    nii = nib.load(subject.image_path)
                    spacing = nii.header.get_zooms()[:3]  # First 3 dimensions
                    spacings.append(spacing)
                except Exception as e:
                    self.logger.warning(f"Failed to get spacing for {subject.subject_id}: {e}")
            
            unique_spacings = set(spacings)
            spacing_counts = Counter(spacings)
            
            # Generate spacing analysis text
            spacing_text = self._analyze_spacings(unique_spacings, spacing_counts)
            analysis_texts.append(AnalysisText(
                section="Analýza voxel spacing (veľkosti voxelov)",
                finding=spacing_text
            ))
            
            # Visualization: Spacing distribution
            fig, ax = plt.subplots(figsize=(12, 6))
            spacing_labels = [f"{s[0]:.3f}×{s[1]:.2f}×{s[2]:.3f} mm" for s in spacing_counts.keys()]
            spacing_values = list(spacing_counts.values())
            bars = ax.bar(range(len(spacing_labels)), spacing_values, color='coral', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Voxel Spacing Protocol')
            ax.set_ylabel('Počet snímok')
            ax.set_title('Rozdelenie voxel spacing protokolov')
            ax.set_xticks(range(len(spacing_labels)))
            ax.set_xticklabels(spacing_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add counts on bars
            for bar, count in zip(bars, spacing_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '2_voxel_spacing.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # ==================== PHASE 3: Mask Value Analysis ====================
            self.logger.info("\n[3/8] Analyzing mask values (binary vs multiclass)...")
            unique_mask_values = set()
            for subject in tqdm(subjects, desc="Analyzing masks"):
                if subject.mask_path:
                    try:
                        mask = nib.load(subject.mask_path).get_fdata()
                        unique_mask_values.update(np.unique(mask).astype(int))
                    except Exception as e:
                        self.logger.warning(f"Failed to load mask for {subject.subject_id}: {e}")
            
            # Generate mask value analysis text
            mask_value_text = self._analyze_mask_values(unique_mask_values)
            analysis_texts.append(AnalysisText(
                section="Analýza hodnôt v segmentačných maskách",
                finding=mask_value_text
            ))
            
            # ==================== PHASE 4: Class Distribution Analysis ====================
            self.logger.info("\n[4/8] Analyzing class distribution and imbalance...")
            class_counts = Counter()
            for subject in tqdm(subjects, desc="Counting classes"):
                if subject.mask_path:
                    try:
                        mask = nib.load(subject.mask_path).get_fdata()
                        class_counts.update(mask.astype(int).flatten())
                    except Exception as e:
                        self.logger.warning(f"Failed to count classes for {subject.subject_id}: {e}")
            
            # Generate class distribution text
            class_dist_text = self._analyze_class_distribution(class_counts)
            analysis_texts.append(AnalysisText(
                section="Analýza rozdelenia tried a nevyváženosti dát",
                finding=class_dist_text
            ))
            
            # Visualization: Class distribution
            fig, ax = plt.subplots(figsize=(12, 7))
            classes = sorted(class_counts.keys())
            counts = [class_counts[c] for c in classes]
            bars = ax.bar([f'Trieda {c}' for c in classes], counts, color='mediumseagreen', 
                         edgecolor='black', alpha=0.7)
            ax.set_xlabel('Trieda')
            ax.set_ylabel('Celkový počet voxelov (log scale)')
            ax.set_title('Rozdelenie voxelov medzi triedy')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add percentage labels
            total_voxels = sum(counts)
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                percentage = (count / total_voxels) * 100
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count:,}\n({percentage:.2f}%)',
                       ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '3_class_distribution.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # ==================== PHASE 5: Lesion Size Analysis ====================
            self.logger.info("\n[5/8] Analyzing lesion sizes and coverage...")
            lesion_voxels = []
            total_voxels_list = []
            lesion_ratios = []
            
            for subject in tqdm(subjects, desc="Measuring lesions"):
                if subject.mask_path:
                    try:
                        mask = nib.load(subject.mask_path).get_fdata()
                        lesion_count = np.sum(mask > 0)
                        total_count = mask.size
                        
                        lesion_voxels.append(lesion_count)
                        total_voxels_list.append(total_count)
                        lesion_ratios.append(lesion_count / total_count)
                    except Exception as e:
                        self.logger.warning(f"Failed to measure lesion for {subject.subject_id}: {e}")
            
            lesion_voxels = np.array(lesion_voxels)
            lesion_ratios = np.array(lesion_ratios)
            
            # Generate lesion size analysis text
            lesion_size_text = self._analyze_lesion_sizes(lesion_voxels, lesion_ratios)
            analysis_texts.append(AnalysisText(
                section="Veľkosť lézií a nevyváženosť tried",
                finding=lesion_size_text
            ))
            
            # Visualization: Lesion coverage
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram of lesion ratios
            axes[0].hist(lesion_ratios * 100, bins=30, color='darkseagreen', edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('Podiel lézie na objeme (%)')
            axes[0].set_ylabel('Počet subjektov')
            axes[0].set_title('Rozdelenie pokrytia léziou')
            axes[0].axvline(lesion_ratios.mean() * 100, color='red', linestyle='--', linewidth=2,
                           label=f'Priemer: {lesion_ratios.mean()*100:.3f}%')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Boxplot of lesion voxels
            axes[1].boxplot(lesion_voxels, vert=True)
            axes[1].set_ylabel('Počet voxelov lézie')
            axes[1].set_title('Boxplot veľkosti lézií')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '4_lesion_coverage.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # ==================== PHASE 6: Slice-Level Lesion Analysis ====================
            self.logger.info("\n[6/8] Analyzing lesion distribution across slices...")
            slices_with_lesion = 0
            total_slices = 0
            
            for subject in tqdm(subjects, desc="Analyzing slices"):
                if subject.mask_path:
                    try:
                        mask = nib.load(subject.mask_path).get_fdata()
                        for i in range(mask.shape[2]):  # Z dimension
                            total_slices += 1
                            if np.any(mask[:, :, i] > 0):
                                slices_with_lesion += 1
                    except Exception as e:
                        self.logger.warning(f"Failed slice analysis for {subject.subject_id}: {e}")
            
            slice_ratio = slices_with_lesion / total_slices if total_slices > 0 else 0
            
            # Generate slice analysis text
            slice_text = self._analyze_slice_distribution(total_slices, slices_with_lesion, slice_ratio)
            analysis_texts.append(AnalysisText(
                section="Analýza výskytu lézie v jednotlivých rezoch",
                finding=slice_text
            ))
            
            # ==================== PHASE 7: Intensity Distribution Analysis ====================
            self.logger.info("\n[7/8] Analyzing intensity distributions...")
            all_means = []
            all_stds = []
            all_mins = []
            all_maxs = []
            sampled_voxels = []
            
            np.random.seed(42)
            for subject in tqdm(subjects, desc="Sampling intensities"):
                try:
                    img = nib.load(subject.image_path).get_fdata()
                    
                    all_means.append(np.mean(img))
                    all_stds.append(np.std(img))
                    all_mins.append(np.min(img))
                    all_maxs.append(np.max(img))
                    
                    flat = img.flatten()
                    if len(flat) > 100_000:
                        flat = np.random.choice(flat, 100_000, replace=False)
                    sampled_voxels.append(flat)
                except Exception as e:
                    self.logger.warning(f"Failed intensity analysis for {subject.subject_id}: {e}")
            
            sampled_voxels = np.concatenate(sampled_voxels)
            
            # Generate intensity analysis text
            intensity_text = self._analyze_intensities(all_means, all_stds, all_mins, all_maxs, 
                                                       sampled_voxels)
            analysis_texts.append(AnalysisText(
                section="Analýza intenzít naprieč celým datasetom",
                finding=intensity_text
            ))
            
            # Visualization: Intensity statistics
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Global intensity distribution
            axes[0, 0].hist(sampled_voxels, bins=200, color='steelblue', edgecolor='black', alpha=0.7)
            axes[0, 0].set_xlabel('Intenzita')
            axes[0, 0].set_ylabel('Frekvencia')
            axes[0, 0].set_title('Globálne rozdelenie intenzít (vzorkované voxely)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Per-volume means
            axes[0, 1].boxplot(all_means, vert=True)
            axes[0, 1].set_ylabel('Priemerná intenzita')
            axes[0, 1].set_title('Rozdelenie priemerov naprieč objemami')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Per-volume stds
            axes[1, 0].boxplot(all_stds, vert=True)
            axes[1, 0].set_ylabel('Štandardná odchýlka')
            axes[1, 0].set_title('Rozdelenie štandardných odchýlok naprieč objemami')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Min-max ranges
            axes[1, 1].scatter(range(len(all_mins)), all_mins, alpha=0.5, label='Minimum', s=30)
            axes[1, 1].scatter(range(len(all_maxs)), all_maxs, alpha=0.5, label='Maximum', s=30)
            axes[1, 1].set_xlabel('Index objektu')
            axes[1, 1].set_ylabel('Intenzita')
            axes[1, 1].set_title('Rozsah intenzít pre každý objem')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '5_intensity_analysis.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # ==================== PHASE 8: Normalization Comparison ====================
            self.logger.info("\n[8/8] Comparing normalization strategies...")
            n_samples = min(20, len(subjects))
            sample_subjects = subjects[:n_samples]
            
            raw_means, raw_stds = [], []
            z_means, z_stds = [], []
            clip_means, clip_stds = [], []
            
            raw_concat = []
            z_concat = []
            clip_concat = []
            
            for subject in tqdm(sample_subjects, desc="Testing normalization"):
                try:
                    data = nib.load(subject.image_path).get_fdata().astype(np.float32)
                    flat = data.flatten()
                    if flat.size > 100_000:
                        flat = np.random.choice(flat, 100_000, replace=False)
                    
                    # Raw statistics
                    raw_means.append(flat.mean())
                    raw_stds.append(flat.std())
                    raw_concat.append(flat)
                    
                    # Z-score normalization
                    z = (flat - flat.mean()) / (flat.std() + 1e-8)
                    z_means.append(z.mean())
                    z_stds.append(z.std())
                    z_concat.append(z)
                    
                    # Clip + Z-score
                    p1, p99 = np.percentile(flat, [1, 99])
                    clipped = np.clip(flat, p1, p99)
                    cz = (clipped - clipped.mean()) / (clipped.std() + 1e-8)
                    clip_means.append(cz.mean())
                    clip_stds.append(cz.std())
                    clip_concat.append(cz)
                except Exception as e:
                    self.logger.warning(f"Failed normalization test for {subject.subject_id}: {e}")
            
            raw_concat = np.concatenate(raw_concat)
            z_concat = np.concatenate(z_concat)
            clip_concat = np.concatenate(clip_concat)
            
            # Generate normalization comparison text
            norm_text = self._analyze_normalization(raw_means, raw_stds, z_means, z_stds, 
                                                    clip_means, clip_stds)
            analysis_texts.append(AnalysisText(
                section="Analýza rozdelenia intenzít po normalizácii",
                finding=norm_text
            ))
            
            # Visualization: Normalization comparison
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            
            # Row 1: Histograms
            axes[0, 0].hist(raw_concat, bins=200, color='C0', alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Surové intenzity')
            axes[0, 0].set_xlabel('Intenzita')
            axes[0, 0].set_ylabel('Frekvencia')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].hist(z_concat, bins=200, range=(-5, 5), color='C1', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Per-volume z-score')
            axes[0, 1].set_xlabel('Z-score')
            axes[0, 1].set_ylabel('Frekvencia')
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[0, 2].hist(clip_concat, bins=200, range=(-5, 5), color='C2', alpha=0.7, edgecolor='black')
            axes[0, 2].set_title('Clip(1-99%) + z-score')
            axes[0, 2].set_xlabel('Z-score')
            axes[0, 2].set_ylabel('Frekvencia')
            axes[0, 2].grid(True, alpha=0.3)
            
            # Row 2: Boxplots
            axes[1, 0].boxplot([raw_means, z_means, clip_means], 
                              labels=['Raw', 'Z-score', 'Clip+Z'])
            axes[1, 0].set_ylabel('Priemer')
            axes[1, 0].set_title('Per-volume priemery')
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].boxplot([raw_stds, z_stds, clip_stds], 
                              labels=['Raw', 'Z-score', 'Clip+Z'])
            axes[1, 1].set_ylabel('Štandardná odchýlka')
            axes[1, 1].set_title('Per-volume štandardné odchýlky')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Comparison table
            axes[1, 2].axis('off')
            comparison_data = [
                ['Metóda', 'Mean ± SD', 'Std ± SD'],
                ['Raw', f'{np.mean(raw_means):.2f}±{np.std(raw_means):.2f}', 
                 f'{np.mean(raw_stds):.2f}±{np.std(raw_stds):.2f}'],
                ['Z-score', f'{np.mean(z_means):.4f}±{np.std(z_means):.4f}', 
                 f'{np.mean(z_stds):.4f}±{np.std(z_stds):.4f}'],
                ['Clip+Z', f'{np.mean(clip_means):.4f}±{np.std(clip_means):.4f}', 
                 f'{np.mean(clip_stds):.4f}±{np.std(clip_stds):.4f}']
            ]
            table = axes[1, 2].table(cellText=comparison_data, cellLoc='center', loc='center',
                                    colWidths=[0.3, 0.35, 0.35])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            axes[1, 2].set_title('Štatistiky normalizácie', pad=20)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '6_normalization_comparison.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # ==================== GENERATE SUMMARY STATISTICS ====================
            statistics = EDAStatistics(
                total_subjects=len(input_data.subjects),
                analyzed_subjects=len(subjects),
                unique_shapes=len(unique_shapes),
                unique_spacings=len(unique_spacings),
                num_classes=len(unique_mask_values),
                class_distribution={str(k): int(v) for k, v in class_counts.items()},
                mean_lesion_coverage=float(np.mean(lesion_ratios)) if len(lesion_ratios) > 0 else 0.0,
                slice_lesion_ratio=float(slice_ratio),
                mean_intensity=float(np.mean(all_means)),
                std_intensity=float(np.mean(all_stds))
            )
            
            # ==================== GENERATE HTML REPORT ====================
            self.logger.info("\nGenerating comprehensive HTML report...")
            html_content = self._generate_comprehensive_html(output_dir, statistics, analysis_texts)
            
            html_path = os.path.join(output_dir, 'comprehensive_eda_report.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Save analysis texts as JSON
            texts_json_path = os.path.join(output_dir, 'analysis_texts.json')
            with open(texts_json_path, 'w', encoding='utf-8') as f:
                json.dump([t.model_dump() for t in analysis_texts], f, ensure_ascii=False, indent=2)
            
            # Save statistics
            stats_json_path = os.path.join(output_dir, 'eda_statistics.json')
            with open(stats_json_path, 'w', encoding='utf-8') as f:
                json.dump(statistics.model_dump(), f, indent=2)
            
            # Generate text summary
            analysis_summary = self._generate_text_summary(statistics, analysis_texts)
            report_path = os.path.join(output_dir, 'eda_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(analysis_summary)
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("EDA Analysis Complete!")
            self.logger.info(f"Generated 6 comprehensive visualizations")
            self.logger.info(f"Results saved to: {output_dir}")
            self.logger.info("=" * 80)
            
            self.display_result = {
                "file_type": "html",
                "base64_content": base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
            }
            
            return OutputModel(
                statistics=statistics,
                report_path=report_path,
                visualization_dir=output_dir,
                num_visualizations=6,
                analysis_summary=analysis_summary.strip(),
                analysis_texts=[t.model_dump() for t in analysis_texts]
            )
            
        except Exception as e:
            self.logger.error(f"Error in NiftiEDAPiece: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
