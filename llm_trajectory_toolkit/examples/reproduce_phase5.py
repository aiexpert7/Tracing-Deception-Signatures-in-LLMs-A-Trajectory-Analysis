"""
Reproduce Phase 5 Research Results

This example shows how to reproduce the statistical analysis from the research paper
using the toolkit API instead of the original scripts.
"""

import sys
sys.path.insert(0, '..')

import json
from pathlib import Path
from llm_trajectory_toolkit import TrajectoryAnalyzer, TrajectoryStorage

def load_dataset(dataset_path: str):
    """Load the Phase 2 dataset"""
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset['all_pairs']

def main():
    print("="*70)
    print("REPRODUCING PHASE 5 RESULTS WITH TOOLKIT")
    print("="*70)

    # Paths (adjust to your setup)
    dataset_path = "../../data/phase2_dataset/dataset.json"
    original_trajectories = "../../data/phase4_trajectories/GPT_2_trajectories.h5"

    # Option 1: Use existing trajectories
    if Path(original_trajectories).exists():
        print(f"\nLoading existing trajectories from {original_trajectories}")
        storage = TrajectoryStorage(original_trajectories)
        data = storage.load_features_only()

        print(f"✓ Loaded {len(data['categories'])} pairs")

        # Extract features as arrays
        import numpy as np
        truthful_array = np.array([
            [data['truthful'][feat][i] for feat in [
                'path_length', 'mean_step', 'mean_curvature', 'max_curvature',
                'mean_acceleration', 'straightness', 'direct_distance'
            ]]
            for i in range(len(data['categories']))
        ])

        deceptive_array = np.array([
            [data['deceptive'][feat][i] for feat in [
                'path_length', 'mean_step', 'mean_curvature', 'max_curvature',
                'mean_acceleration', 'straightness', 'direct_distance'
            ]]
            for i in range(len(data['categories']))
        ])

        # Statistical analysis
        from llm_trajectory_toolkit.analysis import StatisticalAnalyzer

        analyzer = StatisticalAnalyzer()

        feature_names = [
            'path_length', 'mean_step', 'mean_curvature', 'max_curvature',
            'mean_acceleration', 'straightness', 'direct_distance'
        ]

        results = analyzer.analyze_feature_set(
            truthful_array,
            deceptive_array,
            feature_names
        )

        # Apply Bonferroni correction
        p_values = [r.p_value for r in results.values()]
        bonferroni_alpha, significant = analyzer.bonferroni_correction(p_values)

        print("\n" + "="*70)
        print("STATISTICAL RESULTS (REPRODUCED)")
        print("="*70)

        analyzer.print_summary(results, bonferroni_alpha)

        print(f"\n{'='*70}")
        print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.6f}")
        print(f"Effects surviving correction: {sum(significant)}/{len(significant)}")
        print(f"{'='*70}")

        print("\nSurviving effects:")
        for feat, sig in zip(feature_names, significant):
            if sig:
                r = results[feat]
                print(f"  {feat:<20} d={r.cohens_d:>7.3f}, p={r.p_value:.6f}")

    # Option 2: Capture new trajectories (if dataset available)
    elif Path(dataset_path).exists():
        print(f"\nDataset found. To capture new trajectories:")
        print(f"  1. Load dataset from {dataset_path}")
        print(f"  2. Use TrajectoryAnalyzer.batch_compare()")
        print(f"  3. Save with analyzer.save_results()")
        print("\nFor demonstration, see batch_analysis.py example")

    else:
        print("\n⚠️  Neither trajectories nor dataset found")
        print("To reproduce Phase 5 results, you need either:")
        print(f"  1. Existing trajectories: {original_trajectories}")
        print(f"  2. Original dataset: {dataset_path}")

    print("\n" + "="*70)
    print("COMPARISON WITH ORIGINAL PHASE 5")
    print("="*70)
    print("\nOriginal script: scripts/phase5_statistical_analysis.py")
    print("Toolkit version: This script")
    print("\nBoth should produce identical statistical results.")
    print("="*70)

if __name__ == "__main__":
    main()
