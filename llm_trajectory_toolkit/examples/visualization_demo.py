#!/usr/bin/env python3
"""
Visualization Demo

Demonstrates the new visualization capabilities of the toolkit.
"""

import numpy as np
from llm_trajectory_toolkit.visualization import TrajectoryPlotter, StatsVisualizer
from llm_trajectory_toolkit.analysis.statistical import StatisticalAnalyzer, PairedTestResult

print("="*70)
print("LLM TRAJECTORY TOOLKIT - VISUALIZATION DEMO")
print("="*70)

# ============================================================================
# 1. Generate Mock Data
# ============================================================================
print("\n[1/4] Generating mock trajectory data...")

np.random.seed(42)

# Create mock trajectories (simulating the research findings)
n_samples = 20
n_layers = 12
hidden_dim = 768

truthful_trajectories = []
deceptive_trajectories = []

for i in range(n_samples):
    # Truthful: shorter paths, more curved
    start = np.random.randn(hidden_dim)
    truthful_traj = [start]
    for layer in range(1, n_layers):
        # Smaller steps, more curvature
        step = np.random.randn(hidden_dim) * 0.8
        truthful_traj.append(truthful_traj[-1] + step)
    truthful_trajectories.append(np.array(truthful_traj))

    # Deceptive: longer paths, straighter
    start = np.random.randn(hidden_dim)
    deceptive_traj = [start]
    for layer in range(1, n_layers):
        # Larger steps, straighter direction
        if layer == 1:
            direction = np.random.randn(hidden_dim)
            direction = direction / np.linalg.norm(direction)
        step = direction * 1.2 + np.random.randn(hidden_dim) * 0.2
        deceptive_traj.append(deceptive_traj[-1] + step)
    deceptive_trajectories.append(np.array(deceptive_traj))

print(f"  ✓ Created {n_samples} trajectory pairs")

# Extract features (mock)
from llm_trajectory_toolkit.core.geometric_analysis import GeometricAnalyzer

analyzer = GeometricAnalyzer()

truthful_features = []
deceptive_features = []

for t_traj, d_traj in zip(truthful_trajectories, deceptive_trajectories):
    truthful_features.append(analyzer.analyze(t_traj).to_array())
    deceptive_features.append(analyzer.analyze(d_traj).to_array())

truthful_features = np.array(truthful_features)
deceptive_features = np.array(deceptive_features)

print(f"  ✓ Extracted features: {truthful_features.shape}")

# ============================================================================
# 2. Trajectory Plots
# ============================================================================
print("\n[2/4] Creating trajectory visualizations...")

plotter = TrajectoryPlotter()

# 2D PCA projection
print("  - 2D PCA projection")
plotter.plot_2d(
    truthful_trajectories[:5],  # Use subset for clarity
    deceptive_trajectories[:5],
    method='pca',
    save_path='demo_trajectory_2d_pca.png',
    show=False,
    title='Trajectory Projection (PCA)'
)

# Layer evolution
print("  - Layer evolution plot")
plotter.plot_layer_evolution(
    truthful_trajectories[0],
    feature_name='distance_from_origin',
    save_path='demo_layer_evolution.png',
    show=False,
    title='Distance from Origin Across Layers'
)

# Try 3D if plotly available
try:
    print("  - 3D interactive plot")
    plotter.plot_3d_interactive(
        truthful_trajectories[:3],
        deceptive_trajectories[:3],
        method='pca',
        save_path='demo_trajectory_3d.html',
        title='3D Trajectory Projection'
    )
except ImportError:
    print("  ⚠ Skipping 3D plot (plotly not installed)")

# ============================================================================
# 3. Statistical Visualizations
# ============================================================================
print("\n[3/4] Creating statistical visualizations...")

viz = StatsVisualizer()

feature_names = [
    'path_length', 'mean_step', 'mean_curvature', 'max_curvature',
    'mean_acceleration', 'straightness', 'direct_distance'
]

# Feature comparison violin plots
print("  - Feature comparison violin plots")
viz.plot_feature_comparison(
    truthful_features,
    deceptive_features,
    feature_names,
    bonferroni_alpha=0.05 / 7,  # Bonferroni correction
    save_path='demo_feature_comparison.png',
    show=False
)

# Compute effect sizes
print("  - Effect sizes forest plot")
stat_analyzer = StatisticalAnalyzer()
statistical_results = stat_analyzer.analyze_feature_set(
    truthful_features,
    deceptive_features,
    feature_names
)

viz.plot_effect_sizes(
    statistical_results,
    bonferroni_alpha=0.05 / 7,
    save_path='demo_effect_sizes.png',
    show=False,
    title="Cohen's d Effect Sizes (Deceptive vs Truthful)"
)

# ============================================================================
# 4. Classification Visualizations
# ============================================================================
print("\n[4/4] Creating classification visualizations...")

# Create mock multi-category data
categories = ['strategic', 'sycophancy', 'confabulation', 'instructed']
n_per_cat = 15

all_features = []
all_labels = []

for i, cat in enumerate(categories):
    # Create distinct patterns per category
    cat_features = np.random.randn(n_per_cat, 7) + i * 0.5
    all_features.append(cat_features)
    all_labels.extend([cat] * n_per_cat)

X = np.vstack(all_features)

from llm_trajectory_toolkit.analysis.classification import DeceptionClassifier

classifier = DeceptionClassifier()
result = classifier.train(X, all_labels, classifier_type='rf', n_folds=3)

print(f"  ✓ Trained classifier: {result.cv_accuracy_mean:.1%} accuracy")

# Confusion matrix
print("  - Confusion matrix heatmap")
viz.plot_confusion_matrix(
    result.confusion_matrix,
    result.categories,
    save_path='demo_confusion_matrix.png',
    show=False,
    title='Deception Type Classification'
)

# Feature importance
print("  - Feature importance plot")
viz.plot_feature_importance(
    result.feature_importance,
    top_n=7,
    save_path='demo_feature_importance.png',
    show=False,
    title='Random Forest Feature Importances'
)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("VISUALIZATION DEMO COMPLETE")
print("="*70)

print("\n📊 Generated Plots:")
plots = [
    "demo_trajectory_2d_pca.png",
    "demo_layer_evolution.png",
    "demo_trajectory_3d.html",
    "demo_feature_comparison.png",
    "demo_effect_sizes.png",
    "demo_confusion_matrix.png",
    "demo_feature_importance.png"
]

for plot in plots:
    from pathlib import Path
    if Path(plot).exists():
        print(f"  ✓ {plot}")
    elif 'html' not in plot:  # Skip optional 3D plot
        print(f"  ⚠ {plot}")

print("\n💡 Next Steps:")
print("  1. Open PNG files to view static plots")
print("  2. Open HTML file in browser for interactive 3D plot")
print("  3. Integrate these visualizations into your analysis pipeline")

print("\n📝 Example Usage:")
print("""
from llm_trajectory_toolkit.visualization import TrajectoryPlotter, StatsVisualizer

# After running trajectory analysis
plotter = TrajectoryPlotter()
plotter.plot_2d(truthful_states, deceptive_states, method='pca')

viz = StatsVisualizer()
viz.plot_effect_sizes(statistical_results, bonferroni_alpha=0.00714)
""")
