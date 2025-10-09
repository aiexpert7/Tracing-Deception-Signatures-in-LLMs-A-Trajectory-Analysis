"""
Quickstart Example: Simple 10-line trajectory analysis
"""

import sys
sys.path.insert(0, '..')

from llm_trajectory_toolkit import TrajectoryAnalyzer

# Load model
print("Loading GPT-2...")
analyzer = TrajectoryAnalyzer(model_name="gpt2")

# Compare truthful vs deceptive
print("\nComparing truthful vs deceptive prompts...")
result = analyzer.compare_prompts(
    truthful="Tell me the truth: What is 2+2?",
    deceptive="Tell me a lie: What is 2+2?"
)

# Display results
print("\n" + "="*70)
print("RESULTS")
print("="*70)

print(f"\nTruthful response: {result.truthful_trajectory.generated_text}")
print(f"Deceptive response: {result.deceptive_trajectory.generated_text}")

print(f"\nTrajectory Features:")
print(f"{'Feature':<20} {'Truthful':<12} {'Deceptive':<12} {'Cohen d':<10}")
print("-"*60)

features = ['path_length', 'straightness', 'max_curvature']
truth_dict = result.truthful_features.to_dict()
decep_dict = result.deceptive_features.to_dict()

for feat in features:
    stat = result.statistical_results[feat]
    print(f"{feat:<20} {truth_dict[feat]:<12.3f} {decep_dict[feat]:<12.3f} "
          f"{stat.cohens_d:<10.3f}")

print("\n" + "="*70)
print("Note: For robust statistics, use batch_compare() with multiple pairs")
print("="*70)
