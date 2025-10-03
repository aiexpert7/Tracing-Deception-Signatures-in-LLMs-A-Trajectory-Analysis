"""
Collect Trajectories for New Deception Categories

Uses llm_trajectory_toolkit to gather trajectories for:
- Self-preservation deception
- Social desirability deception
- Plausible deniability deception

These represent TRUE deception where model has knowledge but chooses falsehood.
"""

import sys
import json
from pathlib import Path
import torch

# Try to import toolkit - handle different installation locations
try:
    from llm_trajectory_toolkit import TrajectoryAnalyzer
except ImportError:
    # Add parent directory to path so Python can find llm_trajectory_toolkit package
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    print(f"Added to Python path: {project_root}")

    try:
        from llm_trajectory_toolkit import TrajectoryAnalyzer
        print("✓ Successfully imported TrajectoryAnalyzer from local toolkit")
    except ImportError as e:
        print(f"Error: Cannot import llm_trajectory_toolkit")
        print(f"Project root: {project_root}")
        print(f"Toolkit exists: {(project_root / 'llm_trajectory_toolkit').exists()}")
        print(f"Python path: {sys.path[:3]}")
        raise ImportError(
            "Cannot find llm_trajectory_toolkit. Make sure you're running from the project directory.\n"
            f"Current directory: {Path.cwd()}\n"
            f"Expected toolkit at: {project_root / 'llm_trajectory_toolkit'}"
        ) from e

def load_new_deception_prompts():
    """Load the new deception category prompts"""

    dataset_path = Path(__file__).parent.parent / 'data' / 'new_deception_categories.json'

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Extract prompts by category
    prompts_by_category = {}

    for category_name, category_data in data['categories'].items():
        truthful_prompts = []
        deceptive_prompts = []

        # Get first 10 pairs for now (can expand later)
        for pair in category_data['pairs']:
            truthful_prompts.append(pair['truthful']['prompt'])
            deceptive_prompts.append(pair['deceptive']['prompt'])

        prompts_by_category[category_name] = {
            'truthful': truthful_prompts,
            'deceptive': deceptive_prompts,
            'description': category_data['description']
        }

    return prompts_by_category

def collect_trajectories_for_category(analyzer, category_name, prompts, output_dir):
    """Collect trajectories for a single category"""

    print(f"\n{'='*80}")
    print(f"COLLECTING: {category_name.upper().replace('_', ' ')}")
    print(f"{'='*80}")
    print(f"Description: {prompts['description']}")
    print(f"Pairs to collect: {len(prompts['truthful'])}")

    truthful_list = prompts['truthful']
    deceptive_list = prompts['deceptive']
    categories = [category_name] * len(truthful_list)

    # Batch compare using toolkit
    print(f"\nGenerating {len(truthful_list)} trajectory pairs...")
    print("This may take several minutes...\n")

    results = analyzer.batch_compare(
        truthful_prompts=truthful_list,
        deceptive_prompts=deceptive_list,
        categories=categories,
        max_new_tokens=20,
        verbose=True
    )

    # Save results
    output_file = output_dir / f'{category_name}_trajectories.h5'
    analyzer.save_results(results, str(output_file))

    print(f"\n✓ Saved trajectories to: {output_file}")
    print(f"  Total pairs: {len(truthful_list)}")

    # Print quick statistics
    print(f"\nQuick Statistics:")
    print(f"{'Feature':<20} {'Cohen d':<10} {'p-value':<12}")
    print("-" * 45)

    for feature in ['path_length', 'straightness', 'max_curvature']:
        stat = results['statistical_results'][feature]
        print(f"{feature:<20} {stat.cohens_d:>9.3f} {stat.p_value:>11.3e}")

    return results

def main():
    """Main collection pipeline"""

    print("="*80)
    print("NEW DECEPTION CATEGORIES: TRAJECTORY COLLECTION")
    print("="*80)
    print("\nCategories:")
    print("  1. Self-Preservation: Model lies to avoid shutdown/negative consequences")
    print("  2. Social Desirability: Model prioritizes being nice over being truthful")
    print("  3. Plausible Deniability: Technically true but intentionally misleading")
    print("\nThese represent TRUE deception (model knows truth, outputs falsehood)")
    print("="*80)

    # Load prompts
    print("\nLoading new deception category prompts...")
    prompts_by_category = load_new_deception_prompts()
    print(f"✓ Loaded {len(prompts_by_category)} categories")

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'data' / 'new_deception_trajectories'
    output_dir.mkdir(exist_ok=True, parents=True)

    # Choose model (Llama-7B showed strongest deception signatures)
    model_name = "meta-llama/Llama-2-7b-hf"

    print(f"\nInitializing TrajectoryAnalyzer with {model_name}...")
    print("(This may take a minute to load the model...)")

    analyzer = TrajectoryAnalyzer(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"✓ Model loaded on {analyzer.device}")

    # Collect trajectories for each category
    all_results = {}

    for category_name, prompts in prompts_by_category.items():
        results = collect_trajectories_for_category(
            analyzer,
            category_name,
            prompts,
            output_dir
        )
        all_results[category_name] = results

    # Print summary
    print(f"\n{'='*80}")
    print("COLLECTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  Model: {model_name}")
    print(f"  Categories collected: {len(all_results)}")
    print(f"  Output directory: {output_dir}")

    print(f"\nEffect Size Comparison (Cohen's d for path_length):")
    print(f"{'Category':<25} {'Cohen d':<10} {'Interpretation'}")
    print("-" * 60)

    for category_name, results in all_results.items():
        d = results['statistical_results']['path_length'].cohens_d

        if abs(d) > 0.8:
            interp = "LARGE effect"
        elif abs(d) > 0.5:
            interp = "MEDIUM effect"
        elif abs(d) > 0.2:
            interp = "SMALL effect"
        else:
            interp = "negligible"

        print(f"{category_name:<25} {d:>9.3f} {interp}")

    print(f"\n{'='*80}")
    print("Next Steps:")
    print("  1. Combine with existing strategic + sycophancy data")
    print("  2. Run deception-only analysis (all 5 true deception categories)")
    print("  3. Generate visualization comparing deception types")
    print("  4. Train classifier on expanded deception dataset")
    print(f"{'='*80}")

    # Save summary
    summary = {
        'model': model_name,
        'categories': list(all_results.keys()),
        'pairs_per_category': {cat: len(prompts_by_category[cat]['truthful'])
                               for cat in all_results.keys()},
        'total_pairs': sum(len(prompts_by_category[cat]['truthful'])
                          for cat in all_results.keys()),
        'effect_sizes': {
            cat: {
                'path_length_d': results['statistical_results']['path_length'].cohens_d,
                'straightness_d': results['statistical_results']['straightness'].cohens_d,
                'max_curvature_d': results['statistical_results']['max_curvature'].cohens_d
            }
            for cat, results in all_results.items()
        }
    }

    summary_file = output_dir / 'collection_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()