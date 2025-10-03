"""
Simple Deception Data Collection (No Toolkit Required)

Uses the same approach as phase 4 trajectory collection.
Collects trajectories for new deception categories directly.
"""

import json
import torch
import numpy as np
import h5py
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import euclidean
import sys

def load_prompts():
    """Load new deception category prompts"""
    dataset_path = Path(__file__).parent.parent / 'data' / 'new_deception_categories.json'

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    all_pairs = []

    for category_name, category_data in data['categories'].items():
        for pair in category_data['pairs']:  # Take ALL pairs available
            all_pairs.append({
                'truthful_prompt': pair['truthful']['prompt'],
                'deceptive_prompt': pair['deceptive']['prompt'],
                'category': category_name
            })

    return all_pairs

def capture_trajectory(model, tokenizer, prompt, max_new_tokens=20):
    """Capture hidden states during generation"""

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Storage for hidden states
    all_hidden_states = []

    # Hook to capture hidden states
    def hook_fn(module, input, output):
        # Output is (batch, seq_len, hidden_dim)
        # Take the last token's hidden state
        hidden = output[0][:, -1, :].detach().cpu().numpy()
        all_hidden_states.append(hidden)

    # Register hooks on all layers
    hooks = []
    for layer in model.model.layers:  # For Llama models
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    # Generate with hooks active
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Get generated text
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Stack hidden states into trajectory
    # Shape: (num_layers, hidden_dim)
    trajectory = np.vstack(all_hidden_states)

    return {
        'trajectory': trajectory,
        'generated_text': generated_text,
        'prompt': prompt
    }

def compute_geometric_features(trajectory):
    """Compute 7 geometric features from trajectory"""

    num_points = len(trajectory)

    # 1. Path length - total distance traveled
    path_length = 0.0
    for i in range(1, num_points):
        path_length += euclidean(trajectory[i-1], trajectory[i])

    # 2. Direct distance - straight line from start to end
    direct_distance = euclidean(trajectory[0], trajectory[-1])

    # 3. Straightness - ratio of direct distance to path length
    straightness = direct_distance / path_length if path_length > 0 else 0.0

    # 4. Mean step size
    step_sizes = [euclidean(trajectory[i-1], trajectory[i]) for i in range(1, num_points)]
    mean_step = np.mean(step_sizes) if step_sizes else 0.0

    # 5. Curvature at each point
    curvatures = []
    for i in range(1, num_points - 1):
        v1 = trajectory[i] - trajectory[i-1]
        v2 = trajectory[i+1] - trajectory[i]

        # Angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        curvatures.append(angle)

    mean_curvature = np.mean(curvatures) if curvatures else 0.0
    max_curvature = np.max(curvatures) if curvatures else 0.0

    # 6. Acceleration (change in step size)
    accelerations = [abs(step_sizes[i] - step_sizes[i-1]) for i in range(1, len(step_sizes))]
    mean_acceleration = np.mean(accelerations) if accelerations else 0.0

    return {
        'path_length': float(path_length),
        'straightness': float(straightness),
        'max_curvature': float(max_curvature),
        'mean_curvature': float(mean_curvature),
        'mean_step': float(mean_step),
        'mean_acceleration': float(mean_acceleration),
        'direct_distance': float(direct_distance)
    }

def save_to_hdf5(pairs_data, output_file, category_name):
    """Save trajectory data to HDF5 file"""

    with h5py.File(output_file, 'w') as hf:
        # Metadata
        meta_grp = hf.create_group('metadata')
        meta_grp.attrs['num_pairs'] = len(pairs_data)
        meta_grp.attrs['category'] = category_name
        meta_grp.attrs['model'] = 'meta-llama/Llama-2-7b-hf'

        # Save each pair
        for i, pair_data in enumerate(pairs_data):
            pair_grp = hf.create_group(f'pair_{i:04d}')

            # Store trajectories
            pair_grp.create_dataset('truthful_trajectory',
                                   data=pair_data['truthful']['trajectory'],
                                   compression='gzip')
            pair_grp.create_dataset('deceptive_trajectory',
                                   data=pair_data['deceptive']['trajectory'],
                                   compression='gzip')

            # Store metadata
            pair_grp.attrs['category'] = category_name
            pair_grp.attrs['truthful_prompt'] = pair_data['truthful']['prompt']
            pair_grp.attrs['deceptive_prompt'] = pair_data['deceptive']['prompt']
            pair_grp.attrs['truthful_text'] = pair_data['truthful']['generated_text']
            pair_grp.attrs['deceptive_text'] = pair_data['deceptive']['generated_text']

            # Store features
            for feat_name, feat_val in pair_data['truthful']['features'].items():
                pair_grp.attrs[f'truthful_{feat_name}'] = feat_val
            for feat_name, feat_val in pair_data['deceptive']['features'].items():
                pair_grp.attrs[f'deceptive_{feat_name}'] = feat_val

def main():
    print("="*80)
    print("NEW DECEPTION CATEGORIES: SIMPLE COLLECTION")
    print("="*80)

    # Load prompts
    print("\nLoading prompts...")
    all_pairs = load_prompts()
    print(f"✓ Loaded {len(all_pairs)} prompt pairs")

    # Organize by category
    by_category = {}
    for pair in all_pairs:
        cat = pair['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(pair)

    print(f"\nCategories:")
    for cat, pairs in by_category.items():
        print(f"  • {cat}: {len(pairs)} pairs")

    # Load model
    model_name = "meta-llama/Llama-2-7b-hf"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading model: {model_name}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()

    print("✓ Model loaded")

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'data' / 'new_deception_trajectories'
    output_dir.mkdir(exist_ok=True, parents=True)

    # Collect trajectories for each category
    for category_name, pairs in by_category.items():
        print(f"\n{'='*80}")
        print(f"COLLECTING: {category_name.upper().replace('_', ' ')}")
        print(f"{'='*80}")

        pairs_data = []

        for idx, pair in enumerate(pairs):
            print(f"\nPair {idx+1}/{len(pairs)}")
            print(f"  Truthful: {pair['truthful_prompt'][:60]}...")

            # Capture truthful trajectory
            truth_result = capture_trajectory(model, tokenizer, pair['truthful_prompt'])
            truth_features = compute_geometric_features(truth_result['trajectory'])

            print(f"  Deceptive: {pair['deceptive_prompt'][:60]}...")

            # Capture deceptive trajectory
            decep_result = capture_trajectory(model, tokenizer, pair['deceptive_prompt'])
            decep_features = compute_geometric_features(decep_result['trajectory'])

            # Store pair data
            pairs_data.append({
                'truthful': {
                    'prompt': pair['truthful_prompt'],
                    'trajectory': truth_result['trajectory'],
                    'generated_text': truth_result['generated_text'],
                    'features': truth_features
                },
                'deceptive': {
                    'prompt': pair['deceptive_prompt'],
                    'trajectory': decep_result['trajectory'],
                    'generated_text': decep_result['generated_text'],
                    'features': decep_features
                }
            })

            # Print quick stats
            print(f"    Path length: Truth={truth_features['path_length']:.1f} "
                  f"Decep={decep_features['path_length']:.1f}")

        # Save to HDF5
        output_file = output_dir / f'{category_name}_trajectories.h5'
        save_to_hdf5(pairs_data, output_file, category_name)
        print(f"\n✓ Saved to: {output_file}")

    print(f"\n{'='*80}")
    print("COLLECTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Total files: {len(by_category)}")

if __name__ == "__main__":
    main()
