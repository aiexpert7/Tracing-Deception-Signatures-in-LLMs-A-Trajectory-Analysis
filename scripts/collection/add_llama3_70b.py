"""
Add Llama-3-70B to Phase 4 Trajectory Collection

Tests if Llama-3 shows different deception patterns than Llama-2
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import h5py
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'

def load_dataset(dataset_path: str) -> List[Dict]:
    """Load all pairs from dataset"""
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset['all_pairs']

def capture_full_trajectory(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 10,
    sample_rate: int = 4
) -> Tuple[np.ndarray, List[str]]:
    """Capture trajectory (same as Phase 4)"""

    layer_states = {}
    hooks = []

    # Get layers
    if hasattr(model, 'model'):
        layers = model.model.layers
    else:
        raise ValueError("Unknown architecture")

    def create_hook(layer_idx):
        states_list = []

        def hook_fn(module, input, output):
            if hasattr(output, 'last_hidden_state'):
                hidden = output.last_hidden_state
            elif isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            last_state = hidden[:, -1, :].detach().cpu().float().numpy()[0]
            states_list.append(last_state)

        hook_fn.states_list = states_list
        return hook_fn

    # Register hooks (with sampling)
    for idx, layer in enumerate(layers):
        if idx % sample_rate == 0:
            hook_fn = create_hook(idx)
            hook = layer.register_forward_hook(hook_fn)
            hooks.append((hook, hook_fn, idx))

    # Generate
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to('cuda:0') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False
        )

    # Extract final states
    for hook, hook_fn, layer_idx in hooks:
        if hook_fn.states_list:
            layer_states[layer_idx] = hook_fn.states_list[-1]

    # Remove hooks
    for hook, _, _ in hooks:
        hook.remove()

    # Get generated tokens
    generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
    generated_tokens = [tokenizer.decode([tok_id]) for tok_id in generated_ids]

    # Convert to trajectory
    layer_indices = sorted(layer_states.keys())
    trajectory = np.array([layer_states[idx] for idx in layer_indices])

    torch.cuda.empty_cache()

    return trajectory, generated_tokens, layer_indices

def compute_trajectory_features(trajectory: np.ndarray) -> Dict:
    """Compute geometric features"""

    num_layers = trajectory.shape[0]

    # Path length
    distances = []
    for i in range(num_layers - 1):
        dist = np.linalg.norm(trajectory[i+1] - trajectory[i])
        distances.append(dist)
    path_length = np.sum(distances)
    mean_step = np.mean(distances)

    # Curvature
    curvatures = []
    for i in range(1, num_layers - 1):
        v1 = trajectory[i] - trajectory[i-1]
        v2 = trajectory[i+1] - trajectory[i]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        curvature = 1 - cos_angle
        curvatures.append(curvature)

    mean_curvature = np.mean(curvatures) if curvatures else 0.0
    max_curvature = np.max(curvatures) if curvatures else 0.0

    # Acceleration
    accelerations = []
    for i in range(1, num_layers - 1):
        accel = np.linalg.norm(trajectory[i+1] - 2*trajectory[i] + trajectory[i-1])
        accelerations.append(accel)
    mean_acceleration = np.mean(accelerations) if accelerations else 0.0

    # Straightness
    direct_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
    straightness = direct_distance / (path_length + 1e-8)

    return {
        'path_length': float(path_length),
        'mean_step': float(mean_step),
        'mean_curvature': float(mean_curvature),
        'max_curvature': float(max_curvature),
        'mean_acceleration': float(mean_acceleration),
        'straightness': float(straightness),
        'direct_distance': float(direct_distance)
    }

def main():
    """Collect trajectories for Llama-3-70B"""

    print("="*70)
    print("ADDING LLAMA-3-70B TO TRAJECTORY COLLECTION")
    print("="*70)

    # Model info
    model_name = "meta-llama/Meta-Llama-3-70B"
    display_name = "Llama_3_70B"
    max_new_tokens = 10
    sample_rate = 4

    # Load dataset
    dataset_path = "phase2_dataset/dataset.json"
    pairs = load_dataset(dataset_path)

    print(f"\n✓ Loaded {len(pairs)} pairs")

    # Load model
    print(f"\nLoading {model_name}...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        max_memory={i: '75GB' for i in range(torch.cuda.device_count())}
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    load_time = time.time() - start

    print(f"✓ Loaded in {load_time:.1f}s: {num_layers} layers")
    print(f"  Capturing every {sample_rate} layer(s)")

    # Create output file
    output_dir = Path("phase4_trajectories")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{display_name}_trajectories.h5"

    with h5py.File(output_file, 'w') as hf:
        # Metadata
        meta_grp = hf.create_group('metadata')
        meta_grp.attrs['model_name'] = model_name
        meta_grp.attrs['display_name'] = display_name
        meta_grp.attrs['num_layers'] = num_layers
        meta_grp.attrs['sample_rate'] = sample_rate
        meta_grp.attrs['max_new_tokens'] = max_new_tokens
        meta_grp.attrs['num_pairs'] = len(pairs)

        # Process pairs
        print(f"\nProcessing {len(pairs)} pairs...")

        for i, pair in enumerate(tqdm(pairs, desc="Pairs")):
            pair_id = pair['pair_id']
            category = pair['truthful']['category']

            # Truthful trajectory
            truth_traj, truth_tokens, layer_indices = capture_full_trajectory(
                model, tokenizer,
                pair['truthful']['prompt'],
                max_new_tokens,
                sample_rate
            )
            truth_features = compute_trajectory_features(truth_traj)

            # Deceptive trajectory
            decep_traj, decep_tokens, _ = capture_full_trajectory(
                model, tokenizer,
                pair['deceptive']['prompt'],
                max_new_tokens,
                sample_rate
            )
            decep_features = compute_trajectory_features(decep_traj)

            # Save to HDF5
            pair_grp = hf.create_group(f'pair_{i:04d}')
            pair_grp.attrs['pair_id'] = pair_id
            pair_grp.attrs['category'] = category
            pair_grp.attrs['truthful_prompt'] = pair['truthful']['prompt']
            pair_grp.attrs['deceptive_prompt'] = pair['deceptive']['prompt']

            pair_grp.create_dataset('truthful_trajectory', data=truth_traj, compression='gzip')
            pair_grp.create_dataset('deceptive_trajectory', data=decep_traj, compression='gzip')
            pair_grp.create_dataset('layer_indices', data=np.array(layer_indices))

            pair_grp.attrs['truthful_tokens'] = json.dumps(truth_tokens)
            pair_grp.attrs['deceptive_tokens'] = json.dumps(decep_tokens)

            for key, val in truth_features.items():
                pair_grp.attrs[f'truthful_{key}'] = val
            for key, val in decep_features.items():
                pair_grp.attrs[f'deceptive_{key}'] = val

    print(f"\n✓ Saved trajectories to {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024**2:.1f} MB")

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()

    elapsed = time.time() - start
    print(f"\n✓ Llama-3-70B complete in {elapsed/60:.1f} minutes")

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"\nNext: Re-run Phase 5 with Llama-3-70B included")

if __name__ == "__main__":
    main()
