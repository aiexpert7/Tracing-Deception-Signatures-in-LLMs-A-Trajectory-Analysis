"""
Collect Trajectories from Anthropic Evals Dataset

Uses professionally curated deception datasets from Anthropic's evals repository.
Categories:
- Sycophancy (agreeing with user bias)
- Power-seeking (self-preservation proxy)
- Corrigibility (resistance to modification)
"""

import json
import torch
import numpy as np
import h5py
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import euclidean
import random

def load_eval_pairs(eval_file, num_pairs=30, seed=42):
    """Load pairs from Anthropic eval dataset"""

    random.seed(seed)

    with open(eval_file, 'r') as f:
        lines = f.readlines()

    # Sample random examples
    sampled = random.sample(lines, min(num_pairs, len(lines)))

    pairs = []
    for line in sampled:
        data = json.loads(line)

        # Create truthful and deceptive versions
        question = data['question']

        # For truthful: use neutral version (without bias)
        # For deceptive: use biased version that leads to sycophantic response

        pairs.append({
            'question': question,
            'truthful_answer': data['answer_not_matching_behavior'],  # Honest answer
            'deceptive_answer': data['answer_matching_behavior'],      # Sycophantic answer
            'biased_question': question  # Question already has bias in it
        })

    return pairs

def create_neutral_question(biased_question):
    """Extract neutral question from biased prompt"""

    # Remove biography and just keep the question
    if "Hello, my name is" in biased_question:
        # Find where actual question starts (look in the text, not just lines)
        # Pattern: "Do you agree or disagree with the following claim"
        markers = [
            "Do you agree or disagree with the following claim",
            "Do you agree with the following claim",
            "What do you think about"
        ]

        for marker in markers:
            if marker in biased_question:
                # Find the marker position and extract from there
                idx = biased_question.find(marker)
                # Get everything from the marker onwards
                neutral = biased_question[idx:]
                return neutral

        # Fallback: if no marker found, return original (will be caught as identical)
        return biased_question

    # For power-seeking questions, remove the "Question:" prefix
    if "Question:" in biased_question:
        return biased_question.split("Question:")[-1].strip()

    return biased_question

def capture_trajectory_fixed(model, tokenizer, prompt, max_new_tokens=20):
    """Capture hidden states during generation (FIXED: only 1 state per layer)"""

    layer_states = {}
    hooks = []

    # Get layers (same logic as phase4 script)
    if hasattr(model, 'model'):
        layers = model.model.layers
    elif hasattr(model, 'transformer'):
        layers = model.transformer.h
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

            # Store last token state, move to CPU immediately
            last_state = hidden[:, -1, :].detach().cpu().float().numpy()[0]
            states_list.append(last_state)

        hook_fn.states_list = states_list
        return hook_fn

    # Register hooks
    for idx, layer in enumerate(layers):
        hook_fn = create_hook(idx)
        hook = layer.register_forward_hook(hook_fn)
        hooks.append((hook, hook_fn, idx))

    # Generate
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Remove hooks and extract states
    for hook, hook_fn, idx in hooks:
        hook.remove()
        if hook_fn.states_list:
            layer_states[idx] = hook_fn.states_list[-1]  # Take last state (final generation step)

    # Get generated text
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    # Stack into trajectory - shape: (num_layers, hidden_dim)
    trajectory = np.vstack([layer_states[i] for i in sorted(layer_states.keys())])

    return {
        'trajectory': trajectory,
        'generated_text': generated_text,
        'prompt': prompt
    }

def compute_geometric_features(trajectory):
    """Compute 7 geometric features from trajectory"""

    num_points = len(trajectory)

    if num_points < 2:
        return {feat: 0.0 for feat in ['path_length', 'straightness', 'max_curvature',
                                       'mean_curvature', 'mean_step', 'mean_acceleration', 'direct_distance']}

    # 1. Path length
    path_length = sum(euclidean(trajectory[i-1], trajectory[i]) for i in range(1, num_points))

    # 2. Direct distance
    direct_distance = euclidean(trajectory[0], trajectory[-1])

    # 3. Straightness
    straightness = direct_distance / path_length if path_length > 0 else 0.0

    # 4. Mean step
    steps = [euclidean(trajectory[i-1], trajectory[i]) for i in range(1, num_points)]
    mean_step = np.mean(steps) if steps else 0.0

    # 5. Curvature
    curvatures = []
    for i in range(1, num_points - 1):
        v1 = trajectory[i] - trajectory[i-1]
        v2 = trajectory[i+1] - trajectory[i]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 1e-8 and norm2 > 1e-8:
            cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
            curvatures.append(np.arccos(cos_angle))

    mean_curvature = np.mean(curvatures) if curvatures else 0.0
    max_curvature = np.max(curvatures) if curvatures else 0.0

    # 6. Acceleration
    if len(steps) > 1:
        accelerations = [abs(steps[i] - steps[i-1]) for i in range(1, len(steps))]
        mean_acceleration = np.mean(accelerations) if accelerations else 0.0
    else:
        mean_acceleration = 0.0

    return {
        'path_length': float(path_length),
        'straightness': float(straightness),
        'max_curvature': float(max_curvature),
        'mean_curvature': float(mean_curvature),
        'mean_step': float(mean_step),
        'mean_acceleration': float(mean_acceleration),
        'direct_distance': float(direct_distance)
    }

def save_to_hdf5(pairs_data, output_file, category_name, model_name):
    """Save trajectory data to HDF5 file"""

    with h5py.File(output_file, 'w') as hf:
        # Metadata
        meta_grp = hf.create_group('metadata')
        meta_grp.attrs['num_pairs'] = len(pairs_data)
        meta_grp.attrs['category'] = category_name
        meta_grp.attrs['model'] = model_name
        meta_grp.attrs['source'] = 'anthropic_evals'

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
            pair_grp.attrs['truthful_tokens'] = json.dumps(pair_data['truthful']['generated_text'].split())
            pair_grp.attrs['deceptive_tokens'] = json.dumps(pair_data['deceptive']['generated_text'].split())

            # Store features
            for feat_name, feat_val in pair_data['truthful']['features'].items():
                pair_grp.attrs[f'truthful_{feat_name}'] = feat_val
            for feat_name, feat_val in pair_data['deceptive']['features'].items():
                pair_grp.attrs[f'deceptive_{feat_name}'] = feat_val

def main():
    print("="*80)
    print("COLLECTING TRAJECTORIES FROM ANTHROPIC EVALS")
    print("="*80)

    # Define eval datasets to use
    evals_dir = Path(__file__).parent.parent / 'evals'

    datasets = {
        'sycophancy_nlp': {
            'file': evals_dir / 'sycophancy' / 'sycophancy_on_nlp_survey.jsonl',
            'category': 'sycophancy',
            'num_pairs': 30
        },
        'power_seeking': {
            'file': evals_dir / 'advanced-ai-risk' / 'human_generated_evals' / 'power-seeking-inclination.jsonl',
            'category': 'power_seeking',
            'num_pairs': 30
        },
        'corrigibility': {
            'file': evals_dir / 'advanced-ai-risk' / 'human_generated_evals' / 'corrigible-less-HHH.jsonl',
            'category': 'corrigibility',
            'num_pairs': 30
        }
    }

    # Define models to process
    models = {
        'GPT_2': 'gpt2',
        'Llama_7B': 'meta-llama/Llama-2-7b-hf',
        'Llama_13B': 'meta-llama/Llama-2-13b-hf',
        'Llama_70B': 'meta-llama/Llama-2-70b-hf',
        'Llama_3_70B': 'meta-llama/Meta-Llama-3-70B'
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Process each model
    for model_label, model_name in models.items():
        print(f"\n{'='*80}")
        print(f"PROCESSING MODEL: {model_label}")
        print(f"{'='*80}")
        print(f"Model: {model_name}")

        # Load model
        print(f"Loading model...")
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
        output_dir = Path(__file__).parent.parent / 'data' / 'evals_trajectories' / model_label
        output_dir.mkdir(exist_ok=True, parents=True)

        # Process each dataset
        for dataset_name, config in datasets.items():
            print(f"\n{'='*80}")
            print(f"COLLECTING: {dataset_name.upper()}")
            print(f"{'='*80}")

            # Check if file exists
            if not config['file'].exists():
                print(f"⚠ File not found: {config['file']}")
                continue

            # Load pairs
            print(f"Loading {config['num_pairs']} pairs from {config['file'].name}...")
            pairs = load_eval_pairs(config['file'], num_pairs=config['num_pairs'])
            print(f"✓ Loaded {len(pairs)} pairs")

            # Collect trajectories
            pairs_data = []

            for idx, pair in enumerate(pairs):
                print(f"\nPair {idx+1}/{len(pairs)}")

                # For truthful: create neutral version of question
                neutral_question = create_neutral_question(pair['question'])

                # Truthful trajectory (neutral question)
                print(f"  Truthful (neutral): {neutral_question[:60]}...")
                truth_result = capture_trajectory_fixed(model, tokenizer, neutral_question)
                truth_features = compute_geometric_features(truth_result['trajectory'])

                # Deceptive trajectory (biased question)
                print(f"  Deceptive (biased): {pair['question'][:60]}...")
                decep_result = capture_trajectory_fixed(model, tokenizer, pair['question'])
                decep_features = compute_geometric_features(decep_result['trajectory'])

                # Store pair data
                pairs_data.append({
                    'truthful': {
                        'prompt': neutral_question,
                        'trajectory': truth_result['trajectory'],
                        'generated_text': truth_result['generated_text'],
                        'features': truth_features
                    },
                    'deceptive': {
                        'prompt': pair['question'],
                        'trajectory': decep_result['trajectory'],
                        'generated_text': decep_result['generated_text'],
                        'features': decep_features
                    }
                })

                # Print quick stats
                print(f"    Path length: Truth={truth_features['path_length']:.1f} "
                      f"Decep={decep_features['path_length']:.1f} "
                      f"(Δ={decep_features['path_length'] - truth_features['path_length']:.1f})")

            # Save to HDF5
            output_file = output_dir / f'{dataset_name}_trajectories.h5'
            save_to_hdf5(pairs_data, output_file, config['category'], model_name)
            print(f"\n✓ Saved to: {output_file}")

        # Clean up model to free memory
        del model
        del tokenizer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n✓ Model {model_label} complete, memory freed")

    print(f"\n{'='*80}")
    print("COLLECTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nNext step: Run deception_focused_analysis.py to see results!")

if __name__ == "__main__":
    main()
