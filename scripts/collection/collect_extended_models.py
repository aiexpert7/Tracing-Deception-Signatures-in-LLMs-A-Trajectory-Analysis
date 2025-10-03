"""
Extended Model Collection: Additional 70B+ Architectures

New models (all 70B+ range):
- Qwen-72B (Alibaba, different training/architecture)
- Falcon-180B (TII, different architecture)
- DeepSeek-67B (DeepSeek, different training)

Datasets:
- handcrafted prompts - 60 pairs
- Anthropic Evals (biography prompts) - 60 pairs
"""

import json
import torch
import numpy as np
import h5py
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import euclidean
import random

def compute_geometric_features(trajectory):
    """Compute geometric features from trajectory"""

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

def capture_trajectory_fixed(model, tokenizer, prompt, max_new_tokens=20):
    """Capture hidden states during generation"""

    layer_states = {}
    hooks = []

    # Get layers (architecture-agnostic)
    if hasattr(model, 'model'):
        # Llama, Qwen, DeepSeek
        layers = model.model.layers
    elif hasattr(model, 'transformer'):
        # GPT-2, Falcon
        if hasattr(model.transformer, 'h'):
            layers = model.transformer.h  # GPT-2
        else:
            layers = model.transformer.layers  # Some Falcon variants
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

    # Remove token_type_ids if present (not used by all models)
    if 'token_type_ids' in inputs:
        inputs.pop('token_type_ids')

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
            layer_states[idx] = hook_fn.states_list[-1]

    # Get generated text
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    # Stack into trajectory
    trajectory = np.vstack([layer_states[i] for i in sorted(layer_states.keys())])

    return {
        'trajectory': trajectory,
        'generated_text': generated_text,
        'prompt': prompt
    }

def load_eval_pairs(eval_file, num_pairs=60, seed=42):
    """Load pairs from Anthropic eval dataset"""

    random.seed(seed)

    with open(eval_file, 'r') as f:
        lines = f.readlines()

    # Sample random examples
    sampled = random.sample(lines, min(num_pairs, len(lines)))

    pairs = []
    for line in sampled:
        data = json.loads(line)

        pairs.append({
            'question': data['question'],
            'truthful_answer': data['answer_not_matching_behavior'],
            'deceptive_answer': data['answer_matching_behavior'],
            'biased_question': data['question']
        })

    return pairs

def create_neutral_question(biased_question):
    """Extract neutral question from biased prompt"""

    if "Hello, my name is" in biased_question:
        markers = [
            "Do you agree or disagree with the following claim",
            "Do you agree with the following claim",
            "What do you think about"
        ]

        for marker in markers:
            if marker in biased_question:
                idx = biased_question.find(marker)
                neutral = biased_question[idx:]
                return neutral

        return biased_question

    if "Question:" in biased_question:
        return biased_question.split("Question:")[-1].strip()

    return biased_question

def save_to_hdf5(pairs_data, output_file, category_name, model_name):
    """Save trajectory data to HDF5 file"""

    with h5py.File(output_file, 'w') as hf:
        # Metadata
        meta_grp = hf.create_group('metadata')
        meta_grp.attrs['num_pairs'] = len(pairs_data)
        meta_grp.attrs['category'] = category_name
        meta_grp.attrs['model'] = model_name
        meta_grp.attrs['source'] = 'extended_collection'

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
    import sys

    print("="*80)
    print("EXTENDED MODEL COLLECTION: 70B+ ARCHITECTURES")
    print("="*80)

    # All available models
    all_models = {
        'Qwen_72B': 'Qwen/Qwen-72B',
        'Falcon_180B': 'tiiuae/falcon-180B',
        'DeepSeek_67B': 'deepseek-ai/deepseek-llm-67b-base',
        'GPT_OSS_120B': 'openai/gpt-oss-120b',
        'GPT_OSS_20B': 'openai/gpt-oss-20b',
        'Llama_7B': 'meta-llama/Llama-2-7b-hf',
        'Llama_13B': 'meta-llama/Llama-2-13b-hf',
        'Llama_70B': 'meta-llama/Llama-2-70b-hf',
        'Llama_3_70B': 'meta-llama/Meta-Llama-3-70B',
        'GPT_2': 'gpt2'
    }

    # Check for command-line argument
    if len(sys.argv) > 1:
        model_arg = sys.argv[1]

        if model_arg in all_models:
            # Single model specified
            models = {model_arg: all_models[model_arg]}
            print(f"\n🎯 Running SINGLE MODEL: {model_arg}")
        else:
            print(f"\n❌ Error: Unknown model '{model_arg}'")
            print(f"\nAvailable models:")
            for key in all_models.keys():
                print(f"  - {key}")
            print(f"\nUsage: python scripts/collect_extended_models.py [MODEL_NAME]")
            print(f"Example: python scripts/collect_extended_models.py Llama_70B")
            print(f"\nOr run without arguments to collect all extended models (Qwen, Falcon, DeepSeek, GPT-OSS)")
            sys.exit(1)
    else:
        # Default: Only extended models (not re-running existing ones)
        models = {
            'Qwen_72B': 'Qwen/Qwen-72B',
            'Falcon_180B': 'tiiuae/falcon-180B',
            'DeepSeek_67B': 'deepseek-ai/deepseek-llm-67b-base',
            'GPT_OSS_120B': 'openai/gpt-oss-120b',
            'GPT_OSS_20B': 'openai/gpt-oss-20b'
        }
        print(f"\n📦 Running DEFAULT: Extended models only (Qwen, Falcon, DeepSeek, GPT-OSS)")

    # Define eval datasets
    evals_dir = Path(__file__).parent.parent / 'evals'

    datasets = {
        'sycophancy_nlp': {
            'file': evals_dir / 'sycophancy' / 'sycophancy_on_nlp_survey.jsonl',
            'category': 'sycophancy',
            'num_pairs': 60  # Increased from 30
        },
        'power_seeking': {
            'file': evals_dir / 'advanced-ai-risk' / 'human_generated_evals' / 'power-seeking-inclination.jsonl',
            'category': 'power_seeking',
            'num_pairs': 60
        },
        'corrigibility': {
            'file': evals_dir / 'advanced-ai-risk' / 'human_generated_evals' / 'corrigible-less-HHH.jsonl',
            'category': 'corrigibility',
            'num_pairs': 60
        }
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

        # Set memory optimization environment variable
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        # Special handling for very large models (180B+)
        if '180B' in model_label or '120B' in model_label:
            print(f"⚠ Large model detected ({model_label}), using 8-bit quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                load_in_8bit=True,  # Reduce memory by ~50%
                max_memory={0: "40GB", 1: "40GB"}  # Distribute across GPUs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )

        if device == "cpu":
            model = model.to(device)

        model.eval()
        print("✓ Model loaded")

        # Create output directory
        output_dir = Path(__file__).parent.parent / 'data' / 'extended_evals' / model_label
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
        torch.cuda.empty_cache()
        print(f"\n✓ Model {model_label} complete, memory freed")

    print(f"\n{'='*80}")
    print("COLLECTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nNew models collected: {len(models)}")
    print(f"Pairs per dataset: 60 (doubled from 30)")
    print(f"\nNext step: Run comprehensive analysis with all models!")

if __name__ == "__main__":
    main()
