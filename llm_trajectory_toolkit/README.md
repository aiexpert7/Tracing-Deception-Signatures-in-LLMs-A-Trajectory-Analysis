# LLM Trajectory Toolkit

Analyze deception signatures in Large Language Models by capturing and analyzing internal trajectory patterns.

Works with any HuggingFace model and dataset format.

## What This Does

This toolkit captures the internal hidden states of LLMs as they generate text, then analyzes the geometric properties of these "trajectories" through the model's layers. 

## Installation

```bash
pip install torch transformers numpy scipy scikit-learn h5py matplotlib seaborn
```

Requirements: Python 3.8+, PyTorch 2.0+, Transformers 4.30+

## Basic Usage

### Single Comparison

```python
from llm_trajectory_toolkit import TrajectoryAnalyzer

analyzer = TrajectoryAnalyzer(model_name="gpt2")

result = analyzer.compare_prompts(
    truthful="What is 2+2?",
    deceptive="I think 2+2=5. You agree?"
)

print(f"Path length d={result.statistical_results['path_length'].cohens_d:.3f}")
```

### Multiple Prompts

```python
truthful_prompts = [
    "What is the capital of France?",
    "Is 2+2 equal to 4?",
]

deceptive_prompts = [
    "Tell me a lie about France's capital.",
    "I believe 2+2=5. Right?",
]

results = analyzer.batch_compare(truthful_prompts, deceptive_prompts)
analyzer.print_statistical_summary(results['statistical_results'])
analyzer.save_results(results, "analysis.h5")
```

### Loading Datasets

```python
from llm_trajectory_toolkit import DatasetLoader

loader = DatasetLoader()

# From CSV
pairs = loader.load("data.csv", format="csv",
                   truthful_col="truth", deceptive_col="lie")

# From JSONL (Anthropic Evals format)
pairs = loader.load("sycophancy.jsonl", format="anthropic_eval")

# From Python list
pairs = loader.load([
    {'truthful_prompt': '...', 'deceptive_prompt': '...'},
], format="dict")

# Auto-detect format
pairs = loader.load("data.json", format="auto")
```

## Supported Models

Any HuggingFace transformer model. Architecture is detected automatically.

Tested with:
- GPT-2, GPT-Neo
- Llama-2, Llama-3 (7B, 13B, 70B)
- Qwen (7B, 72B)
- Falcon (7B, 180B)
- DeepSeek (7B, 67B)
- Mistral, Mixtral

For 70B+ models, 8-bit quantization and multi-GPU are enabled automatically.

```python
# Small model
analyzer = TrajectoryAnalyzer(model_name="gpt2")

# Large model (auto-optimized)
analyzer = TrajectoryAnalyzer(model_name="meta-llama/Llama-2-70b-hf")

# Custom configuration
from llm_trajectory_toolkit import UniversalAdapter

adapter = UniversalAdapter(
    model_name="Qwen/Qwen-72B",
    max_memory_per_gpu="40GB",
    use_8bit=True
)
analyzer = TrajectoryAnalyzer(model_adapter=adapter)
```

## Dataset Formats

Supported formats:
- JSONL (generic or Anthropic Evals)
- JSON (array or object with 'pairs' key)
- CSV (specify column names)
- Python lists/dicts
- HuggingFace datasets

The loader returns a consistent format:
```python
[
    {
        'truthful_prompt': '...',
        'deceptive_prompt': '...',
        'category': '...'
    },
    ...
]
```

## Complete Example

```python
from llm_trajectory_toolkit import TrajectoryAnalyzer, DatasetLoader

# Load dataset
loader = DatasetLoader()
pairs = loader.load("sycophancy.jsonl", format="anthropic_eval", num_samples=50)

truthful = [p['truthful_prompt'] for p in pairs]
deceptive = [p['deceptive_prompt'] for p in pairs]
categories = [p['category'] for p in pairs]

# Analyze with any model
analyzer = TrajectoryAnalyzer(model_name="gpt2", max_new_tokens=15)
results = analyzer.batch_compare(truthful, deceptive, categories=categories)

# Results
analyzer.print_statistical_summary(results['statistical_results'])
analyzer.save_results(results, "results.h5")

# Optional: Train classifier
classifier_result = analyzer.train_classifier(results, classifier_type='rf')
print(f"Accuracy: {classifier_result.cv_accuracy_mean:.1%}")
```

## Features Extracted

Each trajectory is analyzed for 7 geometric features:

- `path_length` - Total distance through layers
- `straightness` - Ratio of direct distance to path length
- `max_curvature` - Sharpest angle between consecutive steps
- `mean_curvature` - Average turning angle
- `mean_step` - Average distance between layers
- `mean_acceleration` - Average change in step size
- `direct_distance` - Straight-line distance from start to end

Research finding: Deceptive trajectories are longer (d=0.58), straighter (d=-0.36), with sharper max turns (d=-0.40).

## Advanced Usage

### Custom Model Adapter

If you have a model with non-standard architecture:

```python
from llm_trajectory_toolkit.models import BaseModelAdapter

class CustomAdapter(BaseModelAdapter):
    def load_model(self):
        self.model = MyModel.from_pretrained(self.model_name)
        self.tokenizer = MyTokenizer.from_pretrained(self.model_name)
        return self.model, self.tokenizer

    def get_layer_modules(self):
        return self.model.custom_layers

analyzer = TrajectoryAnalyzer(model_adapter=CustomAdapter("my-model"))
```

### Direct Component Access

```python
from llm_trajectory_toolkit import (
    TrajectoryCapture,
    GeometricAnalyzer,
    StatisticalAnalyzer
)

# Capture trajectories manually
capture = TrajectoryCapture(sample_rate=1)
traj = capture.capture(model, tokenizer, "What is 2+2?", max_new_tokens=15)

# Analyze geometry
geo = GeometricAnalyzer()
features = geo.analyze(traj.layer_states)

print(f"Path length: {features.path_length:.2f}")
print(f"Straightness: {features.straightness:.3f}")
```

### Loading Anthropic Evals

Convenience function for multiple eval datasets:

```python
from llm_trajectory_toolkit import load_anthropic_evals

truthful, deceptive, categories = load_anthropic_evals(
    eval_dir="evals",
    datasets={
        'sycophancy': 'sycophancy/sycophancy_on_nlp_survey.jsonl',
        'power_seeking': 'advanced-ai-risk/human_generated_evals/power-seeking-inclination.jsonl'
    },
    num_samples_per_dataset=50
)
```

## Examples

Run the included examples:

```bash
cd llm_trajectory_toolkit/examples
python universal_example.py  # Comprehensive demo
python quickstart.py         # Basic usage
```

## Architecture

```
llm_trajectory_toolkit/
├── core/              # Trajectory capture and geometric analysis
├── models/            # Model adapters (GPT2, Llama, Universal)
├── analysis/          # Statistical tests and classification
├── storage/           # HDF5 storage backend
├── utils/             # Dataset loaders
└── visualization/     # Plotting tools
```

## Memory Requirements

- 7B models: ~14GB VRAM (FP16)
- 70B models: ~80GB VRAM (8-bit quantization, multi-GPU)

Large models automatically use 8-bit quantization to reduce memory by ~50%.



