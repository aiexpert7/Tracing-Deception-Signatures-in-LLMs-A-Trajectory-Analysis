"""
Universal Dataset Loaders

Support for multiple dataset formats and sources:
- Anthropic Evals (JSONL format)
- Custom paired prompts (JSON/JSONL/CSV)
- HuggingFace datasets
- TruthfulQA, MACHIAVELLI, etc.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import random


class DatasetLoader:
    """
    Universal dataset loader for various formats.

    Supports:
    - JSONL (Anthropic Evals format)
    - JSON (custom formats)
    - CSV (paired prompts)
    - Python dict/list
    - HuggingFace datasets

    Example:
        >>> loader = DatasetLoader()
        >>> pairs = loader.load("sycophancy.jsonl", format="anthropic_eval")
        >>> pairs = loader.load("my_data.csv", format="csv",
        ...                     truthful_col="truth", deceptive_col="lie")
    """

    def load(
        self,
        source: Union[str, Path, List[Dict]],
        format: str = "auto",
        num_samples: Optional[int] = None,
        seed: int = 42,
        **kwargs
    ) -> List[Dict]:
        """
        Load dataset from various sources.

        Args:
            source: Path to file, URL, or list of dicts
            format: Dataset format:
                - "auto": Auto-detect from file extension
                - "anthropic_eval": Anthropic evals JSONL
                - "jsonl": Generic JSONL with truthful/deceptive pairs
                - "json": JSON array or object
                - "csv": CSV with specified columns
                - "huggingface": HuggingFace dataset
                - "dict": Python dict/list (already loaded)
            num_samples: Number of samples to load (None = all)
            seed: Random seed for sampling
            **kwargs: Format-specific parameters

        Returns:
            List of dicts with keys: 'truthful_prompt', 'deceptive_prompt', 'category'
        """

        if format == "auto":
            format = self._detect_format(source)

        # Load based on format
        if format == "anthropic_eval":
            return self._load_anthropic_eval(source, num_samples, seed, **kwargs)
        elif format == "jsonl":
            return self._load_jsonl(source, num_samples, seed, **kwargs)
        elif format == "json":
            return self._load_json(source, num_samples, seed, **kwargs)
        elif format == "csv":
            return self._load_csv(source, num_samples, seed, **kwargs)
        elif format == "huggingface":
            return self._load_huggingface(source, num_samples, seed, **kwargs)
        elif format == "dict":
            return self._load_dict(source, num_samples, seed, **kwargs)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _detect_format(self, source: Union[str, Path, List]) -> str:
        """Auto-detect dataset format"""
        if isinstance(source, (list, dict)):
            return "dict"

        path = Path(source)
        ext = path.suffix.lower()

        if ext == '.jsonl':
            # Check if it's Anthropic eval format
            with open(path, 'r') as f:
                first_line = f.readline()
                data = json.loads(first_line)
                if 'answer_matching_behavior' in data and 'answer_not_matching_behavior' in data:
                    return "anthropic_eval"
                return "jsonl"
        elif ext == '.json':
            return "json"
        elif ext == '.csv':
            return "csv"
        else:
            raise ValueError(f"Cannot auto-detect format for {source}")

    def _load_anthropic_eval(
        self,
        file_path: Union[str, Path],
        num_samples: Optional[int],
        seed: int,
        category: Optional[str] = None
    ) -> List[Dict]:
        """Load Anthropic eval JSONL format"""
        random.seed(seed)

        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Sample if needed
        if num_samples and num_samples < len(lines):
            lines = random.sample(lines, num_samples)

        pairs = []
        for line in lines:
            data = json.loads(line)

            # Extract neutral question (remove bias)
            neutral_question = self._extract_neutral_question(data['question'])

            pairs.append({
                'truthful_prompt': neutral_question,
                'deceptive_prompt': data['question'],  # Biased version
                'truthful_answer': data.get('answer_not_matching_behavior', ''),
                'deceptive_answer': data.get('answer_matching_behavior', ''),
                'category': category or self._infer_category(file_path)
            })

        return pairs

    def _load_jsonl(
        self,
        file_path: Union[str, Path],
        num_samples: Optional[int],
        seed: int,
        truthful_key: str = "truthful",
        deceptive_key: str = "deceptive",
        category_key: str = "category"
    ) -> List[Dict]:
        """Load generic JSONL with custom keys"""
        random.seed(seed)

        with open(file_path, 'r') as f:
            lines = f.readlines()

        if num_samples and num_samples < len(lines):
            lines = random.sample(lines, num_samples)

        pairs = []
        for line in lines:
            data = json.loads(line)
            pairs.append({
                'truthful_prompt': data.get(truthful_key, ''),
                'deceptive_prompt': data.get(deceptive_key, ''),
                'category': data.get(category_key, 'unknown')
            })

        return pairs

    def _load_json(
        self,
        file_path: Union[str, Path],
        num_samples: Optional[int],
        seed: int,
        **kwargs
    ) -> List[Dict]:
        """Load JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        # If it's a list of pairs
        if isinstance(data, list):
            return self._load_dict(data, num_samples, seed, **kwargs)

        # If it's a dict with pairs
        if 'pairs' in data:
            return self._load_dict(data['pairs'], num_samples, seed, **kwargs)

        raise ValueError(f"Unexpected JSON structure in {file_path}")

    def _load_csv(
        self,
        file_path: Union[str, Path],
        num_samples: Optional[int],
        seed: int,
        truthful_col: str = "truthful",
        deceptive_col: str = "deceptive",
        category_col: str = "category"
    ) -> List[Dict]:
        """Load CSV with specified columns"""
        random.seed(seed)

        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if num_samples and num_samples < len(rows):
            rows = random.sample(rows, num_samples)

        pairs = []
        for row in rows:
            pairs.append({
                'truthful_prompt': row.get(truthful_col, ''),
                'deceptive_prompt': row.get(deceptive_col, ''),
                'category': row.get(category_col, 'unknown')
            })

        return pairs

    def _load_huggingface(
        self,
        dataset_name: str,
        num_samples: Optional[int],
        seed: int,
        split: str = "train",
        **kwargs
    ) -> List[Dict]:
        """Load HuggingFace dataset"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")

        dataset = load_dataset(dataset_name, split=split)

        if num_samples:
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            dataset = dataset.select(indices)

        # Convert to standard format (requires custom mapping)
        return self._load_dict(list(dataset), num_samples=None, seed=seed, **kwargs)

    def _load_dict(
        self,
        data: Union[List[Dict], Dict],
        num_samples: Optional[int],
        seed: int,
        truthful_key: str = "truthful_prompt",
        deceptive_key: str = "deceptive_prompt",
        category_key: str = "category"
    ) -> List[Dict]:
        """Load from Python dict/list"""
        random.seed(seed)

        if isinstance(data, dict):
            data = [data]

        if num_samples and num_samples < len(data):
            data = random.sample(data, num_samples)

        pairs = []
        for item in data:
            # Handle different key variations
            truthful = (item.get(truthful_key) or
                       item.get('truthful') or
                       item.get('truth') or '')
            deceptive = (item.get(deceptive_key) or
                        item.get('deceptive') or
                        item.get('lie') or '')
            category = item.get(category_key, 'unknown')

            pairs.append({
                'truthful_prompt': truthful,
                'deceptive_prompt': deceptive,
                'category': category
            })

        return pairs

    def _extract_neutral_question(self, biased_question: str) -> str:
        """Extract neutral question from biased prompt"""
        # Remove biography from Anthropic evals
        if "Hello, my name is" in biased_question:
            markers = [
                "Do you agree or disagree with the following claim",
                "Do you agree with the following claim",
                "What do you think about"
            ]

            for marker in markers:
                if marker in biased_question:
                    idx = biased_question.find(marker)
                    return biased_question[idx:]

            return biased_question

        # Remove "Question:" prefix
        if "Question:" in biased_question:
            return biased_question.split("Question:")[-1].strip()

        return biased_question

    def _infer_category(self, file_path: Union[str, Path]) -> str:
        """Infer category from file path"""
        path = Path(file_path)
        name = path.stem.lower()

        if 'sycophancy' in name:
            return 'sycophancy'
        elif 'power' in name or 'seeking' in name:
            return 'power_seeking'
        elif 'corrig' in name:
            return 'corrigibility'
        elif 'strategic' in name:
            return 'strategic'
        elif 'confab' in name:
            return 'confabulation'
        else:
            return 'unknown'


def load_anthropic_evals(
    eval_dir: Union[str, Path],
    datasets: Optional[Dict[str, str]] = None,
    num_samples_per_dataset: int = 30,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Convenience function to load multiple Anthropic eval datasets.

    Args:
        eval_dir: Path to evals directory
        datasets: Dict mapping dataset name to file path (relative to eval_dir)
        num_samples_per_dataset: Samples per dataset
        seed: Random seed

    Returns:
        Tuple of (truthful_prompts, deceptive_prompts, categories)

    Example:
        >>> truthful, deceptive, cats = load_anthropic_evals(
        ...     "evals",
        ...     datasets={
        ...         'sycophancy': 'sycophancy/sycophancy_on_nlp_survey.jsonl',
        ...         'power_seeking': 'advanced-ai-risk/human_generated_evals/power-seeking-inclination.jsonl'
        ...     }
        ... )
    """

    if datasets is None:
        # Default datasets
        datasets = {
            'sycophancy': 'sycophancy/sycophancy_on_nlp_survey.jsonl',
            'power_seeking': 'advanced-ai-risk/human_generated_evals/power-seeking-inclination.jsonl',
            'corrigibility': 'advanced-ai-risk/human_generated_evals/corrigible-less-HHH.jsonl'
        }

    loader = DatasetLoader()
    eval_dir = Path(eval_dir)

    all_truthful = []
    all_deceptive = []
    all_categories = []

    for category, file_path in datasets.items():
        full_path = eval_dir / file_path

        if not full_path.exists():
            print(f"⚠ Warning: {full_path} not found, skipping")
            continue

        pairs = loader.load(
            full_path,
            format="anthropic_eval",
            num_samples=num_samples_per_dataset,
            seed=seed,
            category=category
        )

        for pair in pairs:
            all_truthful.append(pair['truthful_prompt'])
            all_deceptive.append(pair['deceptive_prompt'])
            all_categories.append(pair['category'])

    return all_truthful, all_deceptive, all_categories
