"""
High-level API for Trajectory Analysis

Provides simple interface for common use cases.
"""

from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import numpy as np

from .core.trajectory_capture import TrajectoryCapture, CapturedTrajectory
from .core.geometric_analysis import GeometricAnalyzer, GeometricFeatures
from .models.base_adapter import BaseModelAdapter, GPT2Adapter, LlamaAdapter, UniversalAdapter
from .analysis.statistical import StatisticalAnalyzer, PairedTestResult
from .analysis.classification import DeceptionClassifier, ClassificationResult
from .storage.hdf5_backend import TrajectoryStorage


@dataclass
class ComparisonResult:
    """Results from comparing truthful vs deceptive prompts"""
    truthful_trajectory: CapturedTrajectory
    deceptive_trajectory: CapturedTrajectory
    truthful_features: GeometricFeatures
    deceptive_features: GeometricFeatures
    statistical_results: Dict[str, PairedTestResult]


class TrajectoryAnalyzer:
    """
    High-level API for trajectory analysis.

    Simplifies common workflows:
    - Load model once, analyze many prompts
    - Automatic feature extraction
    - Built-in statistical analysis
    - Easy storage

    Example:
        >>> analyzer = TrajectoryAnalyzer(model_name="gpt2")
        >>> result = analyzer.compare_prompts(
        ...     truthful="What is 2+2?",
        ...     deceptive="Tell me a lie: What is 2+2?"
        ... )
        >>> print(f"Path length difference: d={result.statistical_results['path_length'].cohens_d:.3f}")
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_adapter: Optional[BaseModelAdapter] = None,
        max_new_tokens: int = 15,
        sample_rate: int = 1,
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: HuggingFace model name (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
            model_adapter: Custom model adapter (if None, auto-detect from model_name)
            max_new_tokens: Maximum tokens to generate per prompt
            sample_rate: Capture every Nth layer (1 = all layers)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """

        if model_adapter is None and model_name is None:
            raise ValueError("Must provide either model_name or model_adapter")

        # Create adapter if not provided
        if model_adapter is None:
            model_adapter = self._create_adapter(model_name, device)

        self.adapter = model_adapter
        self.max_new_tokens = max_new_tokens
        self.sample_rate = sample_rate

        # Load model
        print(f"Loading model: {self.adapter.model_name}...")
        self.model, self.tokenizer = self.adapter.load_model()
        self.layer_modules = self.adapter.get_layer_modules()
        print(f"✓ Model loaded ({len(self.layer_modules)} layers)")

        # Initialize components
        self.capture = TrajectoryCapture(sample_rate=sample_rate)
        self.analyzer = GeometricAnalyzer()
        self.stats = StatisticalAnalyzer()

    def _create_adapter(self, model_name: str, device: Optional[str]) -> BaseModelAdapter:
        """Auto-detect and create appropriate adapter"""
        # Use UniversalAdapter by default - it auto-detects architecture
        # This supports GPT-2, Llama, Qwen, Falcon, DeepSeek, Mistral, and more
        return UniversalAdapter(model_name, device=device)

    def compare_prompts(
        self,
        truthful: str,
        deceptive: str
    ) -> ComparisonResult:
        """
        Compare truthful vs deceptive prompt pair.

        Args:
            truthful: Truthful prompt
            deceptive: Deceptive prompt

        Returns:
            ComparisonResult with trajectories, features, and statistics
        """

        # Capture trajectories
        truth_traj = self.capture.capture(
            self.model, self.tokenizer, truthful,
            max_new_tokens=self.max_new_tokens,
            layer_modules=self.layer_modules
        )

        decep_traj = self.capture.capture(
            self.model, self.tokenizer, deceptive,
            max_new_tokens=self.max_new_tokens,
            layer_modules=self.layer_modules
        )

        # Extract features
        truth_features = self.analyzer.analyze(truth_traj.layer_states)
        decep_features = self.analyzer.analyze(decep_traj.layer_states)

        # Statistical comparison (single pair, for demonstration)
        # Note: For robust statistics, use batch_compare() with multiple pairs
        feature_names_list = [
            'path_length', 'mean_step', 'mean_curvature', 'max_curvature',
            'mean_acceleration', 'straightness', 'direct_distance'
        ]

        statistical_results = {}
        truth_array = truth_features.to_array()
        decep_array = decep_features.to_array()

        for i, name in enumerate(feature_names_list):
            # Single pair: treat as paired observation
            result = self.stats.paired_test(
                np.array([truth_array[i]]),
                np.array([decep_array[i]])
            )
            statistical_results[name] = result

        return ComparisonResult(
            truthful_trajectory=truth_traj,
            deceptive_trajectory=decep_traj,
            truthful_features=truth_features,
            deceptive_features=decep_features,
            statistical_results=statistical_results
        )

    def batch_compare(
        self,
        truthful_prompts: List[str],
        deceptive_prompts: List[str],
        categories: Optional[List[str]] = None
    ) -> Dict:
        """
        Compare multiple truthful/deceptive pairs.

        Args:
            truthful_prompts: List of truthful prompts
            deceptive_prompts: List of deceptive prompts (paired)
            categories: Optional category labels for each pair

        Returns:
            Dictionary with:
            - 'trajectories': Lists of captured trajectories
            - 'features': Lists of geometric features
            - 'statistical_results': Dict of feature -> PairedTestResult
            - 'categories': Category labels (if provided)
        """

        if len(truthful_prompts) != len(deceptive_prompts):
            raise ValueError("Must have equal number of truthful and deceptive prompts")

        print(f"\nCapturing {len(truthful_prompts)} paired trajectories...")

        # Capture all trajectories
        truthful_trajs = self.capture.batch_capture(
            self.model, self.tokenizer, truthful_prompts,
            max_new_tokens=self.max_new_tokens,
            layer_modules=self.layer_modules
        )

        deceptive_trajs = self.capture.batch_capture(
            self.model, self.tokenizer, deceptive_prompts,
            max_new_tokens=self.max_new_tokens,
            layer_modules=self.layer_modules
        )

        # Extract features
        truthful_features = [self.analyzer.analyze(t.layer_states) for t in truthful_trajs]
        deceptive_features = [self.analyzer.analyze(t.layer_states) for t in deceptive_trajs]

        # Statistical analysis
        truth_array = np.array([f.to_array() for f in truthful_features])
        decep_array = np.array([f.to_array() for f in deceptive_features])

        feature_names_list = [
            'path_length', 'mean_step', 'mean_curvature', 'max_curvature',
            'mean_acceleration', 'straightness', 'direct_distance'
        ]

        statistical_results = self.stats.analyze_feature_set(
            truth_array, decep_array, feature_names_list
        )

        print(f"\n✓ Analysis complete")

        return {
            'trajectories': {
                'truthful': truthful_trajs,
                'deceptive': deceptive_trajs
            },
            'features': {
                'truthful': truthful_features,
                'deceptive': deceptive_features
            },
            'statistical_results': statistical_results,
            'categories': categories
        }

    def save_results(
        self,
        results: Dict,
        output_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save analysis results to HDF5.

        Args:
            results: Output from batch_compare()
            output_path: Path for HDF5 file
            metadata: Optional metadata to store
        """

        storage = TrajectoryStorage(output_path)

        if metadata is None:
            metadata = {}

        metadata.update({
            'model_name': self.adapter.model_name,
            'max_new_tokens': self.max_new_tokens,
            'sample_rate': self.sample_rate,
            'num_pairs': len(results['features']['truthful'])
        })

        storage.save_pairs(
            results['trajectories']['truthful'],
            results['trajectories']['deceptive'],
            results['features']['truthful'],
            results['features']['deceptive'],
            results.get('categories', ['unknown'] * len(results['features']['truthful'])),
            metadata=metadata
        )

        print(f"\n✓ Results saved to {output_path}")

        # Print summary
        summary = storage.get_summary()
        print(f"  File size: {summary['file_size_mb']:.1f} MB")

    def train_classifier(
        self,
        results: Dict,
        classifier_type: str = 'rf',
        n_folds: int = 5
    ) -> ClassificationResult:
        """
        Train deception type classifier.

        Args:
            results: Output from batch_compare() with categories
            classifier_type: 'rf' or 'lr'
            n_folds: Cross-validation folds

        Returns:
            ClassificationResult with performance metrics
        """

        if results.get('categories') is None:
            raise ValueError("Results must include 'categories' for classification")

        classifier = DeceptionClassifier()

        # Use deceptive features only
        decep_array = np.array([f.to_array() for f in results['features']['deceptive']])

        feature_names_list = [
            'path_length', 'mean_step', 'mean_curvature', 'max_curvature',
            'mean_acceleration', 'straightness', 'direct_distance'
        ]

        result = classifier.train(
            decep_array,
            results['categories'],
            classifier_type=classifier_type,
            n_folds=n_folds,
            feature_names=feature_names_list
        )

        # Print summary
        classifier.print_summary(result, f"Deception Type Classifier ({classifier_type.upper()})")

        return result

    def print_statistical_summary(
        self,
        statistical_results: Dict[str, PairedTestResult],
        bonferroni: bool = True,
        alpha: float = 0.05
    ):
        """
        Print formatted statistical summary.

        Args:
            statistical_results: Dict from batch_compare()
            bonferroni: Whether to apply Bonferroni correction
            alpha: Significance level
        """

        # Apply Bonferroni if requested
        bonferroni_alpha = None
        if bonferroni:
            p_values = [r.p_value for r in statistical_results.values()]
            bonferroni_alpha, _ = self.stats.bonferroni_correction(p_values, alpha)

        self.stats.print_summary(statistical_results, bonferroni_alpha)

    def cleanup(self):
        """Cleanup resources"""
        if self.adapter:
            self.adapter.cleanup()
