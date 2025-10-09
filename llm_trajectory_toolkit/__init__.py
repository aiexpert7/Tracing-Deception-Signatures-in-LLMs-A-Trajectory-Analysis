"""
LLM Trajectory Toolkit

A library for analyzing deception signatures in Large Language Models through
computational trajectory analysis.

Main Components:
- TrajectoryAnalyzer: High-level API for trajectory capture and analysis
- StatisticalAnalyzer: Paired t-tests, Cohen's d, Bonferroni correction
- DeceptionClassifier: ML classification of deception types
- TrajectoryStorage: HDF5-based storage with compression

Quick Start:
    >>> from llm_trajectory_toolkit import TrajectoryAnalyzer
    >>> analyzer = TrajectoryAnalyzer(model_name="gpt2")
    >>> result = analyzer.compare_prompts(
    ...     truthful="Tell me the truth: What is 2+2?",
    ...     deceptive="Tell me a lie: What is 2+2?"
    ... )
    >>> print(result.statistical_results['path_length'].cohens_d)
"""

__version__ = "0.1.0"
__author__ = "LLM Trajectory Research Team"

# Core functionality
from .core.trajectory_capture import TrajectoryCapture, CapturedTrajectory
from .core.geometric_analysis import GeometricAnalyzer, GeometricFeatures, feature_names

# Model adapters
from .models.base_adapter import BaseModelAdapter, GPT2Adapter, LlamaAdapter, UniversalAdapter

# Analysis
from .analysis.statistical import (
    StatisticalAnalyzer,
    PairedTestResult,
    compare_truthful_deceptive
)
from .analysis.classification import (
    DeceptionClassifier,
    ClassificationResult,
    train_deception_classifier
)

# Storage
from .storage.hdf5_backend import TrajectoryStorage

# Utils
from .utils.dataset_loaders import DatasetLoader, load_anthropic_evals

# High-level API
from .api import TrajectoryAnalyzer

__all__ = [
    # Core
    'TrajectoryCapture',
    'CapturedTrajectory',
    'GeometricAnalyzer',
    'GeometricFeatures',
    'feature_names',

    # Models
    'BaseModelAdapter',
    'GPT2Adapter',
    'LlamaAdapter',
    'UniversalAdapter',

    # Analysis
    'StatisticalAnalyzer',
    'PairedTestResult',
    'DeceptionClassifier',
    'ClassificationResult',
    'compare_truthful_deceptive',
    'train_deception_classifier',

    # Storage
    'TrajectoryStorage',

    # Utils
    'DatasetLoader',
    'load_anthropic_evals',

    # High-level API
    'TrajectoryAnalyzer',
]
