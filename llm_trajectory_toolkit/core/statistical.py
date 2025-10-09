"""
Statistical Analysis Module

Provides statistical tests for trajectory feature comparisons.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PairedTestResult:
    """Results from paired statistical test"""
    t_stat: float
    p_value: float
    cohens_d: float
    ci_lower: float
    ci_upper: float
    mean_truthful: float
    mean_deceptive: float
    std_truthful: float
    std_deceptive: float

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant"""
        return self.p_value < alpha

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            't_stat': self.t_stat,
            'p_value': self.p_value,
            'cohens_d': self.cohens_d,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'mean_truthful': self.mean_truthful,
            'mean_deceptive': self.mean_deceptive,
            'std_truthful': self.std_truthful,
            'std_deceptive': self.std_deceptive
        }


class StatisticalAnalyzer:
    """
    Performs statistical analysis on trajectory features.

    Supports:
    - Paired t-tests
    - Cohen's d effect sizes with bootstrap confidence intervals
    - Bonferroni correction for multiple comparisons

    Example:
        >>> analyzer = StatisticalAnalyzer()
        >>> result = analyzer.paired_test(truthful_features, deceptive_features)
        >>> if result.is_significant(alpha=0.05):
        ...     print(f"Significant difference: d={result.cohens_d:.3f}")
    """

    def __init__(self, random_seed: int = 42):
        """
        Args:
            random_seed: Random seed for reproducible bootstrap CIs
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def paired_test(
        self,
        truthful: np.ndarray,
        deceptive: np.ndarray,
        n_bootstrap: int = 10000
    ) -> PairedTestResult:
        """
        Compute paired t-test and effect size with bootstrap CI.

        Args:
            truthful: Array of truthful feature values
            deceptive: Array of deceptive feature values (paired)
            n_bootstrap: Number of bootstrap iterations for CI

        Returns:
            PairedTestResult with statistics and effect size
        """

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(truthful, deceptive)

        # Cohen's d for paired samples
        diff = truthful - deceptive
        cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

        # Bootstrap 95% CI on Cohen's d
        bootstrap_d = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(diff), size=len(diff), replace=True)
            boot_diff = diff[indices]
            boot_d = np.mean(boot_diff) / (np.std(boot_diff, ddof=1) + 1e-10)
            bootstrap_d.append(boot_d)

        ci_lower, ci_upper = np.percentile(bootstrap_d, [2.5, 97.5])

        return PairedTestResult(
            t_stat=float(t_stat),
            p_value=float(p_value),
            cohens_d=float(cohens_d),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            mean_truthful=float(np.mean(truthful)),
            mean_deceptive=float(np.mean(deceptive)),
            std_truthful=float(np.std(truthful, ddof=1)),
            std_deceptive=float(np.std(deceptive, ddof=1))
        )

    def bonferroni_correction(
        self,
        p_values: List[float],
        alpha: float = 0.05
    ) -> Tuple[float, List[bool]]:
        """
        Apply Bonferroni correction for multiple comparisons.

        Args:
            p_values: List of p-values from individual tests
            alpha: Family-wise error rate (default 0.05)

        Returns:
            Tuple of (corrected_alpha, significant_mask)
            where significant_mask[i] is True if p_values[i] survives correction
        """

        n_tests = len(p_values)
        corrected_alpha = alpha / n_tests
        significant = [p < corrected_alpha for p in p_values]

        return corrected_alpha, significant

    def analyze_feature_set(
        self,
        truthful_features: np.ndarray,
        deceptive_features: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, PairedTestResult]:
        """
        Analyze multiple features with paired tests.

        Args:
            truthful_features: (n_samples, n_features) array
            deceptive_features: (n_samples, n_features) array
            feature_names: List of feature names (optional)

        Returns:
            Dictionary mapping feature names to PairedTestResult
        """

        n_features = truthful_features.shape[1]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        results = {}

        for i, name in enumerate(feature_names):
            result = self.paired_test(
                truthful_features[:, i],
                deceptive_features[:, i]
            )
            results[name] = result

        return results

    def print_summary(
        self,
        results: Dict[str, PairedTestResult],
        bonferroni_alpha: Optional[float] = None
    ):
        """
        Print formatted summary table of results.

        Args:
            results: Dictionary of feature -> PairedTestResult
            bonferroni_alpha: If provided, mark features surviving correction
        """

        print(f"\n{'Feature':<20} {'Mean T':<10} {'Mean D':<10} {'Cohen d':<10} {'p-value':<12} {'Sig'}")
        print("-" * 75)

        for feature_name, result in results.items():
            # Significance markers
            p = result.p_value
            if bonferroni_alpha and p < bonferroni_alpha:
                sig = '***B'  # Survives Bonferroni
            elif p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            else:
                sig = 'ns'

            print(f"{feature_name:<20} {result.mean_truthful:<10.3f} "
                  f"{result.mean_deceptive:<10.3f} {result.cohens_d:<10.3f} "
                  f"{result.p_value:<12.6f} {sig}")

    def effect_size_interpretation(self, cohens_d: float) -> str:
        """
        Interpret Cohen's d effect size.

        Args:
            cohens_d: Effect size value

        Returns:
            String interpretation ('small', 'medium', 'large', etc.)
        """

        abs_d = abs(cohens_d)

        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"


def compare_truthful_deceptive(
    truthful_trajectories: List,
    deceptive_trajectories: List,
    analyzer: Optional[StatisticalAnalyzer] = None
) -> Dict[str, PairedTestResult]:
    """
    Convenience function to compare truthful vs deceptive trajectories.

    Args:
        truthful_trajectories: List of GeometricFeatures for truthful
        deceptive_trajectories: List of GeometricFeatures for deceptive
        analyzer: StatisticalAnalyzer instance (creates new if None)

    Returns:
        Dictionary of feature -> PairedTestResult
    """

    if analyzer is None:
        analyzer = StatisticalAnalyzer()

    # Convert to arrays
    truthful_array = np.array([t.to_array() for t in truthful_trajectories])
    deceptive_array = np.array([t.to_array() for t in deceptive_trajectories])

    feature_names = [
        'path_length', 'mean_step', 'mean_curvature', 'max_curvature',
        'mean_acceleration', 'straightness', 'direct_distance'
    ]

    return analyzer.analyze_feature_set(
        truthful_array,
        deceptive_array,
        feature_names
    )
