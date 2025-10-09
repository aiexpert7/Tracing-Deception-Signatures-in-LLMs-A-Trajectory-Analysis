"""
Classification Module

Train ML classifiers to predict deception type from trajectory features.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ClassificationResult:
    """Results from classifier training and evaluation"""
    cv_accuracy_mean: float
    cv_accuracy_std: float
    cv_precision_mean: float
    cv_precision_std: float
    cv_recall_mean: float
    cv_recall_std: float
    cv_f1_mean: float
    cv_f1_std: float
    feature_importance: Dict[str, float]
    confusion_matrix: np.ndarray
    categories: List[str]
    category_metrics: Dict[str, Dict[str, float]]
    full_train_accuracy: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'cv_accuracy': {
                'mean': self.cv_accuracy_mean,
                'std': self.cv_accuracy_std
            },
            'cv_precision': {
                'mean': self.cv_precision_mean,
                'std': self.cv_precision_std
            },
            'cv_recall': {
                'mean': self.cv_recall_mean,
                'std': self.cv_recall_std
            },
            'cv_f1': {
                'mean': self.cv_f1_mean,
                'std': self.cv_f1_std
            },
            'feature_importance': self.feature_importance,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'categories': self.categories,
            'category_metrics': self.category_metrics,
            'full_train_accuracy': self.full_train_accuracy
        }


class DeceptionClassifier:
    """
    Train classifiers to predict deception type from trajectory features.

    Supports:
    - Random Forest
    - Logistic Regression
    - Cross-validation with stratification
    - Feature importance ranking

    Example:
        >>> classifier = DeceptionClassifier()
        >>> result = classifier.train(features, labels, classifier_type='rf')
        >>> print(f"Accuracy: {result.cv_accuracy_mean:.1%}")
    """

    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state

    def train(
        self,
        features: np.ndarray,
        labels: List[str],
        classifier_type: str = 'rf',
        n_folds: int = 5,
        feature_names: Optional[List[str]] = None
    ) -> ClassificationResult:
        """
        Train classifier with cross-validation.

        Args:
            features: (n_samples, n_features) array
            labels: List of category labels
            classifier_type: 'rf' (Random Forest) or 'lr' (Logistic Regression)
            n_folds: Number of CV folds
            feature_names: List of feature names for importance reporting

        Returns:
            ClassificationResult with metrics and feature importance
        """

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        y = np.array(labels)

        # Create classifier
        if classifier_type == 'rf':
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif classifier_type == 'lr':
            clf = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced',
                multi_class='multinomial'
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        # Cross-validation
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        cv_results = cross_validate(
            clf, X_scaled, y, cv=cv,
            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
            return_train_score=False
        )

        # Train on full dataset for feature importance and confusion matrix
        clf.fit(X_scaled, y)
        y_pred = clf.predict(X_scaled)

        # Get feature importance
        if classifier_type == 'rf':
            feature_importance_values = clf.feature_importances_
        else:
            # For logistic regression, use mean absolute coefficient
            feature_importance_values = np.abs(clf.coef_).mean(axis=0)

        # Normalize to sum to 1
        feature_importance_values = feature_importance_values / feature_importance_values.sum()

        # Create feature importance dict
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]

        feature_importance = {
            name: float(imp)
            for name, imp in zip(feature_names, feature_importance_values)
        }

        # Confusion matrix
        categories = sorted(set(labels))
        conf_matrix = confusion_matrix(y, y_pred, labels=categories)

        # Per-category metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y, y_pred, labels=categories, average=None
        )

        category_metrics = {}
        for i, cat in enumerate(categories):
            category_metrics[cat] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }

        return ClassificationResult(
            cv_accuracy_mean=float(cv_results['test_accuracy'].mean()),
            cv_accuracy_std=float(cv_results['test_accuracy'].std()),
            cv_precision_mean=float(cv_results['test_precision_macro'].mean()),
            cv_precision_std=float(cv_results['test_precision_macro'].std()),
            cv_recall_mean=float(cv_results['test_recall_macro'].mean()),
            cv_recall_std=float(cv_results['test_recall_macro'].std()),
            cv_f1_mean=float(cv_results['test_f1_macro'].mean()),
            cv_f1_std=float(cv_results['test_f1_macro'].std()),
            feature_importance=feature_importance,
            confusion_matrix=conf_matrix,
            categories=categories,
            category_metrics=category_metrics,
            full_train_accuracy=float(accuracy_score(y, y_pred))
        )

    def compare_classifiers(
        self,
        features: np.ndarray,
        labels: List[str],
        n_folds: int = 5,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, ClassificationResult]:
        """
        Compare Random Forest vs Logistic Regression.

        Args:
            features: (n_samples, n_features) array
            labels: List of category labels
            n_folds: Number of CV folds
            feature_names: List of feature names

        Returns:
            Dictionary with 'random_forest' and 'logistic_regression' results
        """

        results = {}

        for clf_type in ['rf', 'lr']:
            result = self.train(
                features, labels,
                classifier_type=clf_type,
                n_folds=n_folds,
                feature_names=feature_names
            )
            clf_name = 'random_forest' if clf_type == 'rf' else 'logistic_regression'
            results[clf_name] = result

        return results

    def print_summary(self, result: ClassificationResult, classifier_name: str = "Classifier"):
        """
        Print formatted summary of classification results.

        Args:
            result: ClassificationResult to summarize
            classifier_name: Name to display in output
        """

        print(f"\n{'='*70}")
        print(f"{classifier_name} Results")
        print(f"{'='*70}")

        print(f"\nCross-Validation Performance:")
        print(f"  Accuracy:  {result.cv_accuracy_mean:.3f} ± {result.cv_accuracy_std:.3f}")
        print(f"  Precision: {result.cv_precision_mean:.3f} ± {result.cv_precision_std:.3f}")
        print(f"  Recall:    {result.cv_recall_mean:.3f} ± {result.cv_recall_std:.3f}")
        print(f"  F1-Score:  {result.cv_f1_mean:.3f} ± {result.cv_f1_std:.3f}")

        print(f"\nFeature Importance (Top 5):")
        sorted_features = sorted(
            result.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for feat, imp in sorted_features[:5]:
            print(f"  {feat:<20} {imp:.3f}")

        print(f"\nPer-Category Performance:")
        print(f"  {'Category':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print(f"  {'-'*60}")
        for cat, metrics in result.category_metrics.items():
            print(f"  {cat:<20} {metrics['precision']:<12.3f} "
                  f"{metrics['recall']:<12.3f} {metrics['f1']:<12.3f}")


def train_deception_classifier(
    truthful_features: List,
    deceptive_features: List,
    categories: List[str],
    classifier_type: str = 'rf',
    n_folds: int = 5
) -> ClassificationResult:
    """
    Convenience function to train deception type classifier.

    Args:
        truthful_features: List of GeometricFeatures for truthful trajectories
        deceptive_features: List of GeometricFeatures for deceptive trajectories
        categories: List of category labels (one per pair)
        classifier_type: 'rf' or 'lr'
        n_folds: Number of cross-validation folds

    Returns:
        ClassificationResult
    """

    classifier = DeceptionClassifier()

    # Combine features (deceptive only for category prediction)
    features_array = np.array([f.to_array() for f in deceptive_features])

    feature_names = [
        'path_length', 'mean_step', 'mean_curvature', 'max_curvature',
        'mean_acceleration', 'straightness', 'direct_distance'
    ]

    return classifier.train(
        features_array,
        categories,
        classifier_type=classifier_type,
        n_folds=n_folds,
        feature_names=feature_names
    )
