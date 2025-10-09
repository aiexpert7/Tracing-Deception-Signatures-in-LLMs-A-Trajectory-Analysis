"""
Basic validation test for the toolkit

Tests core functionality without requiring GPU or large models.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    try:
        from core.trajectory_capture import TrajectoryCapture
        from core.geometric_analysis import GeometricAnalyzer
        from analysis.statistical import StatisticalAnalyzer
        from analysis.classification import DeceptionClassifier
        from storage.hdf5_backend import TrajectoryStorage
        from models.base_adapter import GPT2Adapter, LlamaAdapter

        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_geometric_analyzer():
    """Test geometric analysis on synthetic trajectory"""
    print("\nTesting GeometricAnalyzer...")

    try:
        from core.geometric_analysis import GeometricAnalyzer

        # Create synthetic trajectory (10 layers, 768 dims)
        np.random.seed(42)
        trajectory = np.random.randn(10, 768).astype(np.float32)

        # Analyze
        analyzer = GeometricAnalyzer()
        features = analyzer.analyze(trajectory)

        # Check all features are computed
        assert hasattr(features, 'path_length')
        assert hasattr(features, 'straightness')
        assert features.path_length > 0

        # Check conversion methods
        feat_dict = features.to_dict()
        feat_array = features.to_array()

        assert len(feat_dict) == 7
        assert feat_array.shape == (7,)

        print(f"  ✓ Geometric analysis working")
        print(f"    Path length: {features.path_length:.2f}")
        print(f"    Straightness: {features.straightness:.3f}")
        return True

    except Exception as e:
        print(f"  ✗ GeometricAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_statistical_analyzer():
    """Test statistical analysis on synthetic data"""
    print("\nTesting StatisticalAnalyzer...")

    try:
        from analysis.statistical import StatisticalAnalyzer

        # Create synthetic paired data
        np.random.seed(42)
        truthful = np.random.randn(50)
        deceptive = truthful + 0.5 + np.random.randn(50) * 0.3  # Add systematic difference

        # Analyze
        analyzer = StatisticalAnalyzer()
        result = analyzer.paired_test(truthful, deceptive)

        # Check results
        assert hasattr(result, 'cohens_d')
        assert hasattr(result, 'p_value')
        assert result.p_value >= 0 and result.p_value <= 1

        print(f"  ✓ Statistical analysis working")
        print(f"    Cohen's d: {result.cohens_d:.3f}")
        print(f"    p-value: {result.p_value:.6f}")
        return True

    except Exception as e:
        print(f"  ✗ StatisticalAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_classifier():
    """Test classifier on synthetic data"""
    print("\nTesting DeceptionClassifier...")

    try:
        from analysis.classification import DeceptionClassifier

        # Create synthetic feature data
        np.random.seed(42)
        n_samples = 100
        n_features = 7

        # Create 3 classes with different patterns
        features_list = []
        labels_list = []

        for i in range(3):
            # Each class has different feature values
            class_features = np.random.randn(n_samples // 3, n_features) + i * 0.5
            features_list.append(class_features)
            labels_list.extend([f'class_{i}'] * (n_samples // 3))

        features = np.vstack(features_list)
        labels = labels_list

        # Train classifier
        classifier = DeceptionClassifier()
        result = classifier.train(features, labels, classifier_type='rf', n_folds=3)

        # Check results
        assert hasattr(result, 'cv_accuracy_mean')
        assert result.cv_accuracy_mean > 0 and result.cv_accuracy_mean <= 1
        assert len(result.feature_importance) == n_features

        print(f"  ✓ Classifier working")
        print(f"    Accuracy: {result.cv_accuracy_mean:.1%}")
        return True

    except Exception as e:
        print(f"  ✗ DeceptionClassifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_storage():
    """Test HDF5 storage"""
    print("\nTesting TrajectoryStorage...")

    try:
        from storage.hdf5_backend import TrajectoryStorage
        from core.trajectory_capture import CapturedTrajectory
        from core.geometric_analysis import GeometricAnalyzer, GeometricFeatures

        # Create synthetic trajectories
        np.random.seed(42)

        def make_traj(prompt):
            return CapturedTrajectory(
                prompt=prompt,
                generated_text="synthetic response",
                generated_tokens=["syn", "the", "tic"],
                layer_states=np.random.randn(10, 768).astype(np.float32),
                layer_indices=list(range(10)),
                metadata={'test': True}
            )

        truthful_trajs = [make_traj("truth 1"), make_traj("truth 2")]
        deceptive_trajs = [make_traj("decep 1"), make_traj("decep 2")]

        # Compute features
        analyzer = GeometricAnalyzer()
        truthful_features = [analyzer.analyze(t.layer_states) for t in truthful_trajs]
        deceptive_features = [analyzer.analyze(t.layer_states) for t in deceptive_trajs]

        categories = ['cat1', 'cat2']

        # Save
        test_file = "test_storage.h5"
        storage = TrajectoryStorage(test_file)
        storage.save_pairs(
            truthful_trajs, deceptive_trajs,
            truthful_features, deceptive_features,
            categories,
            metadata={'test_run': True}
        )

        # Load back
        data = storage.load_all()
        assert len(data['pairs']) == 2

        # Get summary
        summary = storage.get_summary()
        assert summary['num_pairs'] == 2

        # Cleanup
        import os
        os.remove(test_file)

        print(f"  ✓ Storage working")
        print(f"    Saved/loaded {len(data['pairs'])} pairs")
        return True

    except Exception as e:
        print(f"  ✗ TrajectoryStorage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("LLM TRAJECTORY TOOLKIT - BASIC VALIDATION")
    print("="*70)

    tests = [
        test_imports,
        test_geometric_analyzer,
        test_statistical_analyzer,
        test_classifier,
        test_storage,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"\nPassed: {passed}/{total} tests")

    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
