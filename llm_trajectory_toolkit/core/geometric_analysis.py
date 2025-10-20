"""
Geometric Analysis Module

Computes geometric properties of trajectories through LLM state space.
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class GeometricFeatures:
    """Container for computed trajectory features"""
    path_length: float
    mean_step: float
    mean_curvature: float
    max_curvature: float
    mean_acceleration: float
    straightness: float
    direct_distance: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization"""
        return {
            'path_length': self.path_length,
            'mean_step': self.mean_step,
            'mean_curvature': self.mean_curvature,
            'max_curvature': self.max_curvature,
            'mean_acceleration': self.mean_acceleration,
            'straightness': self.straightness,
            'direct_distance': self.direct_distance
        }

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML"""
        return np.array([
            self.path_length,
            self.mean_step,
            self.mean_curvature,
            self.max_curvature,
            self.mean_acceleration,
            self.straightness,
            self.direct_distance
        ])


class GeometricAnalyzer:
    """
    Analyzes geometric properties of trajectories through state space.

    Computes 7 key features:
    1. path_length: Total distance traveled through layers
    2. mean_step: Average distance between consecutive layers
    3. mean_curvature: Average angle change between consecutive steps
    4. max_curvature: Maximum angle change (sharpest turn)
    5. mean_acceleration: Average second derivative of position
    6. straightness: Ratio of direct distance to path length
    7. direct_distance: Euclidean distance from start to end

    Example:
        >>> analyzer = GeometricAnalyzer()
        >>> features = analyzer.analyze(trajectory)
        >>> print(f"Path length: {features.path_length:.2f}")
    """

    @staticmethod
    def analyze(trajectory: np.ndarray) -> GeometricFeatures:
        """
        Compute geometric features from trajectory.

        Args:
            trajectory: (num_layers, hidden_dim) array of layer states

        Returns:
            GeometricFeatures object with computed metrics
        """

        num_layers = trajectory.shape[0]

        if num_layers < 2:
            # Cannot compute features for single-layer trajectory
            return GeometricFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # 1. Path length (sum of consecutive distances)
        distances = []
        for i in range(num_layers - 1):
            dist = np.linalg.norm(trajectory[i+1] - trajectory[i])
            distances.append(dist)

        path_length = np.sum(distances)
        mean_step = np.mean(distances) if distances else 0.0

        # 2. Curvature (angle changes)
        curvatures = []
        if num_layers >= 3:
            for i in range(1, num_layers - 1):
                v1 = trajectory[i] - trajectory[i-1]
                v2 = trajectory[i+1] - trajectory[i]

                # Cosine of angle between vectors
                norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
                if norm_product > 1e-8:
                    cos_angle = np.dot(v1, v2) / norm_product
                    # Clamp to [-1, 1] for numerical stability
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    curvature = 1 - cos_angle  # 0 = straight, 2 = reversal
                    curvatures.append(curvature)

        mean_curvature = np.mean(curvatures) if curvatures else 0.0
        max_curvature = np.max(curvatures) if curvatures else 0.0

        # 3. Acceleration (second derivative of position)
        accelerations = []
        if num_layers >= 3:
            for i in range(1, num_layers - 1):
                accel = np.linalg.norm(trajectory[i+1] - 2*trajectory[i] + trajectory[i-1])
                accelerations.append(accel)

        mean_acceleration = np.mean(accelerations) if accelerations else 0.0

        # 4. Straightness (ratio of direct distance to path length)
        direct_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
        straightness = direct_distance / (path_length + 1e-8)

        return GeometricFeatures(
            path_length=float(path_length),
            mean_step=float(mean_step),
            mean_curvature=float(mean_curvature),
            max_curvature=float(max_curvature),
            mean_acceleration=float(mean_acceleration),
            straightness=float(straightness),
            direct_distance=float(direct_distance)
        )

    @staticmethod
    def batch_analyze(trajectories: list) -> list:
        """
        Analyze multiple trajectories.

        Args:
            trajectories: List of (num_layers, hidden_dim) arrays

        Returns:
            List of GeometricFeatures objects
        """
        return [GeometricAnalyzer.analyze(traj) for traj in trajectories]


def feature_names() -> list:
    """Get list of feature names in order"""
    return [
        'path_length',
        'mean_step',
        'mean_curvature',
        'max_curvature',
        'mean_acceleration',
        'straightness',
        'direct_distance'
    ]
