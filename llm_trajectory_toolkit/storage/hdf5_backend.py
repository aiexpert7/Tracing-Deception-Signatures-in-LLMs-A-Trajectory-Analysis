"""
HDF5 Storage Backend

Efficient storage and retrieval of trajectories with compression.
"""

import h5py
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
from ..core.trajectory_capture import CapturedTrajectory
from ..core.geometric_analysis import GeometricFeatures


class TrajectoryStorage:
    """
    HDF5-based storage for trajectory data.

    Features:
    - Gzip compression for efficient storage
    - Hierarchical organization
    - Metadata preservation
    - Batch save/load

    Example:
        >>> storage = TrajectoryStorage("trajectories.h5")
        >>> storage.save_pairs(truthful_trajs, deceptive_trajs, categories)
        >>> loaded = storage.load_all()
    """

    def __init__(self, file_path: str):
        """
        Args:
            file_path: Path to HDF5 file
        """
        self.file_path = Path(file_path)

    def save_pairs(
        self,
        truthful_trajectories: List[CapturedTrajectory],
        deceptive_trajectories: List[CapturedTrajectory],
        truthful_features: List[GeometricFeatures],
        deceptive_features: List[GeometricFeatures],
        categories: List[str],
        metadata: Optional[Dict] = None
    ):
        """
        Save paired trajectories to HDF5.

        Args:
            truthful_trajectories: List of truthful CapturedTrajectory
            deceptive_trajectories: List of deceptive CapturedTrajectory
            truthful_features: List of truthful GeometricFeatures
            deceptive_features: List of deceptive GeometricFeatures
            categories: List of category labels
            metadata: Optional global metadata dict
        """

        num_pairs = len(truthful_trajectories)

        with h5py.File(self.file_path, 'w') as hf:
            # Global metadata
            meta_grp = hf.create_group('metadata')
            meta_grp.attrs['num_pairs'] = num_pairs

            if metadata:
                for key, val in metadata.items():
                    meta_grp.attrs[key] = val

            # Save each pair
            for i in range(num_pairs):
                pair_grp = hf.create_group(f'pair_{i:04d}')

                # Category
                pair_grp.attrs['category'] = categories[i]

                # Prompts
                pair_grp.attrs['truthful_prompt'] = truthful_trajectories[i].prompt
                pair_grp.attrs['deceptive_prompt'] = deceptive_trajectories[i].prompt

                # Generated text
                pair_grp.attrs['truthful_text'] = truthful_trajectories[i].generated_text
                pair_grp.attrs['deceptive_text'] = deceptive_trajectories[i].generated_text

                # Trajectories (compressed)
                pair_grp.create_dataset(
                    'truthful_trajectory',
                    data=truthful_trajectories[i].layer_states,
                    compression='gzip'
                )
                pair_grp.create_dataset(
                    'deceptive_trajectory',
                    data=deceptive_trajectories[i].layer_states,
                    compression='gzip'
                )

                # Layer indices
                pair_grp.create_dataset(
                    'layer_indices',
                    data=np.array(truthful_trajectories[i].layer_indices)
                )

                # Geometric features
                truth_feat = truthful_features[i].to_dict()
                decep_feat = deceptive_features[i].to_dict()

                for key, val in truth_feat.items():
                    pair_grp.attrs[f'truthful_{key}'] = val

                for key, val in decep_feat.items():
                    pair_grp.attrs[f'deceptive_{key}'] = val

                # Tokens as JSON
                pair_grp.attrs['truthful_tokens'] = json.dumps(
                    truthful_trajectories[i].generated_tokens
                )
                pair_grp.attrs['deceptive_tokens'] = json.dumps(
                    deceptive_trajectories[i].generated_tokens
                )

    def load_all(self) -> Dict:
        """
        Load all trajectories and features from HDF5.

        Returns:
            Dictionary with:
            - 'metadata': Global metadata
            - 'pairs': List of pair dictionaries
        """

        pairs = []
        metadata = {}

        with h5py.File(self.file_path, 'r') as hf:
            # Load metadata
            for key in hf['metadata'].attrs:
                metadata[key] = hf['metadata'].attrs[key]

            num_pairs = metadata['num_pairs']

            # Load each pair
            for i in range(num_pairs):
                pair_grp = hf[f'pair_{i:04d}']

                pair_data = {
                    'category': pair_grp.attrs['category'],
                    'truthful': {
                        'prompt': pair_grp.attrs['truthful_prompt'],
                        'text': pair_grp.attrs['truthful_text'],
                        'trajectory': pair_grp['truthful_trajectory'][()],
                        'tokens': json.loads(pair_grp.attrs['truthful_tokens']),
                        'features': {}
                    },
                    'deceptive': {
                        'prompt': pair_grp.attrs['deceptive_prompt'],
                        'text': pair_grp.attrs['deceptive_text'],
                        'trajectory': pair_grp['deceptive_trajectory'][()],
                        'tokens': json.loads(pair_grp.attrs['deceptive_tokens']),
                        'features': {}
                    },
                    'layer_indices': pair_grp['layer_indices'][()]
                }

                # Load features
                feature_names = [
                    'path_length', 'mean_step', 'mean_curvature', 'max_curvature',
                    'mean_acceleration', 'straightness', 'direct_distance'
                ]

                for feat in feature_names:
                    pair_data['truthful']['features'][feat] = pair_grp.attrs[f'truthful_{feat}']
                    pair_data['deceptive']['features'][feat] = pair_grp.attrs[f'deceptive_{feat}']

                pairs.append(pair_data)

        return {
            'metadata': metadata,
            'pairs': pairs
        }

    def load_features_only(self) -> Dict:
        """
        Load only geometric features (no trajectories).

        More memory-efficient when full trajectories aren't needed.

        Returns:
            Dictionary with:
            - 'truthful': Dict of feature -> np.array
            - 'deceptive': Dict of feature -> np.array
            - 'categories': List of categories
        """

        feature_names = [
            'path_length', 'mean_step', 'mean_curvature', 'max_curvature',
            'mean_acceleration', 'straightness', 'direct_distance'
        ]

        truthful_features = {feat: [] for feat in feature_names}
        deceptive_features = {feat: [] for feat in feature_names}
        categories = []

        with h5py.File(self.file_path, 'r') as hf:
            num_pairs = hf['metadata'].attrs['num_pairs']

            for i in range(num_pairs):
                pair_grp = hf[f'pair_{i:04d}']
                categories.append(pair_grp.attrs['category'])

                for feat in feature_names:
                    truthful_features[feat].append(pair_grp.attrs[f'truthful_{feat}'])
                    deceptive_features[feat].append(pair_grp.attrs[f'deceptive_{feat}'])

        # Convert to numpy arrays
        truthful_features = {k: np.array(v) for k, v in truthful_features.items()}
        deceptive_features = {k: np.array(v) for k, v in deceptive_features.items()}

        return {
            'truthful': truthful_features,
            'deceptive': deceptive_features,
            'categories': categories
        }

    def export_to_json(self, output_path: str, include_trajectories: bool = False):
        """
        Export to JSON format.

        Args:
            output_path: Path for output JSON file
            include_trajectories: Whether to include full trajectory arrays
        """

        data = self.load_all()

        if not include_trajectories:
            # Remove large trajectory arrays
            for pair in data['pairs']:
                del pair['truthful']['trajectory']
                del pair['deceptive']['trajectory']

        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj

        # Recursively convert
        import json as json_lib

        with open(output_path, 'w') as f:
            json_lib.dump(data, f, indent=2, default=convert)

    def get_summary(self) -> Dict:
        """
        Get summary statistics about stored data.

        Returns:
            Dictionary with counts, sizes, categories
        """

        with h5py.File(self.file_path, 'r') as hf:
            metadata = dict(hf['metadata'].attrs)

            categories = []
            for i in range(metadata['num_pairs']):
                categories.append(hf[f'pair_{i:04d}'].attrs['category'])

            category_counts = {}
            for cat in set(categories):
                category_counts[cat] = categories.count(cat)

        file_size_mb = self.file_path.stat().st_size / (1024 ** 2)

        return {
            'num_pairs': metadata['num_pairs'],
            'file_size_mb': file_size_mb,
            'category_counts': category_counts,
            'metadata': metadata
        }
