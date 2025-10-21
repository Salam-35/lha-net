import numpy as np
import nibabel as nib
import SimpleITK as sitk
from typing import Tuple, Optional, Dict, Union, List
import os
from scipy import ndimage
from pathlib import Path
import torch
from torch.utils.data import Dataset


class AMOS22Preprocessor:
    """Preprocessing pipeline for AMOS22 dataset optimized for LHA-Net"""

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
        target_size: Optional[Tuple[int, int, int]] = None,
        intensity_normalization: str = "z_score",
        clip_range: Tuple[float, float] = (-200, 300),
        resampling_method: str = "linear"
    ):
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.intensity_normalization = intensity_normalization
        self.clip_range = clip_range
        self.resampling_method = resampling_method

    def _load_image(self, image_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """Load medical image and extract metadata"""
        if str(image_path).endswith('.nii.gz') or str(image_path).endswith('.nii'):
            # Use nibabel for NIfTI files
            img = nib.load(str(image_path))
            image_data = img.get_fdata().astype(np.float32)
            affine = img.affine
            header = img.header

            # Get spacing from header
            spacing = header.get_zooms()[:3]

            metadata = {
                'original_spacing': spacing,
                'original_shape': image_data.shape,
                'affine': affine,
                'header': header
            }

        else:
            # Use SimpleITK for other formats
            img = sitk.ReadImage(str(image_path))
            image_data = sitk.GetArrayFromImage(img).astype(np.float32)
            spacing = img.GetSpacing()[::-1]  # ITK uses (x,y,z), we want (z,y,x)
            origin = img.GetOrigin()
            direction = img.GetDirection()

            metadata = {
                'original_spacing': spacing,
                'original_shape': image_data.shape,
                'origin': origin,
                'direction': direction,
                'sitk_image': img
            }

        return image_data, metadata

    def _resample_image(
        self,
        image: np.ndarray,
        original_spacing: Tuple[float, float, float],
        target_spacing: Tuple[float, float, float],
        is_label: bool = False
    ) -> np.ndarray:
        """Resample image to target spacing"""
        # Calculate new shape
        original_shape = image.shape
        scale_factors = [orig_sp / target_sp for orig_sp, target_sp in zip(original_spacing, target_spacing)]

        new_shape = [int(orig_size * scale_factor) for orig_size, scale_factor in zip(original_shape, scale_factors)]

        # Choose interpolation order
        if is_label:
            order = 0  # Nearest neighbor for labels
        else:
            order = 1 if self.resampling_method == "linear" else 3  # Linear or cubic for images

        # Resample using scipy
        resampled = ndimage.zoom(image, scale_factors, order=order, prefilter=False)

        return resampled

    def _normalize_intensity(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Normalize image intensity"""
        # Clip values
        image = np.clip(image, self.clip_range[0], self.clip_range[1])

        if self.intensity_normalization == "z_score":
            if mask is not None:
                # Compute statistics only within the mask (e.g., body region)
                masked_values = image[mask > 0]
                if len(masked_values) > 0:
                    mean_val = np.mean(masked_values)
                    std_val = np.std(masked_values)
                else:
                    mean_val = np.mean(image)
                    std_val = np.std(image)
            else:
                mean_val = np.mean(image)
                std_val = np.std(image)

            if std_val > 0:
                image = (image - mean_val) / std_val
            else:
                image = image - mean_val

        elif self.intensity_normalization == "min_max":
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)

        elif self.intensity_normalization == "percentile":
            # Use 1st and 99th percentiles for robust normalization
            p1 = np.percentile(image, 1)
            p99 = np.percentile(image, 99)
            image = np.clip(image, p1, p99)
            if p99 > p1:
                image = (image - p1) / (p99 - p1)

        return image.astype(np.float32)

    def _resize_to_target_size(self, image: np.ndarray, is_label: bool = False) -> np.ndarray:
        """Resize image to target size if specified"""
        if self.target_size is None:
            return image

        current_shape = image.shape
        scale_factors = [target / current for target, current in zip(self.target_size, current_shape)]

        order = 0 if is_label else 1
        resized = ndimage.zoom(image, scale_factors, order=order, prefilter=False)

        return resized

    def _create_body_mask(self, image: np.ndarray, threshold: float = -500) -> np.ndarray:
        """Create a simple body mask for more accurate normalization"""
        # Simple threshold-based body mask
        body_mask = image > threshold

        # Remove small connected components
        body_mask = ndimage.binary_opening(body_mask, structure=np.ones((3, 3, 3)))

        # Fill holes
        body_mask = ndimage.binary_fill_holes(body_mask)

        return body_mask.astype(np.uint8)

    def preprocess_case(
        self,
        image_path: Union[str, Path],
        label_path: Optional[Union[str, Path]] = None,
        save_preprocessed: bool = False,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Preprocess a single case (image and optionally label)

        Args:
            image_path: Path to the image file
            label_path: Path to the label file (optional)
            save_preprocessed: Whether to save preprocessed data
            output_dir: Directory to save preprocessed data

        Returns:
            Dictionary containing preprocessed image, label, and metadata
        """
        # Load image
        image, image_metadata = self._load_image(image_path)
        original_spacing = image_metadata['original_spacing']

        # Create body mask for better normalization
        body_mask = self._create_body_mask(image)

        # Normalize intensity
        image_normalized = self._normalize_intensity(image, body_mask)

        # Resample to target spacing
        image_resampled = self._resample_image(
            image_normalized,
            original_spacing,
            self.target_spacing,
            is_label=False
        )

        # Resize to target size if specified
        image_final = self._resize_to_target_size(image_resampled, is_label=False)

        result = {
            'image': image_final,
            'original_image': image,
            'body_mask': self._resample_image(body_mask, original_spacing, self.target_spacing, is_label=True),
            'metadata': {
                **image_metadata,
                'preprocessed_spacing': self.target_spacing,
                'preprocessed_shape': image_final.shape,
                'normalization': self.intensity_normalization,
                'clip_range': self.clip_range
            }
        }

        # Process label if provided
        if label_path is not None:
            label, label_metadata = self._load_image(label_path)

            # Resample label (using nearest neighbor)
            label_resampled = self._resample_image(
                label,
                original_spacing,
                self.target_spacing,
                is_label=True
            )

            # Resize label to target size
            label_final = self._resize_to_target_size(label_resampled, is_label=True)

            result['label'] = label_final.astype(np.uint8)
            result['original_label'] = label
            result['metadata']['label_metadata'] = label_metadata

        # Save preprocessed data if requested
        if save_preprocessed and output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            case_name = Path(image_path).stem.replace('.nii', '')

            # Save image
            image_output_path = output_dir / f"{case_name}_image.npy"
            np.save(image_output_path, image_final)

            # Save label if available
            if 'label' in result:
                label_output_path = output_dir / f"{case_name}_label.npy"
                np.save(label_output_path, result['label'])

            # Save metadata
            metadata_path = output_dir / f"{case_name}_metadata.npy"
            np.save(metadata_path, result['metadata'])

        return result

    def preprocess_dataset(
        self,
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        split: str = "train"
    ) -> Dict[str, List[str]]:
        """
        Preprocess entire dataset

        Args:
            data_dir: Directory containing the dataset
            output_dir: Directory to save preprocessed data
            split: Dataset split ("train", "val", "test")

        Returns:
            Dictionary mapping case IDs to preprocessed file paths
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all image files
        if split == "train":
            image_pattern = "amos_*"
        else:
            image_pattern = f"amos_{split}_*"

        image_files = list(data_dir.glob(f"images{os.sep}{image_pattern}.nii.gz"))
        if not image_files:
            image_files = list(data_dir.glob(f"images{os.sep}{image_pattern}.nii"))

        preprocessed_files = {}

        for image_path in image_files:
            case_id = image_path.stem.replace('.nii', '')

            # Find corresponding label file
            label_path = data_dir / "labels" / f"{case_id}.nii.gz"
            if not label_path.exists():
                label_path = data_dir / "labels" / f"{case_id}.nii"
                if not label_path.exists():
                    label_path = None

            print(f"Preprocessing {case_id}...")

            try:
                # Preprocess the case
                result = self.preprocess_case(
                    image_path=image_path,
                    label_path=label_path,
                    save_preprocessed=True,
                    output_dir=output_dir
                )

                # Store file paths
                preprocessed_files[case_id] = {
                    'image': str(output_dir / f"{case_id}_image.npy"),
                    'label': str(output_dir / f"{case_id}_label.npy") if label_path else None,
                    'metadata': str(output_dir / f"{case_id}_metadata.npy")
                }

            except Exception as e:
                print(f"Error preprocessing {case_id}: {str(e)}")
                continue

        # Save preprocessing summary
        summary_path = output_dir / f"preprocessing_summary_{split}.npy"
        np.save(summary_path, preprocessed_files)

        print(f"Preprocessed {len(preprocessed_files)} cases for {split} split")
        return preprocessed_files

    def get_dataset_statistics(self, preprocessed_files: Dict[str, Dict[str, str]]) -> Dict[str, Union[float, List]]:
        """Compute dataset statistics from preprocessed files"""
        intensities = []
        shapes = []
        spacings = []

        for case_id, file_paths in preprocessed_files.items():
            # Load image and metadata
            image = np.load(file_paths['image'])
            metadata = np.load(file_paths['metadata'], allow_pickle=True).item()

            intensities.extend(image.flatten())
            shapes.append(image.shape)
            spacings.append(metadata['preprocessed_spacing'])

        intensities = np.array(intensities)

        statistics = {
            'mean_intensity': float(np.mean(intensities)),
            'std_intensity': float(np.std(intensities)),
            'min_intensity': float(np.min(intensities)),
            'max_intensity': float(np.max(intensities)),
            'percentile_1': float(np.percentile(intensities, 1)),
            'percentile_99': float(np.percentile(intensities, 99)),
            'median_intensity': float(np.median(intensities)),
            'mean_shape': [float(np.mean([s[i] for s in shapes])) for i in range(3)],
            'std_shape': [float(np.std([s[i] for s in shapes])) for i in range(3)],
            'all_shapes': shapes,
            'target_spacing': spacings[0] if spacings else None,
            'num_cases': len(preprocessed_files)
        }

        return statistics


class AMOSDataset(Dataset):
    """PyTorch Dataset for AMOS22 dataset"""

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = 'train',
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        num_classes: int = 16,
        augmentation: bool = True,
        preprocessed_dir: Optional[Union[str, Path]] = None
    ):
        """
        Args:
            data_root: Root directory of AMOS22 dataset
            split: Dataset split ('train', 'val', 'test')
            patch_size: Size of patches to extract [D, H, W]
            num_classes: Number of segmentation classes
            augmentation: Whether to apply data augmentation
            preprocessed_dir: Directory with preprocessed data (optional)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.augmentation = augmentation and (split == 'train')
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None

        # Find all cases
        self.cases = self._find_cases()

        print(f"Found {len(self.cases)} cases for {split} split")

        # Initialize preprocessor
        self.preprocessor = AMOS22Preprocessor()

        # Initialize augmentation
        if self.augmentation:
            from .augmentation import AMOS22Augmentation
            self.augmentor = AMOS22Augmentation()

    def _find_cases(self) -> List[Dict[str, Path]]:
        """Find all image/label pairs"""
        cases = []

        # Check if using preprocessed data
        if self.preprocessed_dir and self.preprocessed_dir.exists():
            # Load from preprocessed directory
            preprocessed_files = list(self.preprocessed_dir.glob(f"*_image.npy"))
            for img_file in preprocessed_files:
                case_id = img_file.stem.replace('_image', '')
                label_file = self.preprocessed_dir / f"{case_id}_label.npy"

                if label_file.exists():
                    cases.append({
                        'case_id': case_id,
                        'image': img_file,
                        'label': label_file,
                        'preprocessed': True
                    })
        else:
            # Load from raw data with explicit split folder support
            # Prefer split-specific folders if present; otherwise fall back to imagesTr/labelsTr and split later
            explicit_split = False

            split_to_dirs = {
                'train': ('imagesTr', 'labelsTr'),
                'val': ('imagesVal', 'labelsVal'),
                'test': ('imagesTs', 'labelsTs'),
            }

            if self.split in split_to_dirs:
                img_dir_name, lbl_dir_name = split_to_dirs[self.split]
                images_dir = self.data_root / img_dir_name
                labels_dir = self.data_root / lbl_dir_name

                if images_dir.exists():
                    explicit_split = True
                else:
                    # Fall back to combined folders
                    images_dir = self.data_root / 'imagesTr'
                    labels_dir = self.data_root / 'labelsTr'

            else:
                images_dir = self.data_root / 'imagesTr'
                labels_dir = self.data_root / 'labelsTr'

            # Alternative generic paths fallback
            if not images_dir.exists():
                images_dir = self.data_root / 'images'
            if not labels_dir.exists():
                labels_dir = self.data_root / 'labels'

            if not images_dir.exists():
                raise FileNotFoundError(f"Images directory not found at {self.data_root}")

            # Find all image files
            image_files = sorted(list(images_dir.glob('*.nii.gz')) + list(images_dir.glob('*.nii')))

            for img_file in image_files:
                case_id = img_file.stem.replace('.nii', '')

                # Find corresponding label (may be missing for test)
                label_file = labels_dir / f"{case_id}.nii.gz"
                if not label_file.exists():
                    label_file = labels_dir / f"{case_id}.nii"

                if self.split == 'test' and not label_file.exists():
                    # Allow unlabeled test data
                    cases.append({
                        'case_id': case_id,
                        'image': img_file,
                        'label': None,
                        'preprocessed': False
                    })
                elif label_file.exists():
                    cases.append({
                        'case_id': case_id,
                        'image': img_file,
                        'label': label_file,
                        'preprocessed': False
                    })

        # Split data only if using combined folders and no explicit split dirs are present
        if not self.preprocessed_dir:
            # Determine if we used explicit split dirs; if not, perform 80/20 split
            # Heuristic: if split requested is 'train' or 'val' and imagesVal exists, we already partitioned
            images_val_dir = self.data_root / 'imagesVal'
            used_explicit = images_val_dir.exists() or (self.data_root / 'imagesTs').exists()

            if not used_explicit:
                if self.split == 'train':
                    cases = cases[:int(0.8 * len(cases))]
                elif self.split == 'val':
                    cases = cases[int(0.8 * len(cases)):]

        return cases

    def _load_case(self, case_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess a single case"""
        if case_info['preprocessed']:
            # Load preprocessed data
            image = np.load(case_info['image'])
            label = np.load(case_info['label'])
        else:
            # Load and preprocess raw data
            result = self.preprocessor.preprocess_case(
                image_path=case_info['image'],
                label_path=case_info['label']
            )
            image = result['image']
            label = result.get('label', None)

        if label is None:
            # For test split without labels, return a dummy label of zeros for compatibility
            label = np.zeros_like(image, dtype=np.uint8)

        return image, label

    def _extract_patch(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a random patch from the volume"""
        # Get volume shape
        d, h, w = image.shape
        pd, ph, pw = self.patch_size

        # If volume is smaller than patch size, pad it
        if d < pd or h < ph or w < pw:
            pad_d = max(0, pd - d)
            pad_h = max(0, ph - h)
            pad_w = max(0, pw - w)

            image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            label = np.pad(label, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

            d, h, w = image.shape

        # Random crop
        d_start = np.random.randint(0, max(1, d - pd + 1))
        h_start = np.random.randint(0, max(1, h - ph + 1))
        w_start = np.random.randint(0, max(1, w - pw + 1))

        image_patch = image[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        label_patch = label[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]

        return image_patch, label_patch

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample"""
        case_info = self.cases[idx]

        # Load case
        image, label = self._load_case(case_info)

        # Extract patch
        image_patch, label_patch = self._extract_patch(image, label)

        # Apply augmentation
        if self.augmentation and hasattr(self, 'augmentor'):
            # Convert to torch tensors for augmentation
            image_tensor_aug = torch.from_numpy(image_patch).float()
            label_tensor_aug = torch.from_numpy(label_patch).long()

            image_tensor_aug, label_tensor_aug = self.augmentor.apply_augmentation(image_tensor_aug, label_tensor_aug)

            # Convert back to numpy
            image_patch = image_tensor_aug.numpy()
            label_patch = label_tensor_aug.numpy()

        # Convert to tensors
        image_tensor = torch.from_numpy(image_patch).unsqueeze(0).float()  # Add channel dimension
        label_tensor = torch.from_numpy(label_patch).long()

        return {
            'image': image_tensor,
            'label': label_tensor,
            'case_id': case_info['case_id']
        }
