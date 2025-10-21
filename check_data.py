#!/usr/bin/env python3
"""
Diagnostic script to check AMOS22 data quality
Checks for CT/MRI mixing, data validity, and preprocessing issues
"""

import yaml
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import sys

def check_amos_data(config_path='configs/lha_net_config.yaml'):
    """Check AMOS22 dataset for common issues"""

    # Load config
    print("Loading configuration...")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_root = Path(config['paths']['data_root'])
    print(f"Data root: {data_root}")

    if not data_root.exists():
        print(f"❌ ERROR: Data root does not exist: {data_root}")
        return

    print("="*80)
    print("AMOS22 DATA DIAGNOSTIC")
    print("="*80)

    # Find all images
    image_dirs = ['imagesTr', 'imagesVa', 'imagesTs']
    label_dirs = ['labelsTr', 'labelsVa', 'labelsTs']

    all_issues = []
    total_ct = 0
    total_mri = 0

    for img_dir, lbl_dir in zip(image_dirs, label_dirs):
        img_path = data_root / img_dir
        lbl_path = data_root / lbl_dir

        if not img_path.exists():
            print(f"\n⚠️  {img_dir}/ not found, skipping...")
            continue

        print(f"\n{'='*80}")
        print(f"Checking {img_dir}/")
        print(f"{'='*80}")

        image_files = sorted(img_path.glob('*.nii.gz'))
        if len(image_files) == 0:
            image_files = sorted(img_path.glob('*.nii'))

        print(f"Found {len(image_files)} images")

        if len(image_files) == 0:
            print(f"❌ No images found in {img_path}")
            all_issues.append(f"No images in {img_dir}/")
            continue

        # Check first 5 images in detail
        print(f"\nDetailed check of first 5 cases:")
        print("-"*80)

        for img_file in image_files[:5]:
            case_id = img_file.stem.replace('.nii', '')

            # Find label file
            lbl_file = lbl_path / f"{case_id}.nii.gz"
            if not lbl_file.exists():
                lbl_file = lbl_path / f"{case_id}.nii"

            try:
                # Load image
                img_nii = nib.load(str(img_file))
                img_data = img_nii.get_fdata()

                # Load label if exists
                if lbl_file.exists():
                    lbl_nii = nib.load(str(lbl_file))
                    lbl_data = lbl_nii.get_fdata()
                else:
                    lbl_data = None

                # Analyze
                print(f"\n📄 Case: {case_id}")
                print(f"   Image shape: {img_data.shape}")
                print(f"   Image range: [{img_data.min():.1f}, {img_data.max():.1f}]")
                print(f"   Image mean: {img_data.mean():.1f}, std: {img_data.std():.1f}")

                # Detect modality (CRITICAL!)
                if img_data.min() < -500 or (img_data.min() < 0 and img_data.max() > 1000):
                    modality = "CT"
                    print(f"   ✓ Detected modality: CT (Hounsfield Units)")
                else:
                    modality = "MRI"
                    print(f"   ⚠️  Detected modality: MRI (arbitrary units)")

                # Check if preprocessing will destroy this data
                clip_min, clip_max = config['data']['preprocessing']['clip_range']
                if modality == "MRI":
                    if img_data.min() > clip_max or img_data.max() < clip_min:
                        issue = f"❌ {case_id}: MRI data will be DESTROYED by CT clipping [{clip_min}, {clip_max}]"
                        print(f"   {issue}")
                        all_issues.append(issue)

                # Check labels
                if lbl_data is not None:
                    print(f"   Label shape: {lbl_data.shape}")
                    unique_labels = np.unique(lbl_data)
                    print(f"   Label classes: {[int(x) for x in unique_labels]}")
                    print(f"   Number of organs: {len(unique_labels) - 1}")

                    # Check shape match
                    if lbl_data.shape != img_data.shape:
                        issue = f"❌ {case_id}: Shape mismatch! Image={img_data.shape}, Label={lbl_data.shape}"
                        print(f"   {issue}")
                        all_issues.append(issue)

                    # Check background ratio
                    bg_ratio = (lbl_data == 0).sum() / lbl_data.size
                    print(f"   Background ratio: {bg_ratio:.1%}")

                    if bg_ratio > 0.98:
                        issue = f"⚠️  {case_id}: Mostly background ({bg_ratio:.1%})"
                        all_issues.append(issue)

                    if len(unique_labels) == 1:
                        issue = f"❌ {case_id}: Only background, no organs!"
                        print(f"   {issue}")
                        all_issues.append(issue)
                else:
                    print(f"   ⚠️  No label file found at {lbl_file}")
                    all_issues.append(f"Missing label for {case_id}")

            except Exception as e:
                issue = f"❌ {case_id}: Error loading - {e}"
                print(f"   {issue}")
                all_issues.append(issue)

        # Quick modality count for ALL files
        print(f"\n{'='*80}")
        print(f"Scanning all {len(image_files)} cases for modality...")
        print(f"{'='*80}")

        ct_cases = []
        mri_cases = []

        for img_file in tqdm(image_files, desc="Checking modalities"):
            try:
                img_data = nib.load(str(img_file)).get_fdata()
                case_id = img_file.stem.replace('.nii', '')

                if img_data.min() < -500 or (img_data.min() < 0 and img_data.max() > 1000):
                    ct_cases.append(case_id)
                else:
                    mri_cases.append(case_id)
            except Exception as e:
                print(f"\n⚠️  Error loading {img_file.name}: {e}")

        total_ct += len(ct_cases)
        total_mri += len(mri_cases)

        print(f"\n📊 {img_dir}/ Summary:")
        print(f"   CT cases: {len(ct_cases)}")
        print(f"   MRI cases: {len(mri_cases)}")

        if len(ct_cases) > 0 and len(mri_cases) > 0:
            issue = f"⚠️  MIXED MODALITIES in {img_dir}: {len(ct_cases)} CT + {len(mri_cases)} MRI"
            print(f"\n   {issue}")
            all_issues.append(issue)

            print(f"\n   First 5 CT cases: {ct_cases[:5]}")
            print(f"   First 5 MRI cases: {mri_cases[:5]}")

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal CT cases: {total_ct}")
    print(f"Total MRI cases: {total_mri}")

    if total_ct > 0 and total_mri > 0:
        print(f"\n⚠️  CRITICAL: You have MIXED CT and MRI data!")
        print(f"   Your current config clips to {config['data']['preprocessing']['clip_range']}")
        print(f"   This will DESTROY all MRI scans!")
        print(f"\n   RECOMMENDATION: Filter to CT-only or use separate preprocessing")

    if all_issues:
        print(f"\n❌ Found {len(all_issues)} issues:")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("\n✅ No critical issues found!")

    print("\n" + "="*80)

    return {
        'ct_count': total_ct,
        'mri_count': total_mri,
        'issues': all_issues
    }

if __name__ == '__main__':
    try:
        results = check_amos_data()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
