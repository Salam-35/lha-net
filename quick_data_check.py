#!/usr/bin/env python3
"""
Quick 1-minute data check
Fast diagnostic to identify obvious problems
"""

import yaml
import nibabel as nib
from pathlib import Path

def quick_check():
    """Quick data check - runs in ~1 minute"""

    print("="*80)
    print("QUICK DATA CHECK")
    print("="*80)

    # Load config
    with open('configs/lha_net_config.yaml') as f:
        config = yaml.safe_load(f)

    data_root = Path(config['paths']['data_root'])
    print(f"\nData root: {data_root}")

    # Check if paths exist
    print("\n1. Checking paths...")
    imagesTr = data_root / 'imagesTr'
    labelsTr = data_root / 'labelsTr'

    if not data_root.exists():
        print(f"   ❌ Data root does not exist!")
        return

    if not imagesTr.exists():
        print(f"   ❌ imagesTr/ not found!")
        return

    if not labelsTr.exists():
        print(f"   ❌ labelsTr/ not found!")
        return

    print(f"   ✓ Paths exist")

    # Count files
    print("\n2. Counting files...")
    images = list(imagesTr.glob('*.nii.gz')) + list(imagesTr.glob('*.nii'))
    labels = list(labelsTr.glob('*.nii.gz')) + list(labelsTr.glob('*.nii'))

    print(f"   Images: {len(images)}")
    print(f"   Labels: {len(labels)}")

    if len(images) == 0:
        print(f"   ❌ No images found!")
        return

    if len(labels) == 0:
        print(f"   ❌ No labels found!")
        return

    # Check first image
    print("\n3. Checking first image...")
    first_img = sorted(images)[0]
    print(f"   File: {first_img.name}")

    img_nii = nib.load(str(first_img))
    img_data = img_nii.get_fdata()

    print(f"   Shape: {img_data.shape}")
    print(f"   Range: [{img_data.min():.1f}, {img_data.max():.1f}]")

    # Detect modality
    if img_data.min() < -500:
        modality = "CT"
        print(f"   ✓ Modality: CT")
    else:
        modality = "MRI"
        print(f"   ⚠️  Modality: MRI")

    # Check preprocessing config
    clip_min, clip_max = config['data']['preprocessing']['clip_range']
    print(f"\n4. Checking preprocessing config...")
    print(f"   Clip range: [{clip_min}, {clip_max}]")

    if modality == "MRI" and (img_data.min() > clip_max or img_data.max() < clip_min):
        print(f"   ❌ CRITICAL: MRI data will be DESTROYED by CT clipping!")
        print(f"   Your data is MRI but config is for CT!")
        return

    # Check first label
    print("\n5. Checking first label...")
    case_id = first_img.stem.replace('.nii', '')
    first_lbl = labelsTr / f"{case_id}.nii.gz"
    if not first_lbl.exists():
        first_lbl = labelsTr / f"{case_id}.nii"

    if not first_lbl.exists():
        print(f"   ❌ Label not found for {case_id}")
        return

    lbl_data = nib.load(str(first_lbl)).get_fdata()
    print(f"   Shape: {lbl_data.shape}")
    print(f"   Classes: {sorted([int(x) for x in set(lbl_data.flatten())])}")

    if lbl_data.shape != img_data.shape:
        print(f"   ❌ Shape mismatch with image!")
        return

    num_organs = len(set(lbl_data.flatten())) - 1
    print(f"   Organs: {num_organs}")

    if num_organs == 0:
        print(f"   ❌ No organs in label!")
        return

    print(f"   ✓ Label looks OK")

    # Quick modality scan
    print("\n6. Scanning modalities (first 10 cases)...")
    ct_count = 0
    mri_count = 0

    for img_file in sorted(images)[:10]:
        img = nib.load(str(img_file)).get_fdata()
        if img.min() < -500:
            ct_count += 1
        else:
            mri_count += 1

    print(f"   CT: {ct_count}, MRI: {mri_count}")

    if ct_count > 0 and mri_count > 0:
        print(f"   ⚠️  MIXED CT and MRI data!")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if modality == "MRI" and clip_min < 0:
        print("\n❌ CRITICAL ISSUE: MRI data with CT preprocessing!")
        print("   ACTION: Use CT-only data or fix preprocessing config")
    elif ct_count > 0 and mri_count > 0:
        print("\n⚠️  WARNING: Mixed CT and MRI data!")
        print("   ACTION: Filter to single modality or use separate preprocessing")
    else:
        print("\n✅ Data looks OK for training")
        print("   You can proceed with training")

if __name__ == '__main__':
    try:
        quick_check()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
