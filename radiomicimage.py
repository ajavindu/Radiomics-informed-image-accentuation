#!/usr/bin/env python
import os
import argparse
import SimpleITK as sitk
from radiomics import featureextractor

def resample_mask_to_image(mask, image):
    """
    Resamples the mask image so that its geometry (origin, spacing, direction) 
    matches that of the provided image. Nearest neighbor interpolation is used.
    
    Args:
        mask (SimpleITK.Image): The input mask image.
        image (SimpleITK.Image): The reference image whose geometry will be used.
        
    Returns:
        SimpleITK.Image: The resampled mask image.
    """
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(image)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetDefaultPixelValue(0)
    return resample.Execute(mask)

def extract_voxelwise_features_no_crop(mri_path, mask_path, output_dir):
    """
    Compute voxel-wise GLCM and GLDM features for the given MRI volume and tumor mask.
    The output feature maps will have the same size and geometry as the MRI (no cropping).
    Each feature is saved as a separate 3D NIfTI in the specified output directory.

    Args:
        mri_path (str): Path to the 3D MRI NIfTI.
        mask_path (str): Path to the corresponding tumor mask NIfTI.
        output_dir (str): Directory where feature maps will be saved.
    """
    # 1) Load NIfTI images with SimpleITK
    if not os.path.isfile(mri_path):
        raise FileNotFoundError(f"Could not find MRI at {mri_path}")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Could not find Mask at {mask_path}")

    mri_image = sitk.ReadImage(mri_path)
    mask_image = sitk.ReadImage(mask_path)

    # 2) Check if images share the same geometry, if not, resample the mask to match the MRI
    if (mri_image.GetSize() != mask_image.GetSize() or 
        mri_image.GetOrigin() != mask_image.GetOrigin() or 
        mri_image.GetDirection() != mask_image.GetDirection()):
        print("[WARNING] MRI and Mask geometry differ. Resampling mask to match MRI...")
        mask_image = resample_mask_to_image(mask_image, mri_image)

    # 3) Create a parameter dictionary for GLCM & GLDM in 3D
    voxel_params = {
        'imageType': {
            'Original': {}
        },
        'featureClass': {
            'glcm': [],  # enable default GLCM features
            'gldm': []   # enable default GLDM features
        },
        'setting': {
            'force2D': False,   # Full 3D extraction
            'binWidth': 8,      # example: custom bin width
        }
    }

    # 4) Instantiate a RadiomicsFeatureExtractor
    voxel_extractor = featureextractor.RadiomicsFeatureExtractor(voxel_params)
    # Optionally reduce verbosity:
    # voxel_extractor.logger.setLevel("WARNING")

    # 5) Perform voxel-based extraction. Passing voxelBased=True returns a dict of 3D feature maps.
    voxel_result = voxel_extractor.execute(mri_image, mask_image, voxelBased=True)

    # 6) Create the output directory if not present
    os.makedirs(output_dir, exist_ok=True)

    # 7) Save each feature map as a 3D .nii.gz file
    print(f"\nSaving voxel-wise features to '{output_dir}':")
    for key, sitk_img in voxel_result.items():
        if key.startswith('diagnostics_'):
            continue  # skip non-feature items

        feature_name = key  # e.g., 'original_glcm_Contrast'
        out_path = os.path.join(output_dir, f"{feature_name}.nii.gz")

        sitk.WriteImage(sitk_img, out_path)
        print(f"  - Saved: {feature_name} -> {out_path}")

    print("\nDone! All voxel-wise features have been saved as separate 3D NIfTIs.")
    print("They retain the same size, spacing, and orientation as the original MRI.")

def main():
    parser = argparse.ArgumentParser(
        description="Extract voxel-wise radiomics features from an MRI and corresponding mask using GLCM and GLDM."
    )
    parser.add_argument("-m", "--mri", required=True, help="Path to the MRI NIfTI file")
    parser.add_argument("-k", "--mask", required=True, help="Path to the mask NIfTI file")
    parser.add_argument("-o", "--output", required=True, help="Directory to save the output feature maps")
    
    args = parser.parse_args()
    extract_voxelwise_features_no_crop(args.mri, args.mask, args.output)

if __name__ == "__main__":
    main()
