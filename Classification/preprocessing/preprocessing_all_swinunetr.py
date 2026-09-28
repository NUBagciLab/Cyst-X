# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 20:15:00 2026

@author: pky0507
"""

from preprocessing import preprocess_image
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_file(args_tuple):
    fname, modality, masks_dict, args = args_tuple
    image_path = os.path.join(args.image, modality, 'images', fname)
    output_path = os.path.join(args.output, modality, fname)
    if fname not in masks_dict:
        fname = fname.replace('.nii.gz', '_1.nii.gz')
    if fname in masks_dict:   
        mask_path = masks_dict[fname]
        # print(f"Preprcess: {fname}")      
        preprocess_image(
            image_path=image_path,
            mask_path=mask_path,
            output_path=output_path,
            target_spacing=tuple(args.spacing)
        )
    else:
        print(fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess image and mask pairs.")
    
    
    parser.add_argument('-i', '--image', default="/dataset/IPMN_images_masks/", type=str, help="Input image folder (.nii.gz)")
    parser.add_argument('-m', '--mask', default="../../Segmentation/Swin-UNETR", type=str, help="Input image folder (.nii.gz)")
    parser.add_argument('-o', '--output', default="./Swin-UNETR", type=str, help="Output preprocessed image (.nii.gz)")
    parser.add_argument('--spacing', nargs=3, type=float, default=[1.0, 1.0, 1.0],
                        help="Target spacing in mm (default: 1.0 1.0 1.0)")
    
    args = parser.parse_args()
    
    
    for modality in ['t1', 't2']:
        
        os.makedirs(os.path.join(args.output, modality), exist_ok=True)
        images = [f for f in os.listdir(os.path.join(args.image, modality, 'images')) if f.endswith(".nii.gz")]
        masks = []
        for fold in range(5):
            masks +=  [os.path.join(args.mask, 'saved', modality, 'fold'+str(fold), 'output', f) for f in os.listdir(os.path.join(args.mask, 'saved', modality, 'fold'+str(fold), 'output')) if f.endswith(".nii.gz")]
        masks_dict = {os.path.basename(p): p for p in masks}
        with Pool(processes=cpu_count()) as pool:
            list(tqdm(pool.imap_unordered(process_file, [(fname, modality, masks_dict, args) for fname in images]), total=len(images)))

        # for fname in os.listdir(os.path.join(args.image, modality, 'images')):            
        #     if fname.endswith('.nii.gz'):
        #         image_path = os.path.join(args.image, modality, 'images', fname)
        #         mask_path = os.path.join(args.image, modality, 'masks', fname)
        #         output_path = os.path.join(args.output, modality, fname)
        #         print(f"Preprcess: {fname}")
                
        #         preprocess_image(
        #             image_path=image_path,
        #             mask_path=mask_path,
        #             output_path=output_path,
        #             target_spacing=tuple(args.spacing)
        #         )