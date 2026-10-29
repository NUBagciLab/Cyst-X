# Cyst-X preprocessing

These scripts are from Cyst-X-v2.3 and prepare RAS-oriented volumes, N4-corrected images, and cropped classification ROIs from aligned image/mask pairs. The segmentation data in the Hugging Face `main` / `v2` release already has matching image/mask orientations, so this pipeline starts at step 1. Install the dependencies in your active Python environment:

```bash
pip install -r preprocessing/requirements.txt
```

Paths below are relative to the repository root. Each segmentation dataset directory has `t1/images`, `t1/masks`, `t2/images`, and `t2/masks`. The published Hugging Face dataset can be downloaded under `Cyst-X/`; its `classification/` folder already contains N4-based crops, so preprocessing is only needed to regenerate data or process new inputs.

## Image orientation

The classification dataset is released in **RAS voxel orientation** (Right, Anterior, Superior). The full-volume segmentation dataset retains its original acquisition orientation, with each mask aligned to its corresponding image; it has not been standardized to RAS. To convert the segmentation images and masks to RAS, use [step1_reorientation_to_RAS.py](https://github.com/NUBagciLab/Cyst-X/blob/main/preprocessing/step1_reorientation_to_RAS.py).

Here, RAS follows the increasing-voxel-axis convention used by [NiBabel](https://nipy.org/nibabel/image_orientation.html) and SimpleITK's orientation helpers. The supplied script uses MONAI `Orientationd(axcodes="RAS")` and writes NIfTI files with NiBabel. For a classification image, `nib.aff2axcodes(nib.load(path).affine)` returns `('R', 'A', 'S')`; the corresponding SimpleITK check is `sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(sitk.ReadImage(path).GetDirection())`. **ITK-SNAP may label the same RAS-oriented volume as LPI**, because its orientation-code convention names the opposite ends of the axes. This is a difference in labeling, not an image/mask misalignment; see the [ITK orientation conventions](https://docs.itk.org/projects/doxygen/en/latest/classitk_1_1AnatomicalOrientation.html).

## Processing steps

| Script | Input | Default output |
|---|---|---|
| `step1_reorientation_to_RAS.py` | `Cyst-X/segmentation` | `Cyst-X-RAS/segmentation` |
| `step2_n4biascorrection.py` | `Cyst-X-RAS/segmentation` | `Cyst-X-n4/segmentation` |
| `step3_crop.py` | N4 images and RAS masks in separate directories | `Cyst-X-n4/classification` |

Starting with the published segmentation folder, run the complete pipeline from the repository root:

```bash
bash preprocessing/run_preprocessing.sh
```

The default input is `Cyst-X/segmentation`. To use another directory of already aligned image/mask pairs, run `bash preprocessing/run_preprocessing.sh /path/to/aligned/segmentation`. Outputs still use the default directories listed above.

Alternatively, run each step separately:

```bash
python preprocessing/step1_reorientation_to_RAS.py --input ./Cyst-X/segmentation --output ./Cyst-X-RAS/segmentation
python preprocessing/step2_n4biascorrection.py --dataset ./Cyst-X-RAS/segmentation --output ./Cyst-X-n4/segmentation
python preprocessing/step3_crop.py --input ./Cyst-X-n4/segmentation --mask-input ./Cyst-X-RAS/segmentation --output ./Cyst-X-n4/classification
```

Each of these three scripts accepts `-t 1` or `-t 2` to process a single modality. Without `-t`, both are processed. Use `--help` for worker counts and visualization options. N4 processing can take considerable CPU time.

Keep classification metadata with the regenerated crops. For a dataset downloaded into `Cyst-X/`, copy the label workbooks and split files:

```python
from pathlib import Path
from shutil import copy2
source = Path("Cyst-X/classification")
target = Path("Cyst-X-n4/classification")
target.mkdir(parents=True, exist_ok=True)
for pattern in ("*.xlsx", "*_split.csv"):
    for path in source.glob(pattern):
        copy2(path, target / path.name)
```

`run_preprocessing.sh` chains steps 1–3 and stops if any step fails. `step2_crop.py` is retained as an alternative cropping/visualization implementation; it is not the N4 correction script.

## Predicted masks

To crop with another segmentation model, orient its masks consistently with the images first, then supply that RAS mask directory to `step3_crop.py --mask-input`. The image input remains the N4-corrected segmentation image directory. For example, `run_seg_crop.sh PanSegNet 1` uses `seg/PanSegNet-RAS/segmentation` and saves the T1 crops under `seg/PanSegNet-n4/classification`. The wrapper accepts `PanSegNet` or `Swin` and modality `1` or `2`; activate your Python environment before running it.
