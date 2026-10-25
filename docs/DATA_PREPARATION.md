# Image inputs and feature extraction

This repository contains code only. Images, labels, class metadata, cached embeddings and model weights are external inputs. Install the repository and its optional OpenAI CLIP dependency as described in the main README. The scripts require Python 3.10 or newer.

Use the pinned OpenAI CLIP revision in `requirements-clip.txt`: `c13005fd422b20dcd11774e9fff46370887218e4`. Its encoder and tokenizer match the evaluated source. Later CLIP revisions change the ResNet attention-pooling computation and can change FP16 rounding, even when the equations are equivalent. Updating this dependency requires a new numerical check.

The raw-image pipeline has two steps:

1. Prepare an ordered manifest and a separate evaluation-label file.
2. Encode the images and text without opening the label file.

The paper uses a single center crop for the ten cross-domain datasets and ImageNet. Its four ImageNet shifts use 64 views and the six views with the lowest prediction entropy. Feature extraction is offline. The adaptation algorithm subsequently processes each cached image representation once in sequence.

## Generic ImageFolder input

Create an external JSON file with explicit class order and prompt templates:

```json
{
  "class_names": ["cat", "dog"],
  "class_folders": ["cat_images", "dog_images"],
  "templates": ["a photo of a {}."]
}
```

The example class names and template only illustrate the format. They are not the paper's benchmark prompts. Labels are zero-based positions in `class_names`. If `class_folders` is omitted, each class name is also its folder name. Images are sorted by relative path inside each folder. The folders must be inside the declared root.

```text
/external/images/
  cat_images/
    image001.jpg
  dog_images/
    image002.jpg
```

Run from the repository root after `pip install -e .`:

```bash
python scripts/prepare_manifest.py --image-root /external/images --classes /external/classes.json --output /external/manifests/example --shuffle-seed 1
```

Outputs:

- `paths.txt`: ordered image paths relative to the image root.
- `classes.json`: ordered class names and prompt templates.
- `labels.npy`: corresponding integer labels, for offline scoring only.
- `order.npy`: permutation of the source rows.
- `metadata.json`: image root, class count, order configuration and file hashes.

The same commands work in PowerShell with Windows paths, for example `--image-root "D:/datasets/example"`. Use quotes when a path contains spaces.

Without an order option, source order is preserved. Use exactly one of `--shuffle-seed`, `--order-file` or `--path-order`. An index order file must be a full permutation, stored as `.npy`, a JSON list or whitespace-separated integers. A path order file must contain every source image once, with absolute paths or paths relative to the image root. This last option can reproduce an existing archived order.

## Optional external PTA dataset adapter

If an external PTA checkout and its prepared benchmark folders are already available, the adapter can use that checkout's dataset classes to build manifests. No PTA checkout, benchmark images or split files are copied into this repository. The external loader may need its own dependencies such as `gdown`.

```bash
python scripts/prepare_manifest.py --pta-root /external/PTA --data-root /external/data --dataset oxford_flowers --shuffle-seed 1 --output /external/manifests/oxford_flowers
```

Accepted dataset keys:

```text
oxford_flowers dtd oxford_pets stanford_cars ucf101 caltech101
food101 sun397 fgvc eurosat
imagenet imagenet-a imagenet-v imagenet-r imagenet-s
```

Use the upstream dataset instructions at <https://github.com/hzhxmu/PTA/blob/main/docs/DATASETS.md> for the external folder layout. Ten-domain loaders use their prescribed test splits. ImageNet uses its validation images. ImageNet-A and ImageNet-R use their respective 200-class subsets, so their text banks and labels must use the same subset order. The adapter stores the exact class names and templates returned by the external loader.

Loading a different repository revision can change prompts, file enumeration or filtering. For the paper's exact experiment, use the same split membership, class mapping, prompts, input representation and stored test order. A generic sorted ImageFolder scan is not a substitute for the prescribed benchmark test split. A newly generated `--shuffle-seed 1` order is identical to an old order only when the underlying source enumeration also matches.

## Single-view extraction

Supply a local OpenAI CLIP checkpoint to avoid automatic downloads:

```bash
python scripts/extract_features.py --manifest /external/manifests/oxford_flowers --checkpoint /external/weights/ViT-B-16.pt --protocol single --output /external/cache/vit/oxford_flowers --batch-size 64 --workers 4
```

For ResNet-50, use the corresponding local checkpoint and a separate cache directory:

```bash
python scripts/extract_features.py --manifest /external/manifests/oxford_flowers --checkpoint /external/weights/RN50.pt --protocol single --output /external/cache/resnet50/oxford_flowers --batch-size 64 --workers 4
```

Alternatively, explicit `--model "ViT-B/16"` or `--model RN50` permits the OpenAI CLIP package to download that model. `--download-root /external/weights` controls its cache. No dataset download is performed.

Single-view output contains `features.npy` with shape `[images, dimension]` and `text_weights.npy` with shape `[classes, dimension]`, both stored as float16. Dimensions are 512 for ViT-B/16 and 1024 for ResNet-50. The center-crop preprocessing is returned by the checkpoint's CLIP loader. Every image vector is L2 normalized. Each prompt vector is normalized before averaging, and each mean class vector is normalized again. No label array is read by this command.

## Multiview extraction

Prepare a manifest for each ImageNet shift using its exact class mapping and prompts. Then run:

```bash
python scripts/extract_features.py --manifest /external/manifests/imagenet-a --checkpoint /external/weights/ViT-B-16.pt --protocol multiview --output /external/cache/vit/imagenet-a --workers 4
```

The extractor uses one center crop plus 63 AugMix views, computes normalized view embeddings and logits, selects the six views with the lowest entropy, and averages their features and logits separately. The mean feature is **not normalized again**. The per-image random state is `1 + stored row index`. A multiview forward call processes all 64 views of one image. `--batch-size` applies only to single-view extraction.

In addition to the feature and text files, this mode writes:

- `logits.npy`: the actual selected-view mean logits, shape `[images, classes]`.
- `selected_indices.npy`: the six selected view indices for each image.

Pass `logits.npy` to cached replay. Recomputing the logits from the averaged feature can change FP16 results because averaging and matrix multiplication do not have identical rounding order.

## Existing paths or text embeddings

The manifest directory is a convenience. Direct inputs are also supported:

```bash
python scripts/extract_features.py --paths /external/paths.txt --image-root /external/images --classes /external/classes.json --checkpoint /external/weights/ViT-B-16.pt --protocol single --output /external/cache/example
```

To reuse an existing class text bank, replace `--classes` with `--text /external/text_weights.npy`. It must contain finite float16 values with the correct class order and encoder dimension. Existing absolute paths need no `--image-root`. When a manifest is moved to another machine, `--image-root` overrides the old root stored in metadata.

## Resume and numerical settings

Outputs include configuration, progress and a completion receipt. Use `--resume` only for an incomplete extraction with identical input hashes, model, encoding protocol and library versions. A completed extraction is protected from overwriting. After an interrupted multiview run, at most the rows since the last checkpoint are recomputed with the same per-row random state.

CUDA is the validated precision path. TF32 is disabled. The normal CUDA OpenAI CLIP path encodes in float16. Keep the environment versions in the repository's requirements when reproducing the reported experiments. A different GPU, kernel or library version can still cause small numerical differences.

`--device cpu` supports functional checks with CLIP float32 computation. Its saved arrays are converted to float16 for the file interface, but **CPU extraction does not reproduce the paper's CUDA numerics**. It must not be presented as the validated benchmark configuration.

## Point-cloud scope

The cached-feature adaptation interface can consume external point-cloud/text embeddings with compatible dimensions. This code release does not include a validated pipeline from raw point clouds to the paper's point-cloud results. The existing third-party PTA point-cloud runner is not an implementation of FGGM. It is therefore not repackaged as if it reproduced those results.

## Third-party augmentation

`third_party/augmix_ops.py` retains Google's Apache-2.0 header. Its license is included in `third_party/licenses/Apache-2.0-AugMix.txt`. OpenAI CLIP is an external dependency with its own license. Model weights and datasets retain their separate upstream terms.
