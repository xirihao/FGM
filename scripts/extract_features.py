"""Extract frozen CLIP inputs from an ordered path manifest. Never opens labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch
import torchvision


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


class ImageRows:
    def __init__(self, paths, transform, protocol, offset=0):
        self.paths, self.transform, self.protocol, self.offset = paths, transform, protocol, offset

    def __len__(self):
        return len(self.paths) - self.offset

    def __getitem__(self, index):
        row = index + self.offset
        if self.protocol == 'multiview':
            random.seed(1 + row)
            np.random.seed(1 + row)
            torch.manual_seed(1 + row)
        with Image.open(self.paths[row]) as image:
            result = self.transform(image.convert('RGB'))
        return torch.stack(result) if self.protocol == 'multiview' else result


def encode_text(model, clip, classes_file: Path, device):
    spec = json.loads(classes_file.read_text(encoding='utf-8-sig'))
    names, templates = spec.get('class_names'), spec.get('templates')
    if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
        raise ValueError('classes.json requires a nonempty class_names list.')
    if not isinstance(templates, list) or not templates or not all(isinstance(x, str) for x in templates):
        raise ValueError('classes.json requires a nonempty templates list.')
    rows = []
    for name in names:
        prompts = [template.format(name.replace('_', ' ')) for template in templates]
        vectors = model.encode_text(clip.tokenize(prompts).to(device))
        vectors /= vectors.norm(dim=-1, keepdim=True)
        mean = vectors.mean(dim=0)
        mean /= mean.norm()
        rows.append(mean)
    return torch.stack(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, help='Directory containing paths.txt, classes.json and optional metadata.json.')
    parser.add_argument('--paths', type=Path, help='Override the ordered image path list.')
    parser.add_argument('--classes', type=Path, help='Override class_names/templates JSON for text encoding.')
    parser.add_argument('--text', type=Path, help='Use an existing FP16 [classes, dimension] text bank instead of encoding prompts.')
    parser.add_argument('--image-root', type=Path, help='Root for relative paths; overrides the manifest metadata image_root.')
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument('--checkpoint', type=Path, help='Local OpenAI CLIP checkpoint. No download is performed for this option.')
    weights.add_argument('--model', help='Explicit OpenAI CLIP model name, e.g. ViT-B/16 or RN50. May download its weights.')
    parser.add_argument('--download-root', type=Path, help='External cache for weights requested with --model.')
    parser.add_argument('--protocol', choices=('single', 'multiview'), default='single')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64, help='Encoding batch size for single-view inputs only.')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda', help='CUDA is required for paper precision. CPU is for functional checks only.')
    parser.add_argument('--resume', action='store_true', help='Resume a matching incomplete extraction.')
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error('--batch-size must be positive and --workers must be nonnegative.')
    paths_file = args.paths or (args.manifest / 'paths.txt' if args.manifest else None)
    classes_file = args.classes or (args.manifest / 'classes.json' if args.manifest else None)
    if paths_file is None or (args.text is None and classes_file is None):
        parser.error('Provide --manifest, or --paths together with --classes or --text.')
    image_root = args.image_root
    if image_root is None and args.manifest and (args.manifest / 'metadata.json').exists():
        # Metadata contains the image root. Evaluation label arrays are never opened.
        metadata = json.loads((args.manifest / 'metadata.json').read_text(encoding='utf-8-sig'))
        image_root = Path(metadata['image_root']) if metadata.get('image_root') else None
    path_strings = paths_file.read_text(encoding='utf-8-sig').splitlines()
    if not path_strings or any(not entry.strip() for entry in path_strings):
        raise ValueError('The ordered paths file must be nonempty and contain no blank rows.')
    paths = []
    for entry in path_strings:
        path = Path(entry)
        if not path.is_absolute():
            if image_root is None:
                raise ValueError('Relative paths require --image-root or a manifest metadata image_root.')
            path = image_root / path
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path.resolve())
    if args.checkpoint and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError('Output is not empty. Use a new output directory, or --resume for an incomplete extraction.')
    if args.resume and not (args.output / 'config.json').is_file():
        raise FileNotFoundError('--resume requires output/config.json from an incomplete extraction.')
    if (args.output / 'receipt.json').exists():
        raise FileExistsError('Extraction is already complete. Use a new output directory.')
    try:
        import clip
    except ImportError as error:
        raise ImportError('Install the external OpenAI CLIP dependency listed in requirements-clip.txt.') from error
    if not hasattr(clip, 'load') or not hasattr(clip, 'tokenize'):
        raise ImportError('The installed clip module is not OpenAI CLIP. Follow requirements-clip.txt.')
    if args.model and args.model not in clip.available_models():
        raise ValueError(f'Unknown CLIP model {args.model!r}. Available models: {clip.available_models()}')
    device = torch.device(args.device)
    if device.type not in ('cuda', 'cpu'):
        raise ValueError('This extractor supports CUDA and CPU only.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install the correct PyTorch build, or use CPU only for functional checks.')
    if device.type == 'cpu':
        print('CPU FUNCTIONAL CHECK: CLIP runs in float32; these caches do not reproduce the paper FP16 CUDA numerics.', flush=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model, preprocess = clip.load(str(args.checkpoint.resolve()) if args.checkpoint else args.model,
                                  device=device, jit=False,
                                  download_root=str(args.download_root) if args.download_root else None)
    model.eval()
    if args.protocol == 'multiview' and int(model.visual.input_resolution) != 224:
        raise ValueError('The evaluated multiview protocol is defined for 224-pixel CLIP encoders.')
    dimension = int(model.text_projection.shape[1])
    with torch.inference_mode():
        if args.text:
            text_array = np.load(args.text, allow_pickle=False)
            if text_array.dtype != np.float16 or text_array.ndim != 2:
                raise ValueError('--text must be an FP16 array of shape [classes, feature_dimension].')
            text = torch.from_numpy(text_array.copy()).to(device=device, dtype=model.dtype)
        else:
            text = encode_text(model, clip, classes_file, device)
    if text.shape[1] != dimension or not torch.isfinite(text).all() or (text.norm(dim=1) == 0).any():
        raise ValueError('Invalid text bank or checkpoint/text dimension mismatch.')
    n, classes = len(paths), len(text)
    identity = {
        'schema_version': 1, 'paths_sha256': sha256(paths_file),
        'classes_sha256': sha256(classes_file) if not args.text else None,
        'text_input_sha256': sha256(args.text) if args.text else None,
        'checkpoint_sha256': sha256(args.checkpoint) if args.checkpoint else None,
        'model_name': args.model, 'protocol': args.protocol, 'images': n, 'classes': classes,
        'dimension': dimension, 'device_type': device.type, 'encoder_dtype': str(model.dtype),
        'output_dtype': 'float16', 'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'torchvision_version': torchvision.__version__, 'pillow_version': pillow_version,
        'clip_sources': {p.name: sha256(p) for p in sorted(Path(clip.__file__).parent.glob('*.py'))},
        'extractor_sha256': sha256(Path(__file__)),
        'augmentation_sha256': sha256(Path(__file__).resolve().parents[1] / 'fgm' / 'augmentation.py') if args.protocol == 'multiview' else None,
        'augmentation_ops_sha256': sha256(Path(__file__).resolve().parents[1] / 'third_party' / 'augmix_ops.py') if args.protocol == 'multiview' else None,
        'single_view_batch_size': args.batch_size if args.protocol == 'single' else None,
        'views': 64 if args.protocol == 'multiview' else 1,
        'selected': 6 if args.protocol == 'multiview' else 1,
        'augmentation_seed': '1 + stored row index' if args.protocol == 'multiview' else None,
        'aggregate_feature_renormalized': False, 'tf32': False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    config_file, progress_file = args.output / 'config.json', args.output / 'progress.json'
    if args.resume:
        if json.loads(config_file.read_text(encoding='utf-8')) != identity:
            raise ValueError('Resume configuration/input hashes differ. Use a new output directory.')
        offset = json.loads(progress_file.read_text(encoding='utf-8'))['processed'] if progress_file.exists() else 0
    else:
        write_json(config_file, identity)
        offset = 0
    if not 0 <= offset <= n:
        raise ValueError('Invalid extraction progress.')
    saved_text = text.detach().cpu().numpy().astype(np.float16)
    text_path = args.output / 'text_weights.npy'
    if args.resume and text_path.exists():
        if not np.array_equal(np.load(text_path, allow_pickle=False), saved_text):
            raise ValueError('Resume text embeddings differ from the saved text bank.')
    else:
        np.save(text_path, saved_text)
    mode = 'r+' if offset else 'w+'
    features = np.lib.format.open_memmap(args.output / 'features.npy', mode=mode, dtype=np.float16, shape=(n, dimension))
    logits = selected = None
    output_names = ['features.npy', 'text_weights.npy']
    if args.protocol == 'multiview':
        from fgm.augmentation import get_ood_preprocess
        preprocess = get_ood_preprocess()
        logits = np.lib.format.open_memmap(args.output / 'logits.npy', mode=mode, dtype=np.float16, shape=(n, classes))
        selected = np.lib.format.open_memmap(args.output / 'selected_indices.npy', mode=mode, dtype=np.int16, shape=(n, 6))
        output_names.extend(['logits.npy', 'selected_indices.npy'])
    dataset = ImageRows(paths, preprocess, args.protocol, offset)
    options = dict(batch_size=1 if args.protocol == 'multiview' else args.batch_size,
                   num_workers=args.workers, pin_memory=device.type == 'cuda', shuffle=False)
    if args.workers:
        options['prefetch_factor'] = 2
    loader = torch.utils.data.DataLoader(dataset, **options)
    position = offset
    with torch.inference_mode():
        for images in loader:
            if args.protocol == 'multiview':
                images = images[0]
            images = images.to(device, non_blocking=device.type == 'cuda')
            vectors = model.encode_image(images)
            vectors /= vectors.norm(dim=-1, keepdim=True)
            if not torch.isfinite(vectors).all():
                raise FloatingPointError(f'Nonfinite CLIP image features at row {position}.')
            if args.protocol == 'multiview':
                scores = 100. * vectors @ text.T
                if not torch.isfinite(scores).all():
                    raise FloatingPointError(f'Nonfinite CLIP logits at row {position}.')
                entropy = -(scores.softmax(1) * scores.log_softmax(1)).sum(1)
                ids = torch.argsort(entropy, descending=False)[:6]
                features[position] = vectors[ids].mean(0).cpu().numpy()
                logits[position] = scores[ids].mean(0).cpu().numpy()
                selected[position] = ids.cpu().numpy()
                position += 1
            else:
                count = len(vectors)
                features[position:position + count] = vectors.cpu().numpy()
                position += count
            if args.protocol == 'single' or position % 128 == 0 or position == n:
                features.flush()
                if logits is not None:
                    logits.flush()
                    selected.flush()
                write_json(progress_file, {'processed': position, 'total': n})
                print(f'{position}/{n}', flush=True)
    if position != n:
        raise RuntimeError('Extraction ended before all rows were processed.')
    features.flush()
    if logits is not None:
        logits.flush()
        selected.flush()
    write_json(args.output / 'receipt.json', {
        'status': 'COMPLETE', 'n': n, 'labels_opened': False,
        'paper_precision': device.type == 'cuda' and model.dtype == torch.float16,
        'config': identity, 'hashes': {name: sha256(args.output / name) for name in output_names},
    })
    print(f'Frozen inputs saved to {args.output.resolve()}. No label array was opened.', flush=True)


if __name__ == '__main__':
    main()
