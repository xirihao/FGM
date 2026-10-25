"""Prepare ordered image paths and evaluation-only labels without copying images."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np

DATASETS = (
    'oxford_flowers', 'dtd', 'oxford_pets', 'stanford_cars', 'ucf101',
    'caltech101', 'food101', 'sun397', 'fgvc', 'eurosat',
    'imagenet', 'imagenet-a', 'imagenet-v', 'imagenet-r', 'imagenet-s',
)
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_classes(spec: dict) -> None:
    names, templates = spec.get('class_names'), spec.get('templates')
    if not isinstance(names, list) or not names or not all(isinstance(x, str) and x.strip() for x in names):
        raise ValueError('classes.json needs a nonempty class_names list of strings.')
    if not isinstance(templates, list) or not templates or not all(isinstance(x, str) for x in templates):
        raise ValueError('classes.json needs a nonempty templates list of strings.')
    for template in templates:
        try:
            if template.format('example class') == template:
                raise ValueError('A template must contain a class placeholder, e.g. {}.')
        except (IndexError, KeyError, ValueError) as error:
            raise ValueError(f'Invalid prompt template: {template!r}') from error


def imagefolder_rows(root: Path, classes_path: Path):
    spec = json.loads(classes_path.read_text(encoding='utf-8-sig'))
    validate_classes(spec)
    folders = spec.get('class_folders', spec['class_names'])
    if not isinstance(folders, list) or len(folders) != len(spec['class_names']):
        raise ValueError('class_folders must have one directory name per class_names entry.')
    if len(set(folders)) != len(folders):
        raise ValueError('class_folders must be unique.')
    paths, labels = [], []
    for index, folder in enumerate(folders):
        if not isinstance(folder, str):
            raise ValueError('class_folders must contain strings.')
        directory = (root / folder).resolve()
        if not directory.is_relative_to(root) or not directory.is_dir():
            raise ValueError(f'Invalid or missing class directory: {folder}')
        images = sorted(
            (p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
             and not any(part.startswith('.') for part in p.relative_to(directory).parts)),
            key=lambda p: p.relative_to(root).as_posix(),
        )
        if not images:
            raise ValueError(f'No images found for class {index}: {folder}')
        paths.extend(images)
        labels.extend([index] * len(images))
    return paths, labels, spec, {'classes_input_sha256': sha256(classes_path)}


def pta_rows(pta_root: Path, data_root: Path, dataset: str):
    if not (pta_root / 'datasets' / '__init__.py').is_file():
        raise FileNotFoundError('--pta-root must be an external PTA checkout with datasets/__init__.py.')
    sys.path.insert(0, str(pta_root))
    module = importlib.import_module('datasets')
    if not Path(module.__file__).resolve().is_relative_to(pta_root):
        raise RuntimeError('Another package named datasets shadowed the requested PTA loader.')
    if dataset == 'imagenet':
        cls = importlib.import_module('datasets.imagenet').ImageNet
        data = cls(str(data_root), None)
        rows = [(Path(path).resolve(), int(label)) for path, label in data.test.samples]
    else:
        data = module.build_dataset(dataset, str(data_root))
        rows = [(Path(item.impath).resolve(), int(item.label)) for item in data.test]
    names = list(data.classnames)
    spec = {'class_names': names, 'templates': list(data.template)}
    validate_classes(spec)
    if set(label for _, label in rows) != set(range(len(names))):
        raise ValueError('PTA class labels are not contiguous or do not cover the text bank.')
    source_hashes = {
        p.relative_to(pta_root).as_posix(): sha256(p)
        for p in sorted((pta_root / 'datasets').glob('*.py'))
    }
    return [p for p, _ in rows], [y for _, y in rows], spec, {'external_loader_hashes': source_hashes}


def path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def select_order(args, paths: list[Path], root: Path) -> np.ndarray:
    n = len(paths)
    if args.order_file:
        if args.order_file.suffix.lower() == '.npy':
            order = np.load(args.order_file, allow_pickle=False)
        else:
            content = args.order_file.read_text(encoding='utf-8-sig')
            if args.order_file.suffix.lower() == '.json':
                order = np.asarray(json.loads(content))
            else:
                order = np.asarray([int(value) for value in content.split()], dtype=np.int64)
        if order.ndim != 1 or not np.issubdtype(order.dtype, np.integer):
            raise ValueError('--order-file must be a one-dimensional integer permutation.')
        order = order.astype(np.int64)
    elif args.path_order:
        lookup = {path_key(path): i for i, path in enumerate(paths)}
        ordered = args.path_order.read_text(encoding='utf-8-sig').splitlines()
        try:
            order = np.asarray([lookup[path_key(Path(p) if Path(p).is_absolute() else root / p)] for p in ordered], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f'--path-order references an image outside this dataset: {error}') from error
    elif args.shuffle_seed is not None:
        order = np.random.default_rng(args.shuffle_seed).permutation(n)
    else:
        order = np.arange(n, dtype=np.int64)
    if len(order) != n or not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError('The selected order must contain every source row exactly once.')
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, help='Root containing class directories for generic ImageFolder input.')
    parser.add_argument('--classes', type=Path, help='Explicit class_names, templates and optional class_folders JSON.')
    parser.add_argument('--pta-root', type=Path, help='Optional external PTA checkout. Its code is not redistributed.')
    parser.add_argument('--data-root', type=Path, help='Dataset root used by the external PTA loader.')
    parser.add_argument('--dataset', choices=DATASETS)
    order = parser.add_mutually_exclusive_group()
    order.add_argument('--shuffle-seed', type=int, help='Permute source rows with NumPy default_rng(seed).')
    order.add_argument('--order-file', type=Path, help='Full source-row permutation as .npy, JSON list or whitespace integers.')
    order.add_argument('--path-order', type=Path, help='Exact desired path list, absolute or relative to the image/data root.')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory is not empty. Choose a new directory to preserve previous manifests.')
    if args.pta_root:
        if not args.data_root or not args.dataset or args.image_root or args.classes:
            parser.error('--pta-root requires --data-root and --dataset; do not combine it with --image-root/--classes.')
        root = args.data_root.resolve()
        paths, labels, spec, provenance = pta_rows(args.pta_root.resolve(), root, args.dataset)
        mode = 'external_pta_loader'
    else:
        if not args.image_root or not args.classes or args.dataset or args.data_root:
            parser.error('Generic mode requires --image-root and --classes; PTA-specific arguments require --pta-root.')
        root = args.image_root.resolve()
        paths, labels, spec, provenance = imagefolder_rows(root, args.classes)
        mode = 'imagefolder'
    if not paths or len({path_key(p) for p in paths}) != len(paths):
        raise ValueError('Input image list must be nonempty and contain no duplicate resolved paths.')
    for path in paths:
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f'Missing image or path outside the declared root: {path}')
    indices = select_order(args, paths, root)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'paths.txt').write_text(''.join(paths[i].relative_to(root).as_posix() + '\n' for i in indices), encoding='utf-8')
    np.save(args.output / 'labels.npy', np.asarray(labels, dtype=np.int64)[indices])
    np.save(args.output / 'order.npy', indices)
    (args.output / 'classes.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    metadata = {
        'schema_version': 1, 'mode': mode, 'dataset': args.dataset,
        'image_root': str(root), 'images': len(paths), 'classes': len(spec['class_names']),
        'ordering': 'explicit_paths' if args.path_order else 'explicit_indices' if args.order_file else 'shuffle' if args.shuffle_seed is not None else 'source_order',
        'shuffle_seed': args.shuffle_seed, 'labels_for_evaluation_only': True,
        'hashes': {name: sha256(args.output / name) for name in ('paths.txt', 'classes.json', 'labels.npy', 'order.npy')},
        **provenance,
    }
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'Manifest saved to {args.output.resolve()} ({len(paths)} images, {len(spec["class_names"])} classes).')


if __name__ == '__main__':
    main()
