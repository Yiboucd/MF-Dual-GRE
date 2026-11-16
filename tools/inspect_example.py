#!/usr/bin/env python3
"""Read one released example without PyTorch, a GPU, or Sionna."""
import argparse
import json
from pathlib import Path

import numpy as np

FREQUENCIES_GHZ = (2.4, 3.65, 4.9, 5.8, 24.0, 28.0, 39.0)
REPO = Path(__file__).resolve().parents[1]


def resolve(root, value):
    path = Path(value)
    if path.is_absolute():
        raise ValueError('Released paths must be relative: ' + str(path))
    target = (root / path).resolve()
    if root.resolve() not in target.parents:
        raise ValueError('Path escapes dataset root')
    return target


def load_example(root, manifest_path):
    root = Path(root).resolve()
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records = {round(float(item['freq_ghz']), 6): item for item in manifest['records']}
    if set(records) != set(FREQUENCIES_GHZ):
        raise ValueError('Expected exactly the seven released frequencies')
    arrays = []
    for frequency in FREQUENCIES_GHZ:
        array = np.load(resolve(root, records[frequency]['path']), allow_pickle=False)
        if array.shape != (256, 256) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError('Invalid raw radio map at {} GHz'.format(frequency))
        arrays.append(array)
    building = None
    coverage = resolve(root, manifest['source_coverage_dir'])
    for filename in ('B_FOOTPRINT_GEOM_1024m_4m.npy', 'B_MAP_GEOM_1024m_4m.npy'):
        if (coverage / filename).exists():
            building = np.load(coverage / filename, allow_pickle=False).astype(bool)
            break
    if building is None or building.shape != (256, 256):
        raise ValueError('Missing or invalid source building mask')
    return np.stack(arrays, axis=-1), building, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO / 'data/dataset')
    parser.add_argument('--manifest', help='Manifest path relative to the dataset root')
    parser.add_argument('--all', action='store_true', help='Validate all extracted examples')
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.manifest:
        paths = [args.root / args.manifest]
    else:
        paths = sorted(args.root.glob('*/xml_*/tx_*/manifest.json'))
        if not args.all:
            paths = paths[:1]
    if not paths:
        raise SystemExit('No examples found. Run tools/download_data.py first.')
    first = None
    for path in paths:
        maps, building, metadata = load_example(args.root, path)
        if first is None:
            first = {'manifest': str(path.relative_to(args.root)), 'maps_shape': list(maps.shape),
                     'maps_dtype': str(maps.dtype), 'frequencies_ghz': FREQUENCIES_GHZ,
                     'path_gain_db_range': [float(maps.min()), float(maps.max())],
                     'valid_free_space_pixels': int((~building).sum())}
    print(json.dumps({'validated_examples': len(paths), 'first_example': first}, indent=2))


if __name__ == '__main__':
    main()
