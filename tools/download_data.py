#!/usr/bin/env python3
"""Download published data archives, verify SHA-256, and extract them."""
import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def extract(archive, root):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        for member in members:
            parts = PurePosixPath(member.name)
            if parts.is_absolute() or '..' in parts.parts or '\\' in member.name:
                raise ValueError('Unsafe archive path: ' + member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError('Archive links/devices are not accepted: ' + member.name)
            target = (root / member.name).resolve()
            if root != target and root not in target.parents:
                raise ValueError('Archive path escapes extraction root: ' + member.name)
        bundle.extractall(root, members=members)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset', choices=('sample', 'full'), default='sample')
    parser.add_argument('--cities', nargs='+', choices=('Davis', 'Woodland', 'Vacaville'),
                        default=['Davis', 'Woodland', 'Vacaville'])
    parser.add_argument('--root', type=Path, default=REPO / 'data/dataset')
    parser.add_argument('--archives-dir', type=Path, default=REPO / 'data/downloads')
    parser.add_argument('--manifest', type=Path, default=REPO / 'data/artifacts.json')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text(encoding='utf-8'))['artifacts']
    selected = [r for r in records if (r['kind'] == 'sample' if args.subset == 'sample'
                else r['kind'] == 'geometry' or r.get('city') in args.cities)]
    if not selected:
        raise ValueError('No artifacts selected')
    args.archives_dir.mkdir(parents=True, exist_ok=True)
    for record in selected:
        path = args.archives_dir / record['name']
        verified = path.exists() and path.stat().st_size == record['bytes'] and sha256(path) == record['sha256']
        if not verified:
            if args.verify_only:
                raise ValueError('Missing or invalid local archive: ' + str(path))
            print('Downloading ' + record['name'], flush=True)
            partial = path.with_suffix(path.suffix + '.part')
            request = urllib.request.Request(record['url'], headers={'User-Agent': 'MF-Dual-GRE-data'})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            if partial.stat().st_size != record['bytes'] or sha256(partial) != record['sha256']:
                raise ValueError('Size/SHA-256 mismatch: ' + str(partial))
            partial.replace(path)
        print('Verified ' + record['name'], flush=True)
        if not args.download_only and not args.verify_only:
            extract(path, args.root)
    print('Archives: ' + str(args.archives_dir.resolve()))
    if not args.download_only and not args.verify_only:
        print('Dataset root: ' + str(args.root.resolve()))


if __name__ == '__main__':
    main()
