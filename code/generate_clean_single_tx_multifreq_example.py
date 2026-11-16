#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, json, time
from pathlib import Path
import numpy as np
from sionna.rt import PlanarArray, Transmitter, load_scene


def generate(xml_path, x, y, h, tilt, freq_hz, samples):
    scene = load_scene(str(xml_path))
    scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, pattern='iso', polarization='V')
    scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, pattern='dipole', polarization='cross')
    scene.add(Transmitter('tx', [x, y, h], [np.deg2rad(tilt), 0, 0]))
    scene.frequency = float(freq_hz)
    scene.synthetic_array = True
    cm = scene.coverage_map(max_depth=5, diffraction=True, cm_cell_size=(4.0, 4.0), cm_center=[0,0,1.5], cm_orientation=[0,0,0], cm_size=[1024,1024], num_samples=int(samples))
    arr = 10.0 * np.log10(cm.path_gain.numpy()[0,:,:] + 1e-12)
    del scene, cm
    gc.collect()
    return arr.astype(np.float32)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--xml', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--x', type=float, default=-472.0)
    ap.add_argument('--y', type=float, default=-4.0)
    ap.add_argument('--h', type=int, default=50)
    ap.add_argument('--tilt', type=int, default=0)
    ap.add_argument('--samples', type=int, default=int(2e7))
    ap.add_argument('--freqs-ghz', default='2.4,3.65,4.9,5.8,24,28,39')
    args=ap.parse_args()
    args.xml=args.xml.resolve()
    args.out_dir=args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rec=[]
    for token in args.freqs_ghz.split(','):
        fg=float(token.strip())
        out=args.out_dir/f'txx_{args.x:.1f}_y_{args.y:.1f}_h{args.h}_f{fg:.2f}G_tilt{args.tilt}_with_building.npy'
        t0=time.time()
        print('generate', out.name, 'samples', args.samples, flush=True)
        arr=generate(args.xml,args.x,args.y,args.h,args.tilt,fg*1e9,args.samples)
        np.save(out, arr)
        rec.append({'freq_ghz':fg,'path':str(out),'seconds':time.time()-t0,'min':float(np.nanmin(arr)),'max':float(np.nanmax(arr)),'mean':float(np.nanmean(arr))})
        (args.out_dir/'manifest.json').write_text(json.dumps({'xml':str(args.xml),'samples':args.samples,'tx':[args.x,args.y,args.h],'tilt':args.tilt,'records':rec},indent=2))
    print('done', args.out_dir)
if __name__=='__main__':
    main()
