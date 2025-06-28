#!/usr/bin/env python3
import argparse
import os
import sys

import trimesh

def main():
    parser = argparse.ArgumentParser(
        description='Combine multiple PLY files into a single binary glTF (.glb) file.')
    parser.add_argument(
        'inputs',
        nargs='+',
        help='Input PLY file paths.')
    parser.add_argument(
        '-o', '--output',
        default='combined.glb',
        help='Output .glb file path (must end with .glb). Default: combined.glb')
    args = parser.parse_args()

    # 1) Check output file extension
    out_path = args.output
    if not out_path.lower().endswith('.glb'):
        print('Error: Output extension must be .glb', file=sys.stderr)
        sys.exit(1)

    # 2) Load each PLY file
    meshes = []
    for ply in args.inputs:
        if not os.path.isfile(ply):
            print(f'Error: File not found: {ply}', file=sys.stderr)
            sys.exit(1)
        try:
            mesh = trimesh.load(ply, force='mesh')
            meshes.append(mesh)
        except Exception as e:
            print(f'Error loading {ply}: {e}', file=sys.stderr)
            sys.exit(1)

    # 3) Add meshes to the Scene
    scene = trimesh.Scene()
    for mesh in meshes:
        scene.add_geometry(mesh)

    # 4) Export to .glb (embed all in one binary file)
    try:
        # Method A: Automatically determine file format
        scene.export(out_path)
        # Or Method B: Explicitly specify file_type
        # with open(out_path, 'wb') as f:
        #     f.write(scene.export(file_type='glb'))
        print(f'Successfully exported to {out_path}')
    except Exception as e:
        print(f'Error exporting to {out_path}: {e}', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
