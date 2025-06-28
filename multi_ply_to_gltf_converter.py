#!/usr/bin/env python3
import argparse
import os
import sys

import trimesh

def main():
    parser = argparse.ArgumentParser(
        description='Combine multiple PLY files into one glTF (GLB/GLTF) file.')
    parser.add_argument(
        'inputs',
        nargs='+',
        help='Input PLY file paths.')
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output file path (must end with .glb or .gltf).')
    args = parser.parse_args()

    # 1) Check for input file existence and load
    meshes = []
    for path in args.inputs:
        if not os.path.isfile(path):
            print(f'Error: File not found: {path}', file=sys.stderr)
            sys.exit(1)
        try:
            mesh = trimesh.load(path, force='mesh')
            meshes.append(mesh)
        except Exception as e:
            print(f'Error loading {path}: {e}', file=sys.stderr)
            sys.exit(1)

    # 2) Add meshes to the Scene
    scene = trimesh.Scene()
    for mesh in meshes:
        scene.add_geometry(mesh)

    # 3) Validate output filename and extension
    out_path = args.output
    ext = os.path.splitext(out_path)[1].lower()
    if ext not in ['.glb', '.gltf']:
        print('Error: Output extension must be .glb or .gltf', file=sys.stderr)
        sys.exit(1)

    # 4) Export
    try:
        scene.export(out_path)
        print(f'Successfully exported combined scene to {out_path}')
    except Exception as e:
        print(f'Error exporting to {out_path}: {e}', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
