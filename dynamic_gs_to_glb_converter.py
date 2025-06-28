import argparse
import glob
import os
import numpy as np
from plyfile import PlyData
import pygltflib

def convert_dynamic_gaussian_to_glb(args):
    """
    Converts a sequence of Gaussian PLY files into a single animated point cloud GLB.
    """
    # 1. Find and sort the list of PLY files in the input directory.
    ply_files = sorted(glob.glob(os.path.join(args.input_dir, '*.ply')))
    if not ply_files:
        print(f"Error: No PLY files found in the directory '{args.input_dir}'.")
        return

    # If the num_frames argument is given, trim the file list.
    if args.num_frames is not None and args.num_frames > 0:
        if len(ply_files) > args.num_frames:
            ply_files = ply_files[:args.num_frames]
            print(f"Processing a maximum of {args.num_frames} frames as specified.")
        else:
            print(f"Warning: The requested number of frames ({args.num_frames}) is greater than or equal to the actual number of files ({len(ply_files)}). Processing all files.")
    
    total_frames = len(ply_files)
    print(f"Starting conversion with a total of {total_frames} frames.")

    # --- Initialize GLTF structure ---
    gltf = pygltflib.GLTF2()
    binary_blob = bytearray()
    scene_nodes = []
    animation = pygltflib.Animation()
    time_points = np.linspace(0, total_frames / args.fps, total_frames + 1, dtype=np.float32)

    # --- 2. Loop to process each PLY file as a frame ---
    for i, ply_path in enumerate(ply_files):
        print(f"Processing... (Frame {i+1}/{total_frames}) {os.path.basename(ply_path)}")
        try:
            with open(ply_path, 'rb') as f:
                plydata = PlyData.read(f)
            vertex_element = plydata['vertex']
            vertex_count = vertex_element.count

            # Extract data
            positions = np.vstack([vertex_element[p] for p in args.pos_props]).T.astype(np.float32)
            colors_dc = np.vstack([vertex_element[p] for p in args.color_props]).T.astype(np.float32)
            opacities = vertex_element[args.opacity_prop].astype(np.float32)
            
            # Coordinate system conversion (Z-up -> Y-up)
            original_y = positions[:, 1].copy()
            positions[:, 1] = positions[:, 2].copy()
            positions[:, 2] = -original_y

            # Final color calculation
            colors = 1.0 / (1.0 + np.exp(-colors_dc))
            alpha = 1.0 / (1.0 + np.exp(-opacities))
            final_colors_float = colors * alpha[:, np.newaxis]
            final_colors_uint8 = (final_colors_float.clip(0, 1) * 255).astype(np.uint8)

            # --- Create GLTF data (repeated for each frame) ---
            pos_offset = len(binary_blob)
            binary_blob.extend(positions.tobytes())
            color_offset = len(binary_blob)
            binary_blob.extend(final_colors_uint8.tobytes())
            
            # BufferViews
            pos_buffer_view_idx = len(gltf.bufferViews)
            gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=pos_offset, byteLength=positions.nbytes))
            color_buffer_view_idx = len(gltf.bufferViews)
            gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=color_offset, byteLength=final_colors_uint8.nbytes))

            # Accessors
            pos_accessor_idx = len(gltf.accessors)
            gltf.accessors.append(pygltflib.Accessor(bufferView=pos_buffer_view_idx, componentType=pygltflib.FLOAT, count=vertex_count, type=pygltflib.VEC3, min=np.min(positions, axis=0).tolist(), max=np.max(positions, axis=0).tolist()))
            color_accessor_idx = len(gltf.accessors)
            gltf.accessors.append(pygltflib.Accessor(bufferView=color_buffer_view_idx, componentType=pygltflib.UNSIGNED_BYTE, count=vertex_count, type=pygltflib.VEC3, normalized=True))

            # Mesh, Primitive, Node
            primitive = pygltflib.Primitive(attributes=pygltflib.Attributes(POSITION=pos_accessor_idx, COLOR_0=color_accessor_idx), mode=pygltflib.POINTS)
            mesh_idx = len(gltf.meshes)
            gltf.meshes.append(pygltflib.Mesh(primitives=[primitive]))
            node_idx = len(gltf.nodes)
            initial_scale = [1.0, 1.0, 1.0] if i == 0 else [0.0, 0.0, 0.0]
            gltf.nodes.append(pygltflib.Node(mesh=mesh_idx, scale=initial_scale))
            scene_nodes.append(node_idx)

            # --- Create animation channel ---
            key_times = [time_points[i], time_points[i+1]] if i == 0 else [time_points[i-1], time_points[i], time_points[i+1]]
            key_scales = [[1,1,1], [0,0,0]] if i == 0 else [[0,0,0], [1,1,1], [0,0,0]]
            
            time_data = np.array(key_times, dtype=np.float32).tobytes()
            scale_data = np.array(key_scales, dtype=np.float32).tobytes()

            time_accessor_idx = len(gltf.accessors)
            # ★★★ Key fix: Convert min/max values to standard Python float ★★★
            gltf.accessors.append(pygltflib.Accessor(bufferView=len(gltf.bufferViews), componentType=pygltflib.FLOAT, count=len(key_times), type=pygltflib.SCALAR, min=[float(min(key_times))], max=[float(max(key_times))]))
            gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=len(binary_blob), byteLength=len(time_data))); binary_blob.extend(time_data)
            
            scale_accessor_idx = len(gltf.accessors)
            gltf.accessors.append(pygltflib.Accessor(bufferView=len(gltf.bufferViews), componentType=pygltflib.FLOAT, count=len(key_scales), type=pygltflib.VEC3))
            gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=len(binary_blob), byteLength=len(scale_data))); binary_blob.extend(scale_data)

            sampler_idx = len(animation.samplers)
            animation.samplers.append(pygltflib.AnimationSampler(input=time_accessor_idx, output=scale_accessor_idx, interpolation="STEP"))
            animation.channels.append(pygltflib.AnimationChannel(sampler=sampler_idx, target=pygltflib.AnimationChannelTarget(node=node_idx, path="scale")))

        except Exception as e:
            print(f"Error: Exception occurred while processing '{ply_path}'. Skipping. ({e})")
            continue
    
    # --- 3. Save the final GLB file ---
    gltf.scene = 0
    gltf.scenes.append(pygltflib.Scene(nodes=scene_nodes))
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_blob)))
    if animation.channels:
        gltf.animations.append(animation)
    
    print("Converting and saving to GLB file...")
    gltf.set_binary_blob(binary_blob)
    output_dir = os.path.dirname(args.output_glb)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    gltf.save(args.output_glb)
    print(f"Success! Animated GLB file '{os.path.basename(args.output_glb)}' has been saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts a dynamic Gaussian PLY sequence into a single animated GLB file.")
    
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing sequential .ply files.")
    parser.add_-argument("output_glb", type=str, help="Path for the output .glb file.")
    parser.add_argument("--num_frames", type=int, default=None, help="Maximum number of frames to process. If not specified, all files in the folder are processed.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the animation (default: 30).")
    parser.add_argument("--pos_props", nargs=3, default=['x', 'y', 'z'], help="Property names for position (x,y,z).")
    parser.add_argument("--color_props", nargs=3, default=['f_dc_0', 'f_dc_1', 'f_dc_2'], help="Property names for DC color (r,g,b).")
    parser.add_argument("--opacity_prop", type=str, default='opacity', help="Property name for opacity.")

    args = parser.parse_args()
    
    convert_dynamic_gaussian_to_glb(args)
