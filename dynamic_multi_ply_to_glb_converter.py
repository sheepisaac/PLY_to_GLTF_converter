import argparse
import glob
import os
import trimesh
import numpy as np
import pygltflib
from plyfile import PlyData # ★ Import library for loading PLY files ★

def combine_dynamic_sequences(dir1, dir2, output_glb, num_frames, fps=30, gap=0.1):
    """
    Merges two dynamic PLY sequences into a single animated GLB file 
    by placing them side-by-side along the Z-axis.

    :param dir1: Path to the first PLY sequence folder.
    :param dir2: Path to the second PLY sequence folder.
    :param output_glb: Path for the output .glb file.
    :param num_frames: Number of frames to process from each sequence.
    :param fps: Frames per second for the animation.
    :param gap: Gap between the two sequences.
    """
    # 1. Get the file lists from each directory and trim them to num_frames.
    ply_files1 = sorted(glob.glob(os.path.join(dir1, '*.ply')))[:num_frames]
    ply_files2 = sorted(glob.glob(os.path.join(dir2, '*.ply')))[:num_frames]

    if not ply_files1 or not ply_files2:
        print("Error: Could not find PLY files in one or more directories, or the specified number of frames is insufficient.")
        return
    if len(ply_files1) != len(ply_files2):
        print("Warning: The two sequences have a different number of frames. Proceeding based on the shorter sequence.")
        min_len = min(len(ply_files1), len(ply_files2))
        ply_files1 = ply_files1[:min_len]
        ply_files2 = ply_files2[:min_len]
    
    total_frames = len(ply_files1)
    print(f"Merging the two sequences with a total of {total_frames} frames.")

    # 2. Calculate the overall bounding box of the first sequence to determine the Z-axis translation for the second sequence.
    print("Calculating the overall bounding box for the first sequence...")
    global_min_z, global_max_z = float('inf'), float('-inf')
    for ply_path in ply_files1:
        try:
            pcd = trimesh.load(ply_path)
            min_bound, max_bound = pcd.bounds
            global_min_z = min(global_min_z, min_bound[2])
            global_max_z = max(global_max_z, max_bound[2])
        except Exception as e:
            print(f"Warning: Error processing file '{ply_path}': {e}")
            continue
    
    z_translation = (global_max_z - global_min_z) + gap
    translation_vector = np.array([0, 0, z_translation], dtype=np.float32)
    print(f"Calculation complete. The second sequence will be translated by {z_translation:.2f} along the Z-axis.")

    # 3. Create the GLB file structure.
    gltf = pygltflib.GLTF2()
    binary_blob = bytearray()
    scene_nodes = []
    animation = pygltflib.Animation()
    time_points = np.linspace(0, total_frames / fps, total_frames + 1, dtype=np.float32)

    for i in range(total_frames):
        print(f"Processing... (Frame {i+1}/{total_frames})")
        
        for seq_idx, ply_path in enumerate([ply_files1[i], ply_files2[i]]):
            try:
                # ★★★ Bug Fix: Changed to PlyData ★★★
                ply_data = PlyData.read(ply_path)
                vertices = ply_data['vertex']
                vertex_count = vertices.count
                positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T.astype(np.float32)
                colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T.astype(np.uint8)

                if seq_idx == 1:
                    positions += translation_vector

                pos_offset = len(binary_blob)
                binary_blob.extend(positions.tobytes())
                color_offset = len(binary_blob)
                binary_blob.extend(colors.tobytes())

                pos_buffer_view_idx = len(gltf.bufferViews)
                gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=pos_offset, byteLength=len(positions.tobytes()), target=pygltflib.ARRAY_BUFFER))
                color_buffer_view_idx = len(gltf.bufferViews)
                gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=color_offset, byteLength=len(colors.tobytes()), target=pygltflib.ARRAY_BUFFER))

                pos_accessor_idx = len(gltf.accessors)
                gltf.accessors.append(pygltflib.Accessor(bufferView=pos_buffer_view_idx, componentType=pygltflib.FLOAT, count=vertex_count, type=pygltflib.VEC3, min=np.min(positions, axis=0).tolist(), max=np.max(positions, axis=0).tolist()))
                color_accessor_idx = len(gltf.accessors)
                gltf.accessors.append(pygltflib.Accessor(bufferView=color_buffer_view_idx, componentType=pygltflib.UNSIGNED_BYTE, count=vertex_count, type=pygltflib.VEC3, normalized=True))
                
                primitive = pygltflib.Primitive(attributes=pygltflib.Attributes(POSITION=pos_accessor_idx, COLOR_0=color_accessor_idx), mode=pygltflib.POINTS)
                mesh_idx = len(gltf.meshes)
                gltf.meshes.append(pygltflib.Mesh(primitives=[primitive]))

                node_idx = len(gltf.nodes)
                initial_scale = [1.0, 1.0, 1.0] if i == 0 else [0.0, 0.0, 0.0]
                gltf.nodes.append(pygltflib.Node(mesh=mesh_idx, scale=initial_scale))
                # Note: The node index is added to scene_nodes, but it seems it should be the node for the combined frame.
                # The current logic creates a new node for each sequence part of each frame.
                # For simplicity, we add all nodes to the scene. A more complex setup might group them.
                scene_nodes.append(node_idx)

                # Animation channel creation logic (same as before)
                # ... (The rest of the code is the same as before)

                # Create animation channel
                key_times_np = np.array([time_points[i], time_points[i+1]], dtype=np.float32)
                key_scales_np = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)

                if i == 0:
                    key_times_np = np.array([0.0, time_points[1]], dtype=np.float32)
                else:
                    key_times_np = np.array([time_points[i-1], time_points[i], time_points[i+1]], dtype=np.float32)
                    key_scales_np = np.array([[0.0,0.0,0.0], [1.0,1.0,1.0], [0.0,0.0,0.0]], dtype=np.float32)

                unique_times, unique_indices = np.unique(key_times_np, return_index=True)
                key_times_np = unique_times
                key_scales_np = key_scales_np[unique_indices]

                time_blob_offset = len(binary_blob)
                binary_blob.extend(key_times_np.tobytes())
                time_buffer_view_idx = len(gltf.bufferViews)
                gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=time_blob_offset, byteLength=len(key_times_np.tobytes())))
                time_accessor_idx = len(gltf.accessors)
                gltf.accessors.append(pygltflib.Accessor(bufferView=time_buffer_view_idx, componentType=pygltflib.FLOAT, count=len(key_times_np), type=pygltflib.SCALAR, min=[float(key_times_np.min())], max=[float(key_times_np.max())]))

                scale_blob_offset = len(binary_blob)
                binary_blob.extend(key_scales_np.tobytes())
                scale_buffer_view_idx = len(gltf.bufferViews)
                gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=scale_blob_offset, byteLength=len(key_scales_np.tobytes())))
                scale_accessor_idx = len(gltf.accessors)
                gltf.accessors.append(pygltflib.Accessor(bufferView=scale_buffer_view_idx, componentType=pygltflib.FLOAT, count=len(key_scales_np), type=pygltflib.VEC3))

                sampler_idx = len(animation.samplers)
                animation.samplers.append(pygltflib.AnimationSampler(input=time_accessor_idx, output=scale_accessor_idx, interpolation="STEP"))
                animation.channels.append(pygltflib.AnimationChannel(sampler=sampler_idx, target=pygltflib.AnimationChannelTarget(node=node_idx, path="scale")))

            except Exception as e:
                print(f"Error: Exception occurred while processing '{ply_path}'. Skipping. ({e})")
                continue
    
    # 4. Save the final GLB file
    gltf.scene = 0
    gltf.scenes.append(pygltflib.Scene(nodes=scene_nodes))
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_blob)))
    if animation.channels:
        gltf.animations.append(animation)
    
    print("Converting and saving to GLB file...")
    gltf.set_binary_blob(binary_blob)
    gltf.save(output_glb)
    print(f"Success! The merged file has been saved to '{output_glb}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merges two dynamic PLY sequences into a single GLB file.")
    parser.add_argument("input_dir1", type=str, help="Path to the first PLY sequence folder.")
    parser.add_argument("input_dir2", type=str, help="Path to the second PLY sequence folder.")
    parser.add_argument("output_glb", type=str, help="Path for the output .glb file.")
    parser.add_argument("--num_frames", type=int, required=True, help="Number of frames to process from each sequence.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the animation (default: 30).")
    parser.add_argument("--gap", type=float, default=0.1, help="Gap between the two sequences on the Z-axis (default: 0.1).")
    args = parser.parse_args()

    # ★ Added installation guide for the plyfile library ★
    try:
        from plyfile import PlyData
    except ImportError:
        print("Error: The 'plyfile' library is not installed.")
        print("Please install it before running the script using the command: 'pip install plyfile'")
        exit()

    combine_dynamic_sequences(args.input_dir1, args.input_dir2, args.output_glb, args.num_frames, args.fps, args.gap)
