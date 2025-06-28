import argparse
import glob
import os
import numpy as np
from plyfile import PlyData
import pygltflib

def create_visibility_animated_glb(input_dir, output_glb, fps=30, num_frames=None):
    """
    (v2) Converts sequential PLY files with varying vertex counts into a single GLB file 
    using a 'show/hide' animation. Improved stability and added data validation.

    :param input_dir: Input directory containing sequential .ply files.
    :param output_glb: Path for the output .glb file.
    :param fps: Frames Per Second for the animation.
    :param num_frames: Maximum number of frames to process. If None, all files are processed.
    """
    ply_files = sorted(glob.glob(os.path.join(input_dir, '*.ply')))
    if not ply_files:
        print(f"Error: No PLY files found in the directory '{input_dir}'.")
        return

    if num_frames is not None and num_frames > 0:
        if len(ply_files) > num_frames:
            ply_files = ply_files[:num_frames]
            print(f"Processing a maximum of {num_frames} frames as specified.")
        else:
            print(f"Warning: The requested number of frames ({num_frames}) is greater than or equal to the actual number of files ({len(ply_files)}). Processing all files.")

    print(f"Converting a total of {len(ply_files)} PLY files.")

    gltf = pygltflib.GLTF2()
    binary_blob = bytearray()
    scene_nodes = []
    animation = pygltflib.Animation()
    
    # Calculate the time for each keyframe of the animation
    total_frames = len(ply_files)
    time_points = np.linspace(0, total_frames / fps, total_frames + 1, dtype=np.float32)

    try:
        # 1. Process each PLY file as a separate node and mesh
        for i, ply_path in enumerate(ply_files):
            print(f"Processing... ({i+1}/{total_frames}) {os.path.basename(ply_path)}")
            
            # Read PLY file and validate data
            ply_data = PlyData.read(ply_path)
            vertices = ply_data['vertex']
            vertex_count = vertices.count

            positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T.astype(np.float32)
            colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T.astype(np.uint8)

            if np.isnan(positions).any() or np.isinf(positions).any():
                print(f"Warning: Skipping '{ply_path}' due to invalid position data (NaN/Inf).")
                continue

            # --- Create GLTF structure (repeated for each frame) ---
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
            scene_nodes.append(node_idx)
            
            # --- Create animation channel ---
            # Implement show/hide effect by controlling the scale of each node.
            
            # Keyframe Times: [ Start-Hidden, CurrentFrame-Start, NextFrame-Start ]
            # Keyframe Scales: [ [0,0,0],      [1,1,1],          [0,0,0]           ]
            
            key_times = []
            key_scales = []

            # At time 0.0, all frames (except the first) start in a hidden state
            if i > 0:
                key_times.append(0.0)
                key_scales.append([0.0, 0.0, 0.0])

            # Time when the current frame appears
            key_times.append(time_points[i])
            key_scales.append([1.0, 1.0, 1.0])

            # Time when the current frame disappears (start time of the next frame)
            key_times.append(time_points[i+1])
            key_scales.append([0.0, 0.0, 0.0])

            anim_times_np = np.array(key_times, dtype=np.float32)
            anim_scales_np = np.array(key_scales, dtype=np.float32)

            # Create Time (Input) Accessor and BufferView
            time_blob_offset = len(binary_blob)
            binary_blob.extend(anim_times_np.tobytes())
            time_buffer_view_idx = len(gltf.bufferViews)
            gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=time_blob_offset, byteLength=len(anim_times_np.tobytes())))
            time_accessor_idx = len(gltf.accessors)
            gltf.accessors.append(pygltflib.Accessor(bufferView=time_buffer_view_idx, componentType=pygltflib.FLOAT, count=len(anim_times_np), type=pygltflib.SCALAR, min=[float(anim_times_np.min())], max=[float(anim_times_np.max())]))

            # Create Scale (Output) Accessor and BufferView
            scale_blob_offset = len(binary_blob)
            binary_blob.extend(anim_scales_np.tobytes())
            scale_buffer_view_idx = len(gltf.bufferViews)
            gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=scale_blob_offset, byteLength=len(anim_scales_np.tobytes())))
            scale_accessor_idx = len(gltf.accessors)
            gltf.accessors.append(pygltflib.Accessor(bufferView=scale_buffer_view_idx, componentType=pygltflib.FLOAT, count=len(anim_scales_np), type=pygltflib.VEC3))

            sampler_idx = len(animation.samplers)
            animation.samplers.append(pygltflib.AnimationSampler(input=time_accessor_idx, output=scale_accessor_idx, interpolation="STEP"))
            animation.channels.append(pygltflib.AnimationChannel(sampler=sampler_idx, target=pygltflib.AnimationChannelTarget(node=node_idx, path="scale")))

    except Exception as e:
        print(f"Error: An unexpected problem occurred during GLB file creation: {e}")
        return

    # 2. Finalize GLTF object structure
    gltf.scene = 0
    gltf.scenes.append(pygltflib.Scene(nodes=scene_nodes))
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_blob)))
    if animation.channels:
        gltf.animations.append(animation)

    # 3. Save as GLB file
    print("All files processed. Converting to GLB file...")
    gltf.set_binary_blob(binary_blob)
    gltf.save(output_glb)
    print(f"Success: Animated GLB file '{output_glb}' has been saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts sequential PLY files with varying vertex counts into a single animated GLB file.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing sequential .ply files.")
    parser.add_argument("output_glb", type=str, help="Path for the output .glb file.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the animation (default: 30).")
    parser.add_argument("--num_frames", type=int, default=None, help="Maximum number of frames to process. If not specified, all files in the folder are processed.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
    else:
        create_visibility_animated_glb(args.input_dir, args.output_glb, args.fps, args.num_frames)
