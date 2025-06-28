import argparse
import os
import numpy as np
from plyfile import PlyData
import pygltflib

def inspect_ply_header(ply_path):
    """Reads the header of a PLY file and prints its property names."""
    try:
        with open(ply_path, 'rb') as f:
            plydata = PlyData.read(f)
        vertex_element = plydata['vertex']
        print(f"--- Header Information for '{os.path.basename(ply_path)}' ---")
        print(f"Total {vertex_element.count} vertices (Gaussians) found.")
        print("Available properties:")
        for prop in vertex_element.properties:
            print(f"- {prop.name} ({prop.val_dtype})")
        print("-" * 20)
    except Exception as e:
        print(f"Error: Failed to read PLY file header. {e}")

def convert_gaussian_to_pointcloud_glb(args):
    """
    Converts a 3D Gaussian PLY file to a standard point cloud GLB file.
    Pre-calculates color and opacity and saves them as vertex colors.
    """
    try:
        with open(args.input_ply, 'rb') as f:
            plydata = PlyData.read(f)
        vertex_element = plydata['vertex']
        vertex_count = vertex_element.count
    except Exception as e:
        print(f"Error: Could not read PLY file: {e}")
        return

    print(f"Load complete: {vertex_count} Gaussians")

    try:
        # --- 1. Extract required data ---
        positions = np.vstack([vertex_element[p] for p in args.pos_props]).T.astype(np.float32)
        
        # ★★★ Key modification: Convert from Z-up to Y-up coordinate system ★★★
        # Temporarily store original y and z values
        original_y = positions[:, 1].copy()
        original_z = positions[:, 2].copy()
        # The new y becomes the original z, and the new z becomes the original -y.
        positions[:, 1] = original_z
        positions[:, 2] = -original_y
        print("Coordinate system conversion complete (Z-up -> Y-up)")
        
        colors_dc = np.vstack([vertex_element[p] for p in args.color_props]).T.astype(np.float32)
        opacities = vertex_element[args.opacity_prop].astype(np.float32)
    except KeyError as e:
        print(f"Error: Property '{e.args[0]}' not found. Use the --inspect option to check the correct property names.")
        return
    except Exception as e:
        print(f"Error: A problem occurred during data extraction. {e}")
        return

    print("Data extraction complete. Starting final color calculation...")

    # --- 2. Calculate final vertex colors ---
    colors = 1.0 / (1.0 + np.exp(-colors_dc))
    alpha = 1.0 / (1.0 + np.exp(-opacities))
    final_colors_float = colors * alpha[:, np.newaxis]
    final_colors_uint8 = (final_colors_float.clip(0, 1) * 255).astype(np.uint8)
    print("Color calculation complete.")
    
    # --- 3. Create standard point cloud GLB structure ---
    gltf = pygltflib.GLTF2()
    gltf.scenes.append(pygltflib.Scene(nodes=[0]))
    gltf.scene = 0
    gltf.nodes.append(pygltflib.Node(mesh=0))

    binary_blob = positions.tobytes() + final_colors_uint8.tobytes()
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_blob)))
    
    pos_byte_length = positions.nbytes
    gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=0, byteLength=pos_byte_length))
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=0, componentType=pygltflib.FLOAT, count=vertex_count, type=pygltflib.VEC3,
        min=np.min(positions, axis=0).tolist(), max=np.max(positions, axis=0).tolist()
    ))

    color_byte_length = final_colors_uint8.nbytes
    gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=pos_byte_length, byteLength=color_byte_length))
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=1, componentType=pygltflib.UNSIGNED_BYTE, count=vertex_count,
        type=pygltflib.VEC3, normalized=True
    ))
    
    attributes = pygltflib.Attributes(POSITION=0, COLOR_0=1)
    mesh = pygltflib.Mesh(primitives=[pygltflib.Primitive(attributes=attributes, mode=pygltflib.POINTS)])
    gltf.meshes.append(mesh)

    # --- 4. Save as GLB file ---
    gltf.set_binary_blob(binary_blob)
    output_dir = os.path.dirname(args.output_glb)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    gltf.save(args.output_glb)
    print(f"Success: Standard point cloud GLB file '{os.path.basename(args.output_glb)}' has been saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts a 3D Gaussian PLY file to a standard point cloud GLB.")
    
    parser.add_argument("input_ply", type=str, help="Path to the source .ply file to be converted")
    parser.add_argument("output_glb", nargs='?', default=None, help="Path for the output .glb file (if omitted, runs in --inspect mode)")
    parser.add_argument("--inspect", action="store_true", help="Inspect the PLY file's header properties and exit.")
    parser.add_argument("--pos_props", nargs=3, default=['x', 'y', 'z'], help="Property names for position (x,y,z). Default: 'x' 'y' 'z'")
    parser.add_argument("--color_props", nargs=3, default=['f_dc_0', 'f_dc_1', 'f_dc_2'], help="Property names for DC color (r,g,b). Default: 'f_dc_0' 'f_dc_1' 'f_dc_2'")
    parser.add_argument("--opacity_prop", type=str, default='opacity', help="Property name for opacity. Default: 'opacity'")
    
    args = parser.parse_args()

    if args.inspect or args.output_glb is None:
        if not args.input_ply:
            print("Error: The input_ply path must be specified to use the --inspect option.")
        else:
            inspect_ply_header(args.input_ply)
    else:
        convert_gaussian_to_pointcloud_glb(args)
