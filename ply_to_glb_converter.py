import numpy as np
from plyfile import PlyData
import pygltflib
import struct
import os
import argparse # Import library for CLI argument processing

def convert_ply_to_glb(ply_path, glb_path):
    """
    Converts a PLY file containing vertex positions and vertex colors
    to a GLB file.

    :param ply_path: Path to the input .ply file
    :param glb_path: Path to the output .glb file
    """
    # --- 1. Load PLY file and extract data ---
    try:
        plydata = PlyData.read(ply_path)
        vertex_element = plydata['vertex']
        vertex_count = vertex_element.count
    except Exception as e:
        print(f"Error: Could not read PLY file. Check the file path and format. Error: {e}")
        return

    positions = np.vstack([
        vertex_element['x'],
        vertex_element['y'],
        vertex_element['z']
    ]).T.astype(np.float32)

    colors = np.vstack([
        vertex_element['red'],
        vertex_element['green'],
        vertex_element['blue']
    ]).T.astype(np.uint8)

    print(f"PLY file loaded successfully: {vertex_count} vertices")

    # --- 2. Convert the extracted data into a single binary blob ---
    positions_binary_blob = positions.tobytes()
    colors_binary_blob = colors.tobytes()
    binary_blob = positions_binary_blob + colors_binary_blob

    # --- 3. Create glTF 2.0 structure ---
    gltf = pygltflib.GLTF2()
    gltf.scenes.append(pygltflib.Scene(nodes=[0]))
    gltf.scene = 0
    gltf.nodes.append(pygltflib.Node(mesh=0))
    
    buffer = pygltflib.Buffer()
    buffer.byteLength = len(binary_blob)
    gltf.buffers.append(buffer)

    positions_buffer_view = pygltflib.BufferView(
        buffer=0,
        byteOffset=0,
        byteLength=len(positions_binary_blob),
        target=pygltflib.ARRAY_BUFFER
    )
    gltf.bufferViews.append(positions_buffer_view)

    colors_buffer_view = pygltflib.BufferView(
        buffer=0,
        byteOffset=len(positions_binary_blob),
        byteLength=len(colors_binary_blob),
        target=pygltflib.ARRAY_BUFFER
    )
    gltf.bufferViews.append(colors_buffer_view)

    positions_accessor = pygltflib.Accessor(
        bufferView=0,
        componentType=pygltflib.FLOAT,
        count=vertex_count,
        type=pygltflib.VEC3,
        min=np.min(positions, axis=0).tolist(),
        max=np.max(positions, axis=0).tolist()
    )
    gltf.accessors.append(positions_accessor)

    colors_accessor = pygltflib.Accessor(
        bufferView=1,
        componentType=pygltflib.UNSIGNED_BYTE,
        count=vertex_count,
        type=pygltflib.VEC3,
        normalized=True
    )
    gltf.accessors.append(colors_accessor)

    mesh = pygltflib.Mesh()
    primitive = pygltflib.Primitive(
        attributes=pygltflib.Attributes(POSITION=0, COLOR_0=1),
        mode=pygltflib.POINTS
    )
    mesh.primitives.append(primitive)
    gltf.meshes.append(mesh)

    # --- 4. Save to .glb file including the binary data ---
    gltf.set_binary_blob(binary_blob)
    
    # Create the output directory if it does not exist
    output_dir = os.path.dirname(glb_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    gltf.save(glb_path)
    print(f"Success: '{os.path.basename(glb_path)}' has been saved.")


# --- Script execution section ---
if __name__ == '__main__':
    # 1. Create ArgumentParser object
    parser = argparse.ArgumentParser(
        description="A script to convert PLY files to GLB files."
    )

    # 2. Define CLI arguments
    # First argument: Input file path (required)
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the source .ply file to be converted"
    )
    # Second argument: Output file path (required)
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the output .glb file"
    )

    # 3. Parse the defined arguments
    args = parser.parse_args()

    # 4. Check if the input file exists, then call the conversion function
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
    elif not args.input_file.lower().endswith('.ply'):
        print(f"Error: Input file must be in .ply format.")
    else:
        convert_ply_to_glb(args.input_file, args.output_file)
