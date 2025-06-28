import argparse
import os
import trimesh
import numpy as np

def combine_ply_side_by_side(ply_path1, ply_path2, output_glb, axis='x', gap=0.0):
    """
    Loads two PLY files, places them side-by-side based on their bounding boxes,
    and saves them as a single GLB file.

    :param ply_path1: Path to the first input .ply file.
    :param ply_path2: Path to the second input .ply file.
    :param output_glb: Path for the output .glb file.
    :param axis: The axis along which to place the files side-by-side ('x', 'y', 'z').
    :param gap: The gap between the two files.
    """
    # 1. Load the two PLY files as point clouds.
    print(f"Loading files '{os.path.basename(ply_path1)}' and '{os.path.basename(ply_path2)}'...")
    try:
        pc1 = trimesh.load(ply_path1)
        pc2 = trimesh.load(ply_path2)
    except Exception as e:
        print(f"Error: A problem occurred while loading files. {e}")
        return

    # If trimesh loaded a mesh, process it to use only vertex data.
    if isinstance(pc1, trimesh.Trimesh):
        pc1 = trimesh.points.PointCloud(pc1.vertices, colors=pc1.visual.vertex_colors)
    if isinstance(pc2, trimesh.Trimesh):
        pc2 = trimesh.points.PointCloud(pc2.vertices, colors=pc2.visual.vertex_colors)
        
    print("Files loaded successfully.")

    # 2. Calculate the bounding box of the first point cloud.
    # pc1.bounds is a numpy array of the form [[min_x, min_y, min_z], [max_x, max_y, max_z]].
    min_bound, max_bound = pc1.bounds
    
    # 3. Calculate the translation (displacement) for the second point cloud.
    # Calculate the translation amount based on the first point cloud's size for a cleaner alignment.
    translation = np.zeros(3)
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis in axis_map:
        axis_index = axis_map[axis]
        # Move by the width/height/depth of the first object + gap.
        size = max_bound[axis_index] - min_bound[axis_index]
        translation[axis_index] = size + gap
        print(f"Translating the second object by {translation[axis_index]:.2f} along the {axis.upper()}-axis.")
    else:
        print(f"Error: Invalid axis '{axis}'. Please choose one of 'x', 'y', 'z'.")
        return

    # 4. Apply the calculated translation to the second point cloud.
    pc2.apply_translation(translation)
    print("Translation applied successfully.")

    # 5. Add both point clouds to a single Scene.
    scene = trimesh.Scene()
    scene.add_geometry(pc1)
    scene.add_geometry(pc2)
    print("Merge into a single scene complete.")

    # 6. Export the scene to a GLB file.
    try:
        # Create directory if it doesn't exist.
        output_dir = os.path.dirname(output_glb)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        scene.export(file_obj=output_glb, file_type='glb')
        print(f"Success: The combined file has been saved to '{output_glb}'.")
    except Exception as e:
        print(f"Error: A problem occurred while exporting to GLB file. {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Combines two PLY point clouds side-by-side into a single GLB file."
    )
    parser.add_argument("input_ply1", type=str, help="Path to the first source .ply file.")
    parser.add_argument("input_ply2", type=str, help="Path to the second source .ply file.")
    parser.add_argument("output_glb", type=str, help="Path for the combined output .glb file.")
    parser.add_argument(
        "--axis", type=str, default='x', choices=['x', 'y', 'z'],
        help="Specifies the axis to place the objects side-by-side (default: 'x')."
    )
    parser.add_argument(
        "--gap", type=float, default=0.1,
        help="Specifies the gap between the two objects (default: 0.1)."
    )
    args = parser.parse_args()

    combine_ply_side_by_side(args.input_ply1, args.input_ply2, args.output_glb, args.axis, args.gap)
