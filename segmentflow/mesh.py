import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import sys


SHOW_WIREFRAME = False

def save_stl(
        save_path,
        o3d_mesh,
        mkdirs=False,
        allow_overwrite=False,
        suppress_save_message=False):
    """Save triangular mesh defined by vertices and face indices as an STL file.
    ----------
    Parameters
    ----------
    save_path : Path or str
        Path at which STL file will be saved. If doesn't end with '.stl',
        it will be added.
    o3d_mesh : open3d.geometry.TriangleMesh
        Triangle mesh loaded with Open3D package.
    suppress_save_message : bool, optional
        If True, particle label and STL file path will not be printed. By
        default False
    """
    save_path = Path(save_path)
    if not save_path.stem.endswith('.stl'):
        save_path = Path(save_path.parent) / f'{save_path.stem}.stl'
    # Make the parent directories if they don't exist
    if mkdirs:
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
    if save_path.exists() and not allow_overwrite:
        raise ValueError(f'File already exists: {save_path}')
    else:
        o3d_mesh.compute_triangle_normals()
        o3d_mesh.compute_vertex_normals()
        # Write the mesh to STL file
        o3d.io.write_triangle_mesh(str(save_path), o3d_mesh)
        if not suppress_save_message:
            print(f'STL saved: {save_path}')

def check_properties(stl_mesh):
    n_triangles = len(stl_mesh.triangles)
    edge_manifold = stl_mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = stl_mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = stl_mesh.is_vertex_manifold()
    self_intersecting = stl_mesh.is_self_intersecting()
    watertight = stl_mesh.is_watertight()
    orientable = stl_mesh.is_orientable()
    print(f"  n_triangles:            {n_triangles}")
    print(f"  watertight:             {watertight}")
    print(f"  self_intersecting:      {self_intersecting}")
    print(f"  orientable:             {orientable}")
    print(f"  vertex_manifold:        {vertex_manifold}")
    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    print()

def handle_args(args):
    """Function for plotting all the STL files in a directory.
    ----------
    Parameters
    ----------
    args : list
        Arguments passed to terminal in the form of a list
        (separated by spaces)
    """
    try:
        if args[0] == '-i':
            if len(args) == 2:
                meshes = load_stl_meshes(
                    args[1],
                    # fn_prefix='cake_0_',
                    # separate_color='cake_0'
                    # iter_size=100,
                )
            else:
                help()
                return
        elif args[0] == '-h':
            help()
            return
        else:
            help()
            return
        if len(meshes) != 0:
            o3d.visualization.draw_geometries(
                meshes,
                mesh_show_wireframe=SHOW_WIREFRAME,
                mesh_show_back_face=True,
                width=720, height=720, left=200, top=80,
            )
        else:
            print('No meshes loaded.')
    except IndexError:
        help()
        return

def help():
    print(
        'To view multiple STL files in a single Open3D window,'
        ' enter the following command:'
    )
    print('python -m view_mult_stl -i <Path to STL file directory>')

def load_stl_meshes(
        stl_dir_path,
        fn_prefix='',
        fn_suffix='',
        particleIDs=None,
        separate_color=None,
        colors='tab10',
        iter_size=1):
    stl_dir_path = Path(stl_dir_path)
    n_digits = 2
    if colors == 'four':
        colors = [
            (1.0, 0.7, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.7, 0.0),
            (0.0, 0.7, 1.0),
        ]
    elif colors == 'tab10':
        colors = plt.cm.tab10.colors
    if particleIDs is not None:
        stl_paths = [
            (
                f'{stl_dir_path}/{fn_prefix}'
                f'{str(particleID).zfill(n_digits)}{fn_suffix}.stl'
            )
            for i, particleID in enumerate(particleIDs)
            if i % iter_size == 0
        ]
    else:
        print(fn_prefix)
        stl_paths = [
            str(path) for i, path in enumerate(stl_dir_path.glob('*.stl'))
            if path.stem.startswith(fn_prefix)
            and path.stem.endswith(fn_suffix)
        ]
        print(len(stl_paths))
    meshes = []
    for i, path in enumerate(stl_paths):
        if Path(path).exists():
            print(f'Loading mesh: {path}')
            # stl_mesh = segment.postprocess_mesh(
            #     path, smooth_iter=1, simplify_n_tris=250, save_mesh=False,
            #     recursive_simplify=True, return_mesh=True, return_props=False
            # )
            # segment.check_properties(stl_mesh)
            # stl_mesh = segment.repair_mesh(stl_mesh)
            stl_mesh = o3d.io.read_triangle_mesh(str(path))
            stl_mesh.compute_triangle_normals()
            stl_mesh.compute_vertex_normals()
            meshes.append(stl_mesh)
        else:
            raise ValueError(f'Path not found: {path}')
    for i, m in enumerate(meshes):
        if separate_color is not None:
            if separate_color in Path(stl_paths[i]).stem:
                color = colors[0]
            else:
                color = colors[1]
        else:
            color = colors[i % len(colors)]
        m.paint_uniform_color(color)
    return meshes

def repair_mesh(stl_mesh):
    stl_mesh.remove_degenerate_triangles()
    stl_mesh.remove_duplicated_triangles()
    stl_mesh.remove_duplicated_vertices()
    stl_mesh.remove_non_manifold_edges()
    return stl_mesh

def simplify_mesh(
    stl_mesh, n_tris, recursive=False, failed_iter=10
):
    simplified_mesh = stl_mesh.simplify_quadric_decimation(n_tris)
    stl_mesh = repair_mesh(stl_mesh)
    stl_mesh.compute_triangle_normals()
    stl_mesh.compute_vertex_normals()
    if recursive and not simplified_mesh.is_watertight():
        simplified_mesh, n_tris = simplify_mesh(
            stl_mesh, n_tris + failed_iter, recursive=True
        )
    return simplified_mesh, n_tris

def simplify_mesh_iterative(
    stl_mesh, target_n_tris, return_mesh=True, iter_factor=2,
    suppress_save_msg=True
):
    og_n_tris = len(stl_mesh.triangles)
    prev_n_tris = len(stl_mesh.triangles)
    n_iters = 0
    while prev_n_tris > target_n_tris:
        stl_mesh, n_tris = simplify_mesh(stl_mesh, prev_n_tris // iter_factor)
        if n_tris == prev_n_tris:
            break
        prev_n_tris = n_tris
        n_iters += 1
    if not suppress_save_msg:
        print(
            f'Mesh simplified: {og_n_tris} -> {len(stl_mesh.triangles)}'
            f' in {n_iters} iterations'
        )
    if return_mesh:
        return stl_mesh

def postprocess_mesh(
        stl_save_path,
        smooth_iter=1,
        simplify_n_tris=250,
        iterative_simplify_factor=None,
        recursive_simplify=False,
        resave_mesh=False):
    stl_save_path = str(stl_save_path)
    stl_mesh = o3d.io.read_triangle_mesh(stl_save_path)
    stl_mesh = repair_mesh(stl_mesh)
    if smooth_iter is not None:
        stl_mesh = stl_mesh.filter_smooth_laplacian(
            number_of_iterations=smooth_iter
        )
    if simplify_n_tris is not None:
        if iterative_simplify_factor is not None:
            stl_mesh = simplify_mesh_iterative(
                stl_mesh, simplify_n_tris, iter_factor=iterative_simplify_factor
            )
        else:
            stl_mesh, n_tris = simplify_mesh(
                stl_mesh, simplify_n_tris, recursive=recursive_simplify,
                failed_iter=1
            )
    if resave_mesh:
        stl_mesh.compute_triangle_normals()
        stl_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(
            stl_save_path, stl_mesh,
            # Currently unsupported to save STLs in ASCII format
            # write_ascii=True
        )
    mesh_props = {}
    mesh_props['n_triangles'] = len(stl_mesh.triangles)
    mesh_props['watertight'] = stl_mesh.is_watertight()
    mesh_props['self_intersecting'] = stl_mesh.is_self_intersecting()
    mesh_props['orientable'] = stl_mesh.is_orientable()
    mesh_props['edge_manifold'] = stl_mesh.is_edge_manifold(
        allow_boundary_edges=True
    )
    mesh_props['edge_manifold_boundary'] = stl_mesh.is_edge_manifold(
        allow_boundary_edges=False
    )
    mesh_props['vertex_manifold'] = stl_mesh.is_vertex_manifold()
    return stl_mesh, mesh_props

def postprocess_meshes(
        stl_dir_path,
        skip_first_stl=False,
        smooth_iter=None,
        simplify_n_tris=None,
        iterative_simplify_factor=None,
        recursive_simplify=False,
        suppress_save_message=True,
        resave_mesh=False,
        save_dir_path=None):
    print('Postprocessing surface meshes...')
    # Iterate through each STL file, load the mesh, and smooth/simplify
    stl_path_list = [path for path in Path(stl_dir_path).glob('*.stl')]
    if skip_first_stl:
        stl_path_list = stl_path_list[1:]
    print(f'--> {len(stl_path_list)} STL file(s) to postprocess')
    for i, stl_path in enumerate(stl_path_list):
        stl_mesh, mesh_props = postprocess_mesh(
            stl_path,
            smooth_iter=smooth_iter,
            simplify_n_tris=simplify_n_tris,
            iterative_simplify_factor=iterative_simplify_factor,
            recursive_simplify=recursive_simplify,
            resave_mesh=resave_mesh)
        # props = {**props, **mesh_props}
        # Save mesh in separate location if resave is false
        if not resave_mesh:
            if save_dir_path is None:
                raise ValueError(
                    'If not resaving meshes, you must pass value for '
                    '"save_dir_path" keyword argument.')
            else:
                save_stl(
                    Path(save_dir_path) / str(stl_path.name),
                    stl_mesh,
                    suppress_save_message=suppress_save_message,
                    mkdirs=True)
    try:
        print(f'--> {i + 1} surface meshes postprocessed.')
    except NameError:
        print('No meshes found to postprocess.')

if __name__ == "__main__":
    # wrap_lines_in_file(sys.argv[-1])
    handle_args(sys.argv[1:])
