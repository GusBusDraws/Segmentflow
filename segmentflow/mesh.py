import open3d as o3d
from pathlib import Path


def save_stl(
        save_path,
        o3d_mesh,
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
    save_path = str(save_path)
    if not save_path.endswith('.stl'):
        save_path = f'{save_path}.stl'
    if Path(save_path).exists() and not allow_overwrite:
        raise ValueError(f'File already exists: {save_path}')
    else:
        o3d_mesh.compute_triangle_normals()
        o3d_mesh.compute_vertex_normals()
        # Write the mesh to STL file
        o3d.io.write_triangle_mesh(save_path, o3d_mesh)
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
        stl_save_path,
        smooth_iter=None,
        simplify_n_tris=None,
        iterative_simplify_factor=None,
        recursive_simplify=False,
        resave_mesh=False):
    print('Postprocessing surface meshes...')
    # Iterate through each STL file, load the mesh, and smooth/simplify
    for i, stl_path in enumerate(Path(stl_save_path).glob('*.stl')):
        stl_mesh, mesh_props = postprocess_mesh(
                stl_path,
                smooth_iter=smooth_iter,
                simplify_n_tris=simplify_n_tris,
                iterative_simplify_factor=iterative_simplify_factor,
                recursive_simplify=recursive_simplify,
                resave_mesh=resave_mesh)
        # props = {**props, **mesh_props}
    try:
        print(f'--> {i + 1} surface meshes postprocessed.')
    except NameError:
        print('No meshes found to postprocess.')

