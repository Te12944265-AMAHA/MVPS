import numpy as np
import copy
import scipy.interpolate
import open3d as o3d
import matplotlib.pyplot as plt
import os

from configs import maps_dir, cams_dir, cams_dir_raw, pcd_path

def fit_surface(ids, vis=False):
    print("Fitting mesh to point cloud..")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    #print(pcd)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(30)
    if vis:
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8)
    vertices_to_remove = densities < np.quantile(densities, 0.015)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    if vis:
        o3d.visualization.draw_geometries([mesh])

    mesh.compute_vertex_normals()
    if vis:
        o3d.visualization.draw_geometries([mesh])

    # all images
    for cam_id in ids:
        img_id = f"Camera.{cam_id:03d}"
        print(f"Processing {img_id}...")
        cam_matrix_save_path = f"{cams_dir}/{img_id}.npz"
        # load camera (from sfm). T^w_c. Not the cam used in stereo_flash_no_flash
        cam_path = f"{cams_dir_raw}/{img_id}.npz"
        with np.load(cam_path) as X:
            K, RT = [X[i].astype(np.float64) for i in ("K", "RT")]
        # print(K)
        # print(RT)
        # continue
        RT2 = np.eye(4)
        R = RT[:3,:3].T
        t = -R @ RT[:3,3]
        RT2[:3,:3] = R
        RT2[:3,3] = t
            
        normals_vis = mesh_to_normal(K, RT2, mesh)
        # change to [-1,1]
        h, w, _ = normals_vis.shape
        print(h, w)
        normals = normals_vis * 2.0 - 1.0
        norm = np.linalg.norm(normals, axis=2).reshape((h, w, 1))
        if vis:
            plt.imshow(norm)
            plt.show()
            plt.close("all")
        denom = np.where(norm > 1.7, 1e-5, norm)
        normals = np.where(norm > 1.7, 0.0, normals/denom)

        dmap = np.zeros((h, w))

        # save normal and depth maps
        maps_save_path = os.path.join(maps_dir, f"{img_id}.npz")
        res_out = {
            "depth_map": dmap,
            "normal_map": normals,
        }
        np.savez(maps_save_path, **res_out)

        # save transformed camera matrix
        res_out = {
            "K": K,
            "RT": RT2[:3,:],
        }
        np.savez(cam_matrix_save_path, **res_out)

        if vis:
            plt.imshow(normals_vis)
            plt.show()
            plt.close("all")



def mesh_to_normal(K, RT2, mesh):
    cx = K[0,2]
    cy = K[1,2]
    img_width = int(cx * 2)
    img_height = int(cy * 2)
    cx -= 0.5
    cy -= 0.5
    fx = K[0,0]
    fy = K[1,1]

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_width, height=img_height, visible=False)
    vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
    vis.add_geometry(mesh)

    view_ctl = vis.get_view_control()

    cam = view_ctl.convert_to_pinhole_camera_parameters()
    cam.extrinsic = RT2
    cam.intrinsic.set_intrinsics(img_width, img_height, fx, fy, cx, cy)
    res = view_ctl.convert_from_pinhole_camera_parameters(cam)

    img = vis.capture_screen_float_buffer(True)

    vis.destroy_window()
    return np.asarray(img)


if __name__ == "__main__":
    fit_surface([23], vis=True)