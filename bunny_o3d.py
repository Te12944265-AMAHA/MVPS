import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

obj_name = "bunny"

base_dir = Path("datasets")
data_dir = base_dir / obj_name
sfm_dir = data_dir / "sfm"
cam_dir = sfm_dir / "cam"
recon_dir = data_dir / "recon"

pcd_path = recon_dir / "reconstructed_filtered.pcd"
cam_path = cam_dir / "Camera.006.npz"

# load camera
with np.load(cam_path) as X:
    K, RT = [X[i].astype(np.float64) for i in ("K", "RT")]
print(K)
RT2 = np.eye(4)
R = RT[:3,:3].T
t = -R @ RT[:3,3]
RT2[:3,:3] = R
RT2[:3,3] = t
#RT2[:3,:] = RT


# bunny = o3d.data.BunnyMesh()
# gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
# gt_mesh.compute_vertex_normals()

# pcd = gt_mesh.sample_points_poisson_disk(5000)
#o3d.visualization.draw_geometries([pcd])
pcd = o3d.io.read_point_cloud(str(pcd_path))
print(pcd)

# invalidate existing normals
pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
pcd.estimate_normals()
#o3d.visualization.draw_geometries([pcd], point_show_normal=True)

pcd.orient_normals_towards_camera_location(camera_location=RT[:,3])
#pcd.orient_normals_consistent_tangent_plane(100)
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
#print(type(pcd.normals))


# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as cm:
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8)
#print(mesh)
o3d.visualization.draw_geometries([mesh])

mesh.compute_vertex_normals()
#print(np.asarray(mesh.triangle_normals))
o3d.visualization.draw_geometries([mesh])

cx = K[0,2]
cy = K[1,2]
img_width = int(cx * 2)
img_height = int(cy * 2)
cx -= 0.5
cy -= 0.5
print(img_width, img_height)
print(cx, cy)
fx = K[0,0]
fy = K[1,1]

vis = o3d.visualization.Visualizer()
vis.create_window(width=img_width, height=img_height, visible=False)
vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
vis.add_geometry(mesh)

view_ctl = vis.get_view_control()
#assert id(view_ctl) == id(vis.get_view_control())


cam = view_ctl.convert_to_pinhole_camera_parameters()
#RT2[2,3] = 10.00
cam.extrinsic = RT2
#pinhole = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, K)
cam.intrinsic.set_intrinsics(img_width, img_height, fx, fy, cx, cy)
res = view_ctl.convert_from_pinhole_camera_parameters(cam)
print(res)


img = vis.capture_screen_float_buffer(True)


# current_param = view_ctl.convert_to_pinhole_camera_parameters()
# print(current_param.intrinsic.intrinsic_matrix)
# print(current_param.extrinsic)

#vis.run()
vis.destroy_window()

plt.imshow(np.asarray(img))
plt.show()
plt.close("all")