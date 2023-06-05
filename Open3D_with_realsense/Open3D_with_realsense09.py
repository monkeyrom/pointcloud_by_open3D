import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

align = rs.align(rs.stream.color)
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
pipeline = rs.pipeline()

profile = pipeline.start(config)

intr = profile.get_stream(
    rs.stream.color).as_video_stream_profile().get_intrinsics()
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
extrinsic = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

def convert_rs_frames_to_pointcloud(rs_frames):
    aligned_frames = align.process(rs_frames)
    rs_depth_frame = aligned_frames.get_depth_frame()
    np_depth = np.asanyarray(rs_depth_frame.get_data())
    o3d_depth = o3d.geometry.Image(np_depth)

    rs_color_frame = aligned_frames.get_color_frame()
    np_color = np.asanyarray(rs_color_frame.get_data())
    o3d_color = o3d.geometry.Image(np_color)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth, depth_scale=4000.0, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, pinhole_camera_intrinsic, extrinsic)

    pcd = pcd.voxel_down_sample(voxel_size=0.001)

    return pcd

def main():
    rs_frames = pipeline.wait_for_frames()
    pcd = convert_rs_frames_to_pointcloud(rs_frames)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Visualizer",
                      width=800, height=800)
    vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = 2
    vis.get_render_option().point_show_normal = True
    
    while True:
        rs_frames = pipeline.wait_for_frames()
        pcd_new = convert_rs_frames_to_pointcloud(rs_frames)

        pcd.points = pcd_new.points
        pcd.colors = pcd_new.colors     

        pcd_new = pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                pcd.cluster_dbscan(eps=0.01, min_points=5, print_progress=True))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        vis.update_geometry(pcd)
        if vis.poll_events():
            vis.update_renderer()
        else:
            break

    vis.destroy_window()
    pipeline.stop()

if __name__ == "__main__":
    main()