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

    return pcd

def main():
    rs_frames = pipeline.wait_for_frames()
    pcd = convert_rs_frames_to_pointcloud(rs_frames)

    pcd_down = pcd.voxel_down_sample(voxel_size=0.001)
    
    plane_model, inliers = pcd_down.segment_plane(distance_threshold=0.0025, ransac_n=3, num_iterations=1000)

    inlier_cloud = pcd_down.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd_down.select_by_index(inliers, invert=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Visualizer",width=800, height=800, left=50, top=50, visible=True)
    
    vis.add_geometry(inlier_cloud)
    vis.add_geometry(outlier_cloud)

    render_opt = vis.get_render_option()
    render_opt.point_size = 2

    while True:
        rs_frames = pipeline.wait_for_frames()
        pcd_new = convert_rs_frames_to_pointcloud(rs_frames)

        pcd_down = pcd_new.voxel_down_sample(voxel_size=0.001)

        plane_model, inliers = pcd_down.segment_plane(distance_threshold=0.003, ransac_n=3, num_iterations=1000)

        inlier_cloud_new = pcd_down.select_by_index(inliers)
        inlier_cloud_new.paint_uniform_color([1.0, 1.0, 1.0])
        outlier_cloud_new = pcd_down.select_by_index(inliers, invert=True)

        outlier_cloud.points = outlier_cloud_new.points
        outlier_cloud.colors = outlier_cloud_new.colors 

        inlier_cloud.points = inlier_cloud_new.points
        inlier_cloud.colors = inlier_cloud_new.colors    

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(outlier_cloud.cluster_dbscan(eps=0.005, min_points=50, print_progress=True))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

        vis.update_geometry(outlier_cloud)
        vis.update_geometry(inlier_cloud)

        if vis.poll_events():
            vis.update_renderer()
        else:
            break

    vis.destroy_window()
    pipeline.stop()

if __name__ == "__main__":
    main()