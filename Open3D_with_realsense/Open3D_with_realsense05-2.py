import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d


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

    pcd = pcd.voxel_down_sample(voxel_size=0.003)

    return pcd

def main():
    rs_frames = pipeline.wait_for_frames()
    pcd = convert_rs_frames_to_pointcloud(rs_frames)

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    while True:
        rs_frames = pipeline.wait_for_frames()
        pcd_new = convert_rs_frames_to_pointcloud(rs_frames)

        plane_model, inliers = pcd_new.segment_plane(distance_threshold=0.0025, ransac_n=3, num_iterations=1000)

        inlier_cloud_new = pcd_new.select_by_index(inliers)
        inlier_cloud_new.paint_uniform_color([1.0, 0, 0])
        outlier_cloud_new = pcd_new.select_by_index(inliers, invert=True)

        outlier_cloud.points = outlier_cloud_new.points
        outlier_cloud.colors = outlier_cloud_new.colors     

        o3d.visualization.add_geometries([inlier_cloud, outlier_cloud])


if __name__ == "__main__":
    main()