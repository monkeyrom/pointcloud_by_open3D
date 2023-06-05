import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import math

align = rs.align(rs.stream.color)
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
pipeline = rs.pipeline()

profile = pipeline.start(config)

voxel_param = 0.0008
distance_thres = 0.006
num_it = 1000
rans=3

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

    pcd = pcd.voxel_down_sample(voxel_size=voxel_param)

    return pcd

def main():
    rs_frames = pipeline.wait_for_frames()
    pcd = convert_rs_frames_to_pointcloud(rs_frames)

    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_thres, ransac_n=rans, num_iterations=num_it)

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    
    obb = o3d.geometry.OrientedBoundingBox.get_oriented_bounding_box(outlier_cloud)
    #aabb = o3d.geometry.AxisAlignedBoundingBox.get_oriented_bounding_box(outlier_cloud)
    #obb = outlier_cloud.get_oriented_bounding_box()
    #obb = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(outlier_cloud)
    

    #bb = o3d.geometry.AxisAlignedBoundingBox.get_oriented_bounding_box(inlier_cloud)
    #bb = inlier_cloud.get_oriented_bounding_box()
    #bb = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(inlier_cloud)
    
    obb.color =     (0.0, 1.0, 0.0)
    #aabb.color =     (0.0, 0.0, 1.0)
    #bb.color =      (0.0, 0.0, 1.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Visualizer",width=1080, height=640)
    
    vis.add_geometry(inlier_cloud)
    vis.add_geometry(outlier_cloud)
    vis.add_geometry(obb)
    #vis.add_geometry(aabb)

    render_opt = vis.get_render_option()
    render_opt.point_size = 2

    while True:
        rs_frames = pipeline.wait_for_frames()
        pcd_new = convert_rs_frames_to_pointcloud(rs_frames)

        plane_model, inliers = pcd_new.segment_plane(distance_threshold=distance_thres, ransac_n=rans, num_iterations=num_it)

        inlier_cloud_new = pcd_new.select_by_index(inliers)
        inlier_cloud_new.paint_uniform_color([1, 0, 0])
        outlier_cloud_new = pcd_new.select_by_index(inliers, invert=True)

        outlier_cloud.points = outlier_cloud_new.points
        outlier_cloud.colors = outlier_cloud_new.colors 

        inlier_cloud.points = inlier_cloud_new.points
        inlier_cloud.colors = inlier_cloud_new.colors    

        obb_new = o3d.geometry.OrientedBoundingBox.get_oriented_bounding_box(outlier_cloud_new)
        #aabb_new = o3d.geometry.AxisAlignedBoundingBox.get_oriented_bounding_box(outlier_cloud)
        #bb_new = o3d.geometry.AxisAlignedBoundingBox.get_oriented_bounding_box(inlier_cloud_new)

        obb.center = obb_new.center
        obb.R = obb_new.R
        obb.extent = obb_new.extent
        #aabb.center = aabb_new.center
        #aabb.extent = aabb_new.extent
        #bb.center = bb_new.center
        #bb.extent = bb_new.extent

        obb.color =     (0.0, 1.0, 0.0)
        #aabb.color =    (0.0, 0.0, 1.0)

        vis.update_geometry(outlier_cloud)
        vis.update_geometry(inlier_cloud)
        vis.update_geometry(obb)
        #vis.update_geometry(aabb)
        obb_x = round(obb.center[0]*1000, 2)
        obb_y = round(obb.center[1]*1000, 2)
        obb_z = round(obb.center[2]*1000, 2)

        roll = math.atan2(obb.R[2, 1], obb.R[2, 2])
        pitch = math.asin(-obb.R[2, 0])
        yaw = math.atan2(obb.R[1, 0], obb.R[0, 0])

        roll_deg = round(math.degrees(roll), 2)
        pitch_deg = round(math.degrees(pitch), 2)
        yaw_deg = round(math.degrees(yaw), 2)

        print("Center of OBB")
        print("x: ",obb_x,"\ny: ",obb_y,"\nz: ",obb_z,"\n")
        
        print("Rotation of OBB")
        print("roll: ",roll_deg,"\npitch: ",pitch_deg,"\nyaw: ",yaw_deg,"\n")
        print("--------------------------------------------------------")

        if vis.poll_events():
            vis.update_renderer()
        else:
            break

    vis.destroy_window()
    pipeline.stop()

if __name__ == "__main__":
    main()