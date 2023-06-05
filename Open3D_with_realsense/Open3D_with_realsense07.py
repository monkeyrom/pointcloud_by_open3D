import open3d as o3d

def main():

    pcd = o3d.io.read_point_cloud("C:\\Users\\ROM\\Desktop\\Visual Studio\\New Point Cloud\\CAD\\Oximeter.ply")
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)

    o3d.visualization.draw_geometries([pcd, aabb])

if __name__ == "__main__":
    main()