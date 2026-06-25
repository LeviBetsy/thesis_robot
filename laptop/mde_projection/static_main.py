def main(imshow=False):
    # Initialize the depth model (will auto-select mps, cuda, or cpu)
    print("Loading Depth Anything V2 model...")
    predictor = DepthAnythingPredictor()

    # Initialize Robot
    robot = Robot("fisheye_calib.npz")
    fsc = FloorScaleCorrection("z_real")
    pcd_processor = PointCloudProcessor(robot)


    pcd = o3d.geometry.PointCloud() #Visualizer

    prev_time = time.perf_counter()
    try:
        while True:
            frame = cv2.imread("./data/test/und_ref8.png")

            rel_depth_map = predictor.model.infer_image(frame)

            # ***************** Visualize ******************
            if imshow:
                color_depth = predictor.colorize(rel_depth_map)
                cv2.imshow("Robot Live Camera Feed - Depth Map", color_depth)
                # Check for 'q' key press to break loop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            # ***********************************************
            fsc.scale_correction(rel_depth_map, True, "error.png")
            metric_map = fsc.relative_to_metric(rel_depth_map)

            point_cloud_cc = pcd_processor.proj_pcd_cc(metric_map)
            point_cloud_rc = pcd_processor.pcd_camera_to_robot(point_cloud_cc)
            z_avg = pcd_processor.average_floor_z(point_cloud_rc)
            print(f"average z: {z_avg}")

            pcd.points = o3d.utility.Vector3dVector(point_cloud_rc)


            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd, axes])


            # ***************** Debug ***********************
            current_time = time.perf_counter()
            elapsed = current_time - prev_time
            print(f"FPS: {1/elapsed:.2f} | Latency: {elapsed * 1000:.1f}ms")
            prev_time = current_time
            # ***********************************************
            break

    finally:
        # Graceful cleanup
        # cap.release()
        cv2.destroyAllWindows()
        print("Pipeline stopped and windows closed.")

if __name__ == "__main__":
    main(imshow=True)