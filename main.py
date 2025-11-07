from Tasks.first_tasks import count_intersections

if __name__ == "__main__":
    img_path = "Examples/example5.png"
    cnt, merged_segments, intersections = count_intersections(img_path,
                                                               canny_thresh1=50,
                                                               canny_thresh2=150,
                                                               hough_threshold=40,
                                                               minLineLength=30,
                                                               maxLineGap=10,
                                                               angle_merge_deg=5.0,
                                                               rho_merge_px=10.0,
                                                               cluster_dist=12.0,
                                                               visualize=True,
                                                               out_vis_path="intersections_vis.png",
                                                               apply_clahe=True,
                                                               apply_blur=True,
                                                               use_auto_scaling=True,
                                                               auto_base_diag=1000.0)
    print("Найдено пересечений (после слияния и кластеризации):", cnt) # type: ignore   
    print("Визуализация сохранена в intersections_vis.png") # type: ignore  
