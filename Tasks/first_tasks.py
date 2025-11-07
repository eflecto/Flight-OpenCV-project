from typing import Any, Iterable, Tuple
import cv2
import numpy as np
from itertools import combinations
import math


def normalize_angle_theta(theta: float) -> float:
    """
    Нормализует угол theta в диапазоне [0, pi)
    """
    if theta < 0:
        theta += math.pi
    if theta >= math.pi:
        theta -= math.pi
    return theta


def merge_collinear_segments(segments: Iterable[Tuple[float, float, float, float]],
                             angle_thresh_deg: float = 5.0,
                             rho_thresh: float = 20.0,
                             extend_px: float = 0.0) -> list[Tuple[float, float, float, float]]:
    """
    segments: list of (x1,y1,x2,y2)
    Возвращает список объединённых сегментов (по кластерам коллинеарности).
    Параметры:
      angle_thresh_deg - максимально допустимая разница углов (в градусах)
      rho_thresh - максимально допустимая разница rho (по нормали), px
    """
    seg_infos: list[dict[str, Any]] = []
    for (x1, y1, x2, y2) in segments:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        theta_dir = math.atan2(dy, dx)
        theta_dir = normalize_angle_theta(theta_dir)

        dx_dir = math.cos(theta_dir)
        dy_dir = math.sin(theta_dir)

        nx = -dy_dir
        ny = dx_dir

        rho = x1 * nx + y1 * ny

        t1 = x1 * dx_dir + y1 * dy_dir
        t2 = x2 * dx_dir + y2 * dy_dir
        tmin = min(t1, t2)
        tmax = max(t1, t2)
        seg_infos.append({
            'orig': (x1, y1, x2, y2),
            'theta': theta_dir,
            'rho': rho,
            'tmin': tmin,
            'tmax': tmax,
            'd': (dx_dir, dy_dir),
            'n': (nx, ny)
        })

    used = [False] * len(seg_infos)
    merged_segments: list[tuple[float, float, float, float]] = []

    angle_thresh = math.radians(angle_thresh_deg)

    for i, si in enumerate(seg_infos):
        if used[i]:
            continue
        cluster: list[int] = [i]
        used[i] = True
        for j in range(i + 1, len(seg_infos)):
            if used[j]:
                continue
            sj: dict[str, Any] = seg_infos[j]

            da = abs(si['theta'] - sj['theta'])
            da = min(da, math.pi - da)
            if da <= angle_thresh and abs(si['rho'] - sj['rho']) <= rho_thresh:
                cluster.append(j)
                used[j] = True

        tmin: float = min(seg_infos[k]['tmin'] for k in cluster)
        tmax: float = max(seg_infos[k]['tmax'] for k in cluster)

        theta_avg: float = sum(seg_infos[k]['theta'] for k in cluster) / len(cluster)
        theta_avg = normalize_angle_theta(theta_avg)
        dx_dir: float = math.cos(theta_avg)
        dy_dir: float = math.sin(theta_avg)
        nx = -dy_dir
        ny = dx_dir
        rho_avg: float = sum(seg_infos[k]['rho'] for k in cluster) / len(cluster)
        if extend_px > 0.0:
            tmin -= extend_px
            tmax += extend_px

        x_start: float = dx_dir * tmin + nx * rho_avg
        y_start: float = dy_dir * tmin + ny * rho_avg
        x_end: float = dx_dir * tmax + nx * rho_avg
        y_end: float = dy_dir * tmax + ny * rho_avg

        merged_segments.append((x_start, y_start, x_end, y_end))

    return merged_segments


def _point_segment_distance(px: float, py: float, seg: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = seg
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _extend_segment(seg: tuple[float, float, float, float], extend: float) -> tuple[float, float, float, float]:
    if extend <= 0:
        return seg
    x1, y1, x2, y2 = seg
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return seg
    ex = extend * dx / length
    ey = extend * dy / length
    return (x1 - ex, y1 - ey, x2 + ex, y2 + ey)


def _segment_length(seg: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = seg
    return math.hypot(x2 - x1, y2 - y1)


def intersection_point(seg1: tuple[float, float, float, float],
                       seg2: tuple[float, float, float, float],
                       eps: float = 1e-9,
                       gap_tol: float = 0.0,
                       extend: float = 0.0) -> tuple[float, float] | None:
    """
    Возвращает точку пересечения (x,y) если отрезки пересекаются (включая в концевых точках),
    иначе None.
    seg = (x1,y1,x2,y2) - могут быть float
    """
    x1, y1, x2, y2 = _extend_segment(seg1, extend)
    x3, y3, x4, y4 = _extend_segment(seg2, extend)

    # Решение векторно: p + t r  and q + u s
    r: tuple[float, float] = (x2 - x1, y2 - y1)
    s: tuple[float, float] = (x4 - x3, y4 - y3)

    r_cross_s = r[0] * s[1] - r[1] * s[0]
    q_p: tuple[float, float] = (x3 - x1, y3 - y1)
    q_p_cross_r = q_p[0] * r[1] - q_p[1] * r[0]

    if abs(r_cross_s) < eps:
        return None

    t: float = (q_p[0] * s[1] - q_p[1] * s[0]) / r_cross_s
    u: float = (q_p[0] * r[1] - q_p[1] * r[0]) / r_cross_s

    ix: float = x1 + t * r[0]
    iy: float = y1 + t * r[1]

    tol: float = 1e-7
    if -tol <= t <= 1 + tol and -tol <= u <= 1 + tol:
        return (ix, iy)
    if gap_tol > 0.0:
        if (_point_segment_distance(ix, iy, seg1) <= gap_tol and
                _point_segment_distance(ix, iy, seg2) <= gap_tol):
            return (ix, iy)
    return None


def cluster_points(points: Iterable[Tuple[float, float]], cluster_dist: float = 10.0) -> list[Tuple[float, float]]:
    if cluster_dist <= 0:
        return list[Tuple[float, float]](points)
    clusters: list[list[float]] = []  # [cx, cy, count]
    dist_sq = float(cluster_dist) * float(cluster_dist)
    for (x, y) in points:
        placed = False
        for cluster in clusters:
            cx, cy, count = cluster
            if (x - cx) ** 2 + (y - cy) ** 2 <= dist_sq:
                new_count = count + 1
                cluster[0] = (cx * count + x) / new_count
                cluster[1] = (cy * count + y) / new_count
                cluster[2] = new_count
                placed = True
                break
        if not placed:
            clusters.append([x, y, 1])
    return [(float(cx), float(cy)) for cx, cy, _ in clusters]


def count_intersections(image_path: str,
                        canny_thresh1=50, canny_thresh2=150,
                        hough_threshold=40, minLineLength=0, maxLineGap=10,
                        angle_merge_deg=5.0, rho_merge_px=20.0,
                        cluster_dist=14.0,
                        visualize=True,
                        out_vis_path="intersections_vis.png",
                        apply_clahe=True,
                        apply_blur=True,
                        use_auto_scaling=True,
                        auto_base_diag=1500.0,
                        segment_extend_px=6.0,
                        gap_tol_px=10.0,
                        gap_tol_ratio=0.18,
                        return_raw_points=False):
    img: np.ndarray | None = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if apply_clahe:
        clahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    if apply_blur:
        gray: np.ndarray = cv2.GaussianBlur(gray, (3, 3), 0)


    edges: np.ndarray = cv2.Canny(gray, canny_thresh1, canny_thresh2, apertureSize=3)

    h, w = gray.shape[:2]
    if use_auto_scaling and auto_base_diag > 0:
        diag: float = float((h ** 2 + w ** 2) ** 0.5)
        scale: float = diag / auto_base_diag
        scaled_cluster_dist: float = float(cluster_dist) * scale
        scaled_rho_merge: float = float(rho_merge_px) * scale

        # Если minLineLength не задан (0), подбераю от размера изображения
        scaled_min_line_length: int = int(max(minLineLength, 0.05 * max(h, w)))
        scaled_segment_extend: float = float(segment_extend_px) * scale
        scaled_gap_tol: float = float(gap_tol_px) * scale
    else:
        scaled_cluster_dist: float = cluster_dist
        scaled_rho_merge: float = rho_merge_px
        scaled_min_line_length: int = minLineLength
        scaled_segment_extend: float = segment_extend_px
        scaled_gap_tol: float = gap_tol_px

    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=hough_threshold,
                            minLineLength=scaled_min_line_length,
                            maxLineGap=maxLineGap)

    if lines is None:
        if visualize:
            cv2.imwrite(out_vis_path, img) # type: ignore
        return 0, [], []

    segments: list[tuple[float, float, float, float]] = [tuple(map(float, line[0])) for line in lines]

    merged: list[tuple[float, float, float, float]] = merge_collinear_segments(segments,
                                      angle_thresh_deg=angle_merge_deg,
                                      rho_thresh=scaled_rho_merge,
                                      extend_px=0.0)

    pts: list[tuple[float, float]] = []
    for a, b in combinations(merged, 2):
        gap_tol = max(scaled_gap_tol,
                      gap_tol_ratio * min(_segment_length(a), _segment_length(b)))
        p: tuple[float, float] | None = intersection_point(a, b,
                                                          gap_tol=gap_tol,
                                                          extend=scaled_segment_extend)
        if p is not None:
            pts.append(p)

    clustered: list[tuple[float, float]] = cluster_points(pts, cluster_dist=scaled_cluster_dist)

    if visualize:
        vis: np.ndarray = img.copy() # type: ignore
        for (x1, y1, x2, y2) in merged:
            cv2.line(vis, (int(round(x1)), int(round(y1))),
                     (int(round(x2)), int(round(y2))), (200, 200, 200), 2) # type: ignore
        for (x, y) in clustered:
            cv2.circle(vis, (int(round(x)), int(round(y))), 6, (0, 0, 255), -1) # type: ignore
        cv2.imwrite(out_vis_path, vis) # type: ignore

    if return_raw_points:
        return len(clustered), merged, clustered, pts # type: ignore
    return len(clustered), merged, clustered # type: ignore

