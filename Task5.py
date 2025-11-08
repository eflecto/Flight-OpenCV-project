# c3_radius50_proj.py
# Печатает ТОЛЬКО одно число (метры, 2 знака).
# 1 px = 1 см. Диаметр дрона = 50 см → радиус 25 px.

import sys, math, cv2, numpy as np
from heapq import heappush, heappop

RADIUS_PX = 25
EPS = 1.0  # аппроксимация контуров после дилатации

def orient(a,b,c): return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
def segs_cross(a,b,c,d):
    o1=orient(a,b,c); o2=orient(a,b,d); o3=orient(c,d,a); o4=orient(c,d,b)
    return (o1*o2<0) and (o3*o4<0)

def load_masks(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("0.00"); sys.exit(0)
    H, W = img.shape
    # исходные препятствия (чёрное → 255)
    _, obs_orig = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # расширяем препятствия на радиус (конфигурационное пространство)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RADIUS_PX*2+1, RADIUS_PX*2+1))
    obs_infl = cv2.dilate(obs_orig, k, 1)
    return img, obs_orig, obs_infl, W, H

def inflated_polys(obs_infl):
    cnts,_ = cv2.findContours(obs_infl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys=[]
    for c in cnts:
        if cv2.contourArea(c) < 10: 
            continue
        approx = cv2.approxPolyDP(c, EPS, True)
        poly=[(int(p[0][0]), int(p[0][1])) for p in approx]
        if len(poly)>=3:
            polys.append(poly)
    return polys

def segment_blocked(p,q, polys):
    # Запрещаем входить в ВНУТРЕННОСТЬ расширенных полигонов (касаться/идти вдоль ребра можно)
    for poly in polys:
        n=len(poly)
        # быстрое отсечение по bbox
        minx=min(p[0],q[0]); maxx=max(p[0],q[0])
        miny=min(p[1],q[1]); maxy=max(p[1],q[1])
        bx1=min(v[0] for v in poly); bx2=max(v[0] for v in poly)
        by1=min(v[1] for v in poly); by2=max(v[1] for v in poly)
        if maxx<bx1 or minx>bx2 or maxy<by1 or miny>by2:
            continue
        for i in range(n):
            a,b=poly[i], poly[(i+1)%n]
            if segs_cross(p,q,a,b): 
                return True
        mx,my=(p[0]+q[0])/2.0, (p[1]+q[1])/2.0
        if cv2.pointPolygonTest(np.array(poly, np.int32), (mx,my), False) > 0:
            return True
    return False

def border_blocked_by_original(p,q, obs_orig, W, H):
    # По условию С2 двигаться по границе нельзя ТАМ, где препятствия касаются края.
    # Проверяем исходную маску; касание в концах отрезка разрешаем.
    if p[1]==0 and q[1]==0:
        x1,x2=sorted([p[0],q[0]]); a=obs_orig[0, x1:x2+1].copy(); a[0]=a[-1]=0; return np.any(a>0)
    if p[1]==H-1 and q[1]==H-1:
        x1,x2=sorted([p[0],q[0]]); a=obs_orig[H-1, x1:x2+1].copy(); a[0]=a[-1]=0; return np.any(a>0)
    if p[0]==0 and q[0]==0:
        y1,y2=sorted([p[1],q[1]]); a=obs_orig[y1:y2+1, 0].copy(); a[0]=a[-1]=0; return np.any(a>0)
    if p[0]==W-1 and q[0]==W-1:
        y1,y2=sorted([p[1],q[1]]); a=obs_orig[y1:y2+1, W-1].copy(); a[0]=a[-1]=0; return np.any(a>0)
    return False

def solve(path):
    img, obs_orig, obs_infl, W, H = load_masks(path)
    polys = inflated_polys(obs_infl)

    start, goal = (0,0), (W-1, H-1)

    # Узлы: старт, финиш, все вершины расширенных препятствий
    nodes = [start, goal] + [v for poly in polys for v in poly]

    # АДАПТИВНЫЕ узлы на границе — проекции всех вершин на четыре стороны + углы
    verts = [v for poly in polys for v in poly] + [start, goal]
    for (x,y) in verts:
        nodes.append((x, 0));    nodes.append((x, H-1))
        nodes.append((0, y));    nodes.append((W-1, y))
    nodes += [(W-1,0), (0,H-1)]
    nodes = list(dict.fromkeys(nodes))  # уникальные

    # Граф видимости
    n = len(nodes)
    adj = [[] for _ in range(n)]
    for i in range(n):
        p = nodes[i]
        for j in range(i+1, n):
            q = nodes[j]
            if border_blocked_by_original(p, q, obs_orig, W, H):
                continue
            if not segment_blocked(p, q, polys):
                w = math.hypot(q[0]-p[0], q[1]-p[1])
                adj[i].append((j, w)); adj[j].append((i, w))

    # Дейкстра
    sid, gid = nodes.index(start), nodes.index(goal)
    INF = 1e18
    dist = [INF]*n; dist[sid] = 0.0
    pq = [(0.0, sid)]
    while pq:
        d,i = heappop(pq)
        if d != dist[i]: continue
        if i == gid: break
        for j,w in adj[i]:
            nd = d + w
            if nd < dist[j]:
                dist[j] = nd
                heappush(pq, (nd, j))

    return dist[gid] * 0.01  # метры

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "input.png"
    m = solve(path)
    print("0.00" if m >= 1e17 else f"{m:.2f}")

if __name__ == "__main__":
    main()
