# Voronoi_3D_Periodic_250925.py
import numpy as np, pyvoro, pathlib, math

# ---------- parameters (overridable from caller before reload) ----------
N_POINTS, R_MIN, R_MAX, SEED = 3, 0.1, 0.2, 8
BLOCKS                       = 4
REPL_MODE                    = "26"   # "none" | "6" | "26"
CUT_TO_UNIT                  = True   # clip segments to [0,1]^3
SHIFT_DIR                    = 0      # 0=no extra duplicate; ±1=±X, ±2=±Y, ±3=±Z (for ParaView)
OUT   = pathlib.Path("cells_obj"); OUT.mkdir(exist_ok=True)
TOL   = 1e-6                 # tolerance for coordinate matching (periodicity check)
USER_POINTS = None           # if not None: use this (M,3) in [0,1) instead of RNG
# -----------------------------------------------------------------------

def _wrap01(pts):
    """Wrap to [0,1) torus."""
    return np.mod(np.asarray(pts, float), 1.0)

def _dedup_tol(pts, tol=1e-12):
    """Optional dedup by hashing coordinates at ~tol grid."""
    if pts.shape[0] <= 1:
        return pts
    key = np.floor(pts / max(tol, 1e-12)).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(idx)]

def _get_base_points():
    """Return an (M,3) array in [0,1) using USER_POINTS or RNG."""
    global N_POINTS, USER_POINTS, SEED
    if USER_POINTS is not None:
        pts = _wrap01(USER_POINTS)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("USER_POINTS must be an (M,3) array.")
        pts = _dedup_tol(pts, tol=1e-10)
        if pts.shape[0] != N_POINTS:
            # Make it convenient: override N_POINTS to actual length
            print(f"[voronoi] INFO: overriding N_POINTS {N_POINTS} → {pts.shape[0]} to match USER_POINTS")
            N_POINTS = int(pts.shape[0])
        return pts
    # RNG fallback
    rng = np.random.default_rng(SEED)
    return rng.random((N_POINTS, 3))

def _sample_radii(n, rmin, rmax, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(rmin, rmax, n)

def _build_shifts(mode="6"):
    """
    mode:
      - "none": only center tile
      - "6":    center + 6 faces
      - "26":   center + faces + edges + corners (total 27 with center)
    """
    shifts = [np.zeros(3)]
    m = str(mode).lower()
    if m == "none":
        return shifts
    if m == "6":
        for ax in range(3):
            for s in (-1, 1):
                e = np.zeros(3); e[ax] = s
                shifts.append(e)
        return shifts
    if m == "26":
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    shifts.append(np.array([dx, dy, dz], float))
        return shifts
    raise ValueError('REPL_MODE must be "none", "6", or "26".')

def _clip_segment_to_unit_cube(p, q, tol=1e-12):
    """
    Liang–Barsky clip of segment p--q to [0,1]^3.
    Returns [] if no intersection, or [(a,b)] inside the cube.
    """
    p = np.asarray(p, float); q = np.asarray(q, float)
    d = q - p
    t0, t1 = 0.0, 1.0
    for ax in range(3):
        if abs(d[ax]) < tol:
            if p[ax] < -tol or p[ax] > 1.0 + tol:
                return []
            continue
        inv = 1.0/d[ax]
        t_enter = (0.0 - p[ax]) * inv
        t_exit  = (1.0 - p[ax]) * inv
        if t_enter > t_exit:
            t_enter, t_exit = t_exit, t_enter
        t0 = max(t0, t_enter); t1 = min(t1, t_exit)
        if t0 - t1 > tol:
            return []
    a = np.clip(p + t0*d, 0.0, 1.0)
    b = np.clip(p + t1*d, 0.0, 1.0)
    if np.linalg.norm(a - b) <= tol:
        return []
    return [(a, b)]

def get_periodic_edges():
    """
    Compute and return (verts, edges) on demand with current globals.
    verts: list[np.ndarray(3)]
    edges: list[(i,j)] using 1-based vertex indices
    """
    global N_POINTS, R_MIN, R_MAX, SEED, BLOCKS, REPL_MODE, CUT_TO_UNIT, SHIFT_DIR, OUT, TOL

    # 1) Base seeds and weights (radii)
    ctr = _get_base_points()                  # (N,3)
    rad = _sample_radii(len(ctr), R_MIN, R_MAX, SEED)  # (N,)
    w   = rad**2

    # 2) Compute periodic Voronoi with pyvoro
    cells = pyvoro.compute_voronoi(
        ctr, [[0, 1]]*3, BLOCKS, radii=w, periodic=[True, True, True]
    )

    # 3) Dedup vertices and collect edges (replicate + clip)
    verts, edges, vmap = [], set(), {}

    def vid(p, tol=1e-9):
        """Deduplicate vertex with rounding-based key; return 1-based index."""
        p = np.asarray(p, float)
        k = tuple(np.round(p/tol)*tol)
        if k not in vmap:
            vmap[k] = len(verts) + 1
            verts.append(p)
        return vmap[k]

    SHIFTS = _build_shifts(REPL_MODE)

    for cell in cells:
        base = np.asarray(cell["vertices"])
        for face in cell["faces"]:
            # pyvoro face structure can be dict or tuple (older versions) – handle both
            if isinstance(face, dict):
                loop = face["vertices"]
                off  = face.get("offset") or face.get("adjacent_cell_offset") or [0, 0, 0]
            else:
                # (cell_id, loop, offset)
                loop, off = face[1], face[2]
            poly = base[loop] + np.asarray(off, float)  # possibly outside [0,1]^3
            for p, q in zip(poly, np.roll(poly, -1, axis=0)):
                for s in SHIFTS:
                    ps, qs = p + s, q + s
                    segs = _clip_segment_to_unit_cube(ps, qs) if CUT_TO_UNIT else [(ps, qs)]
                    for a, b in segs:
                        i, j = vid(a), vid(b)
                        if i != j:
                            edges.add(tuple(sorted((i, j))))

    # 4) Optional extra duplicate (for ParaView continuity preview)
    if SHIFT_DIR:
        shift = np.zeros(3); shift[abs(SHIFT_DIR)-1] = math.copysign(1, SHIFT_DIR)
        dup = {i: len(verts) + i for i in range(1, len(verts) + 1)}
        verts += [v + shift for v in verts]
        edges |= {(dup[i], dup[j]) for (i, j) in edges}

    # 5) Periodicity sanity check (base faces)
    faces = {'x0': {}, 'x1': {}, 'y0': {}, 'y1': {}, 'z0': {}, 'z1': {}}
    def key2D(a, b): return (round(a/TOL)*TOL, round(b/TOL)*TOL)

    for v in verts:
        x, y, z = v
        if abs(x - 0.0) < TOL: faces['x0'][key2D(y, z)] = True
        if abs(x - 1.0) < TOL: faces['x1'][key2D(y, z)] = True
        if abs(y - 0.0) < TOL: faces['y0'][key2D(x, z)] = True
        if abs(y - 1.0) < TOL: faces['y1'][key2D(x, z)] = True
        if abs(z - 0.0) < TOL: faces['z0'][key2D(x, y)] = True
        if abs(z - 1.0) < TOL: faces['z1'][key2D(x, y)] = True

    def missing(fa, fb): return sum(1 for k in faces[fa] if k not in faces[fb])
    miss = {
        'x': missing('x0','x1') + missing('x1','x0'),
        'y': missing('y0','y1') + missing('y1','y0'),
        'z': missing('z0','z1') + missing('z1','z0')
    }
    print(f"REPL_MODE: {REPL_MODE} | CUT_TO_UNIT: {CUT_TO_UNIT}")
    print("Periodic check – missing opposite twins:", miss)

    # 6) OBJ export (clean edges only)
    with open(OUT / "edges_clean.obj", "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for i, j in sorted(edges):
            f.write(f"l {i} {j}\n")
    print("✅  edges_clean.obj stored in", OUT.resolve())

    return [np.asarray(v, float) for v in verts], sorted(edges)
