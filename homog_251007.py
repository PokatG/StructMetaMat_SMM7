#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homogenization bricks (steps 1–4), Voronoi-driven geometry — fixed A/B/C seeds.
Filename: homog_251007.py

- Step 1: Voronoi → beams → φ(x) via fast NumPy rasterizer
- Step 2: smoothed s(φ) → E(x), ν const
- Step 3: periodic SPD elasticity with nullspace (no anchor), assemble A once
- Step 4: reuse solver for 6 RHS, build C̄ (Voigt)

Changes vs previous version:
- Allows explicit placement of the first three Voronoi seeds A, B, C via CLI
  (default A=(0.1,0.1,0.1), B=(0.9,0.9,0.9), C=(0.2,0.3,0.4)).
- These three points are inserted exactly (modulo periodic wrap). Remaining
  points (if any) are filled with Poisson-disk-like random picks on the torus
  respecting `--eps-rest`.
- Still periodic (REPL_MODE=6/26) and cut to 1×1×1 when requested.

Outputs:
  <out>_phi.xdmf, <out>_sphi.xdmf, <out>_w_E11.xdmf
  <out>_Cbar_tensor.csv/.npy (+ engineering .csv if enabled)
  <out>_summary.csv (params + 21 coeffs + time)
"""
import Voronoi_3D_Periodic_250925 as VOR
import argparse, importlib, sys, time, math, os
import numpy as np
import dolfin as df

# --------------------- PBC class (robust) ---------------------
class PeriodicBoundary(df.SubDomain):
    """
    Periodic BCs for an axis-aligned box [xmin,xmax]×[ymin,ymax]×[zmin,zmax].
    Unique masters: x=min; y=min (excluding x=min); z=min (excluding x=min or y=min).
    Slaves: x=max, y=max, z=max (including edges/corners).
    """
    def __init__(self, mesh=None, tol=None):
        super().__init__()
        if mesh is None:
            self.xmin, self.xmax = 0.0, 1.0
            self.ymin, self.ymax = 0.0, 1.0
            self.zmin, self.zmax = 0.0, 1.0
        else:
            X = mesh.coordinates()
            self.xmin, self.xmax = float(X[:,0].min()), float(X[:,0].max())
            self.ymin, self.ymax = float(X[:,1].min()), float(X[:,1].max())
            self.zmin, self.zmax = float(X[:,2].min()), float(X[:,2].max())
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin
        self.Lz = self.zmax - self.zmin
        self.tol = 1e-10 * max(self.Lx, self.Ly, self.Lz, 1.0) if tol is None else float(tol)

    def inside(self, x, on_boundary):
        return bool(
            on_boundary and (
                df.near(x[0], self.xmin, self.tol)
                or (df.near(x[1], self.ymin, self.tol) and not df.near(x[0], self.xmin, self.tol))
                or (df.near(x[2], self.zmin, self.tol)
                    and not df.near(x[0], self.xmin, self.tol)
                    and not df.near(x[1], self.ymin, self.tol))
            )
        )

    def map(self, x, y):
        if df.near(x[0], self.xmax, self.tol):
            y[0] = x[0] - self.Lx; y[1] = x[1];             y[2] = x[2]
        elif df.near(x[1], self.ymax, self.tol):
            y[0] = x[0];             y[1] = x[1] - self.Ly; y[2] = x[2]
        elif df.near(x[2], self.zmax, self.tol):
            y[0] = x[0];             y[1] = x[1];             y[2] = x[2] - self.Lz
        else:
            y[0] = x[0]; y[1] = x[1]; y[2] = x[2]

# ---------- tiny diagnostics ----------
def debug_pbc_faces(mesh, tol=None, max_print=3):
    X = mesh.coordinates()
    xmin, xmax = X[:,0].min(), X[:,0].max()
    ymin, ymax = X[:,1].min(), X[:,1].max()
    zmin, zmax = X[:,2].min(), X[:,2].max()
    if tol is None:
        L = max(xmax-xmin, ymax-ymin, zmax-zmin, 1.0)
        tol = 1e-10 * L

    on_xmin = np.where(np.isclose(X[:,0], xmin, atol=tol))[0]
    on_xmax = np.where(np.isclose(X[:,0], xmax, atol=tol))[0]
    on_ymin = np.where(np.isclose(X[:,1], ymin, atol=tol))[0]
    on_ymax = np.where(np.isclose(X[:,1], ymax, atol=tol))[0]
    on_zmin = np.where(np.isclose(X[:,2], zmin, atol=tol))[0]
    on_zmax = np.where(np.isclose(X[:,2], zmax, atol=tol))[0]

    print(f"  PBC diag: bbox x=({xmin:.6g},{xmax:.6g}) y=({ymin:.6g},{ymax:.6g}) z=({zmin:.6g},{zmax:.6g}), tol={tol:.2e}")
    print(f"  Face counts: x[min]={len(on_xmin)} x[max]={len(on_xmax)} | y[min]={len(on_ymin)} y[max]={len(on_ymax)} | z[min]={len(on_zmin)} z[max]={len(on_zmax)}")

    def show_samples(idx, label, axis):
        if len(idx) == 0: return
        print(f"   Sample {label} slave→master mappings (first {min(max_print, len(idx))}):")
        for ii in idx[:max_print]:
            x = X[ii]
            if axis == 0:
                y = np.array([x[0]-(xmax-xmin), x[1], x[2]])
            elif axis == 1:
                y = np.array([x[0], x[1]-(ymax-ymin), x[2]])
            else:
                y = np.array([x[0], x[1], x[2]-(zmax-zmin)])
            print(f"     x={x} -> y={y}")

    def _ss(idx): return list(idx)[:max_print]
    show_samples(on_xmax, "x=max", 0)
    show_samples(on_ymax, "y=max", 1)
    show_samples(on_zmax, "z=max", 2)

# ------------------- helpers: FE math -------------------
def strain(u): return df.sym(df.grad(u))

def stress(u, lam, mu):  # Hooke isotropic
    e = strain(u)
    return lam*df.tr(e)*df.Identity(3) + 2.0*mu*e


def lame_from_E_nu(E, nu):
    mu = E/(2.0*(1.0+nu))
    lam = E*nu/((1.0+nu)*(1.0-2.0*nu))
    return lam, mu


def domain_volume(mesh):
    return df.assemble(df.Constant(1.0)*df.dx(mesh))


def mean_vector(u, mesh):
    vol = domain_volume(mesh)
    return np.array([df.assemble(u[i]*df.dx(mesh))/vol for i in range(3)])


def dump_beams_obj(beams, path="beams_used.obj"):
    with open(path, "w") as f:
        vid = 1
        for (p0, p1, r) in beams:
            f.write(f"v {p0[0]:.9f} {p0[1]:.9f} {p0[2]:.9f}\n")
            f.write(f"v {p1[0]:.9f} {p1[1]:.9f} {p1[2]:.9f}\n")
            f.write(f"l {vid} {vid+1}\n")
            vid += 2
    print(f"  ✓ Dumped beams OBJ used for φ: {path}")


def probe_phi_on_beams(phi, beams, n_beams=8, n_samples=7):
    P = df.Point
    n_beams = min(n_beams, len(beams))
    vals = []
    for b in range(n_beams):
        p0, p1, r = beams[b]
        for t in np.linspace(0.05, 0.95, n_samples):
            p = (1-t)*p0 + t*p1
            vals.append(phi(P(p[0], p[1], p[2])))
    if len(vals):
        vals = np.array(vals)
        print(f"  φ on-beam samples: min={vals.min():.3f}, mean={vals.mean():.3f}, max={vals.max():.3f}")


def affine_field(label, x):
    if   label == "E11": return df.as_vector([x[0], 0*x[1], 0*x[2]])
    elif label == "E22": return df.as_vector([0*x[0], x[1], 0*x[2]])
    elif label == "E33": return df.as_vector([0*x[0], 0*x[1], x[2]])
    elif label == "E12": return df.as_vector([0.5*x[1], 0.5*x[0], 0*x[2]])  # ε12=1
    elif label == "E13": return df.as_vector([0.5*x[2], 0*x[1], 0.5*x[0]])  # ε13=1
    elif label == "E23": return df.as_vector([0*x[0], 0.5*x[2], 0.5*x[1]])  # ε23=1
    else: raise ValueError(label)

# ------------------- periodic sampling (Option C friendly) ---
def mod1(x): return x - np.floor(x)

def torus_delta(u):
    # componentwise signed shortest displacement in [-0.5, 0.5)
    return u - np.round(u)


def torus_dist(x, y):
    d = torus_delta(x - y)
    return np.linalg.norm(d)


def near_enough(x, others, eps):
    if not others: return True
    return min(torus_dist(x, p) for p in others) >= eps

# ---------- fast vectorized distances for rasterizer ----------
def _point_to_segments_sq(p, A, B):
    AP = p - A
    AB = B - A
    denom = (AB*AB).sum(axis=1) + 1e-30
    t = (AP*AB).sum(axis=1) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = A + t[:,None]*AB
    diff = p - closest
    return (diff*diff).sum(axis=1)


def _build_spatial_hash(beams, cell=0.08):
    h = float(cell)
    nx = max(1, int(np.ceil(1.0 / h)))
    ny = max(1, int(np.ceil(1.0 / h)))
    nz = max(1, int(np.ceil(1.0 / h)))
    H = {}
    for bi, (p0, p1, r) in enumerate(beams):
        lo = np.minimum(p0, p1) - r
        hi = np.maximum(p0, p1) + r
        lo = np.clip(lo, 0.0, 1.0)
        hi = np.clip(hi, 0.0, 1.0)
        i0, j0, k0 = np.floor(lo / h).astype(int)
        i1, j1, k1 = np.floor(hi / h).astype(int)
        i0 = max(i0, 0); j0 = max(j0, 0); k0 = max(k0, 0)
        i1 = min(i1, nx-1); j1 = min(j1, ny-1); k1 = min(k1, nz-1)
        for i in range(i0, i1+1):
            for j in range(j0, j1+1):
                for k in range(k0, k1+1):
                    H.setdefault((i,j,k), []).append(bi)
    return H, h, (nx, ny, nz)


def indicator_from_voronoi_edges_numpy_L2(mesh, beams, k=100.0, cell=0.03, qorder=1, progress=True):
    V = df.FunctionSpace(mesh, "CG", 1)
    phi = df.Function(V, name="phi")

    A = np.array([b[0] for b in beams], dtype=float)
    B = np.array([b[1] for b in beams], dtype=float)
    r = float(beams[0][2]) if beams else 0.0
    H, h, (nx, ny, nz) = _build_spatial_hash(beams, cell=cell)
    for key, lst in list(H.items()):
        H[key] = np.array(lst, dtype=int)

    def clamp_index(ii, N): return 0 if ii < 0 else (N-1 if ii >= N else ii)
    def key_of(p):
        ii = clamp_index(int(math.floor(p[0] / h)), nx)
        jj = clamp_index(int(math.floor(p[1] / h)), ny)
        kk = clamp_index(int(math.floor(p[2] / h)), nz)
        return (ii, jj, kk)

    neighbor_cache = {}
    neighbor_offsets = [(di,dj,dk) for di in (-1,0,1) for dj in (-1,0,1) for dk in (-1,0,1)]
    def neighbor_cands(idx):
        if idx in neighbor_cache: return neighbor_cache[idx]
        ii,jj,kk = idx; chunks=[]
        for di,dj,dk in neighbor_offsets:
            key = (ii+di, jj+dj, kk+dk)
            if 0 <= key[0] < nx and 0 <= key[1] < ny and 0 <= key[2] < nz:
                arr = H.get(key)
                if arr is not None and arr.size: chunks.append(arr)
        arr = np.unique(np.concatenate(chunks)) if chunks else np.empty(0, dtype=int)
        neighbor_cache[idx] = arr
        return arr

    if qorder == 1:
        wq = np.array([1.0])
        lam = np.array([[0.25, 0.25, 0.25, 0.25]])
    else:
        wq = np.array([0.25, 0.25, 0.25, 0.25])
        lam = np.array([
            [0.58541020, 0.13819660, 0.13819660, 0.13819660],
            [0.13819660, 0.58541020, 0.13819660, 0.13819660],
            [0.13819660, 0.13819660, 0.58541020, 0.13819660],
            [0.13819660, 0.13819660, 0.13819660, 0.58541020],
        ])

    dofmap = V.dofmap()
    phi_acc   = np.zeros(V.dim())
    mass_lump = np.zeros(V.dim())

    n_cells = mesh.num_cells()
    t0 = time.perf_counter(); next_tick = 0; tick_every = max(2000, n_cells//100)

    for ci, c in enumerate(df.cells(mesh)):
        verts = np.asarray(c.get_vertex_coordinates(), float).reshape((-1, 3))
        dofs  = dofmap.cell_dofs(c.index())
        a,b,c3,d = verts
        vol = abs(np.linalg.det(np.column_stack((b-a, c3-a, d-a)))) / 6.0
        Xq = lam @ verts

        fq = np.empty(len(wq))
        for iq, xq in enumerate(Xq):
            cand = neighbor_cands(key_of(xq))
            if cand.size == 0:
                fq[iq] = 0.0; continue
            dsq = _point_to_segments_sq(xq, A[cand], B[cand])
            best = math.sqrt(dsq.min()) - r
            fq[iq] = 1.0 / (1.0 + math.exp(k * best))

        for local_i, gi in enumerate(dofs):
            rhs_i = float((wq * fq * lam[:, local_i]).sum())
            m_i   = float((wq *        lam[:, local_i]).sum())
            phi_acc[gi]   += vol * rhs_i
            mass_lump[gi] += vol * m_i

        if progress and ci >= next_tick:
            done = (ci+1) / n_cells
            elapsed = time.perf_counter() - t0
            rate = (ci+1) / max(elapsed, 1e-9)
            remain = (n_cells - (ci+1)) / max(rate, 1e-9)
            bar = int(30 * done)
            sys.stdout.write(f"\r  painting φ: [{'#'*bar}{'.'*(30-bar)}] {100*done:5.1f}%  {ci+1}/{n_cells}  ETA {remain:5.1f}s")
            sys.stdout.flush()
            next_tick += tick_every
    if progress:
        sys.stdout.write("\r  painting φ: [##############################] 100.0%  done        \n")
        sys.stdout.flush()

    mass_lump[mass_lump == 0.0] = 1.0
    phi.vector().set_local(phi_acc / mass_lump)
    phi.vector().apply("insert")
    return phi


def _beams_from_voronoi_module(Vmodule, beam_diam):
    verts, edges = Vmodule.get_periodic_edges()
    r = 0.5 * float(beam_diam)
    beams = []
    for (i, j) in edges:  # 1-based indices
        p0 = np.asarray(verts[i-1], float)
        p1 = np.asarray(verts[j-1], float)
        beams.append((p0, p1, r))
    return beams

# ------------------- in-process API (for optimization) -------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="Homog bricks, Voronoi-driven geometry (fixed A/B/C)")
    ap.add_argument("--verbose", type=int, default=1, help="1=print steps, 0=quiet (optimization-friendly)")
    ap.add_argument("--return-energy", type=int, default=0,
                help="1=compute and return DG0 energy maps for E11/E22/E33 (for semi-adjoint)")

    # Flags
    ap.add_argument("--qorder", type=int, choices=[1,4], default=1, help="Quadrature points per tetra for φ painter")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    # Mesh
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--nz", type=int, default=20)
    ap.add_argument("--out-prefix", default="demo_251007")
    # Voronoi params
    ap.add_argument("--n-points", type=int, default=3, help="Total number of Voronoi seeds (>=3)")
    ap.add_argument("--r-min", type=float, default=0.1)
    ap.add_argument("--r-max", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--repl-mode", choices=["none", "6", "26"], default="26")
    ap.add_argument("--cut-to-unit", type=int, default=1)
    # Level-set / rasterization
    ap.add_argument("--beam-diam", type=float, default=0.06)
    ap.add_argument("--k", type=float, default=100.0, help="Sharpness for φ logistic")
    ap.add_argument("--hash-cell", type=float, default=0.08, help="Spatial hash cell size")
    # Smooth material field
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--E-solid", type=float, default=1.0)
    ap.add_argument("--E-void", type=float, default=1e-6)
    ap.add_argument("--nu", type=float, default=0.30)
    ap.add_argument("--save-w-all", type=int, default=0, help="1=save w for all 6 cases")
    ap.add_argument("--voigt-eng", type=int, default=1, help="Also write engineering-Voigt C (shears ×1/2)")
    # Solver backend options
    ap.add_argument("--pc", choices=["hypre","gamg"], default="hypre", help="AMG preconditioner backend")
    ap.add_argument("--rtol", type=float, default=1e-8, help="Krylov relative tolerance")
    # Fixed A/B/C
    ap.add_argument("--fixed-abc", type=int, default=1, help="If 1, use fixed A,B,C provided below")
    ap.add_argument("--A", type=float, nargs=3, metavar=("AX","AY","AZ"), default=[0.1, 0.1, 0.1])
    ap.add_argument("--B", type=float, nargs=3, metavar=("BX","BY","BZ"), default=[0.9, 0.9, 0.9])
    ap.add_argument("--C", type=float, nargs=3, metavar=("CX","CY","CZ"), default=[0.5, 0.55, 0.5])
    ap.add_argument("--eps-rest", type=float, default=0.03, help="Min torus distance when filling remaining points")

    # NEW: allow turning file output on/off (important for optimization loops)
    ap.add_argument("--save-outputs", type=int, default=1, help="1=write XDMF/OBJ/CSV, 0=silent (faster for loops)")

    return ap





def run_homog(args, engine=None):
    """
    Core routine. Runs steps 1-4 and returns:
      C_sym (6x6), (C_eng or None), vf (float), timings (float seconds)

    NOTE: respects args.save_outputs (0/1) to skip file writes.
    """
    out = args.out_prefix
    save_outputs = bool(int(getattr(args, "save_outputs", 1)))

    verbose = bool(int(getattr(args, "verbose", 1)))

    def vprint(*a, **k):
        if verbose:
            print(*a, **k)


    t_all = time.perf_counter()

    # ---------- Step 1: mesh + Voronoi edges + φ ----------
    vprint("=== Step 1: Mesh + Voronoi-driven level-set φ ===")
    mesh = df.UnitCubeMesh(args.nx, args.ny, args.nz)

    importlib.reload(VOR)

    # Configure Voronoi module
    VOR.N_POINTS    = args.n_points
    VOR.R_MIN       = args.r_min
    VOR.R_MAX       = args.r_max
    VOR.SEED        = args.seed
    VOR.BLOCKS      = args.blocks
    VOR.REPL_MODE   = args.repl_mode
    VOR.CUT_TO_UNIT = bool(args.cut_to_unit)

    # VOR.VERBOSE = bool(verbose)
    # VOR.EXPORT_OBJ = bool(save_outputs)      # když nic neukládáš, nevypisuj a nepiš obj
    # VOR.CHECK_PERIODICITY = bool(verbose)    # periodic check log jen když verbose
    VOR.VERBOSE = False
    VOR.EXPORT_OBJ = False
    VOR.CHECK_PERIODICITY = False



    # inject A/B/C + rest
    if int(args.fixed_abc) == 1:
        A = mod1(np.array(args.A, float))
        B = mod1(np.array(args.B, float))
        C = mod1(np.array(args.C, float))
        pts = [A, B, C]
        if args.n_points > 3:
            rng = np.random.default_rng(args.seed+123)
            while len(pts) < args.n_points:
                x = rng.random(3)
                if near_enough(x, pts, float(args.eps_rest)):
                    pts.append(mod1(x))
        VOR.USER_POINTS = np.vstack(pts)
        vprint(f"  Using fixed seeds: A={A}, B={B}, C={C}; total N={len(VOR.USER_POINTS)}")
    else:
        VOR.USER_POINTS = None

    verts, edges = VOR.get_periodic_edges()
    vprint(f"  Voronoi: |verts|={len(verts)}, |edges|={len(edges)}  (REPL_MODE={VOR.REPL_MODE}, CUT_TO_UNIT={VOR.CUT_TO_UNIT})")

    beams = _beams_from_voronoi_module(VOR, beam_diam=args.beam_diam)

    t1 = time.perf_counter()
    phi = indicator_from_voronoi_edges_numpy_L2(
        mesh, beams, k=args.k, cell=args.hash_cell,
        qorder=args.qorder, progress=(not args.no_progress)
    )
    vprint(f"  φ painting done in {time.perf_counter()-t1:.2f}s")

    phi.rename("phi", "levelset")
    if save_outputs:
        dump_beams_obj(beams, f"{out}_beams_used.obj")
    probe_phi_on_beams(phi, beams)

    vals = phi.compute_vertex_values(mesh)
    vprint(f"  φ: min={vals.min():.4f}, max={vals.max():.4f}, mean≈{vals.mean():.4f}")
    if save_outputs:
        with df.XDMFFile(mesh.mpi_comm(), f"{out}_phi.xdmf") as xdmf:
            xdmf.parameters["functions_share_mesh"] = True
            xdmf.parameters["flush_output"] = False
            xdmf.write(mesh); xdmf.write(phi)
        vprint(f"  ✓ Saved: {out}_phi.xdmf")

    # ---------- Step 2: smooth interpolation s(φ) ----------
    vprint("=== Step 2: Smooth material interpolation E(x) from φ ===")
    t2 = time.perf_counter()

    Vphi  = phi.function_space()
    s_phi = df.Function(Vphi)
    arr = phi.vector().get_local()
    arr = 1.0 / (1.0 + np.exp(-float(args.beta)*(arr - float(args.thresh))))
    arr = np.clip(arr, 0.0, 1.0)
    s_phi.vector().set_local(arr); s_phi.vector().apply("insert")

    svals = s_phi.compute_vertex_values(mesh)
    vprint(f"  s(φ): min={svals.min():.4f}, max={svals.max():.4f}, mean≈{svals.mean():.4f}")

    vol = domain_volume(mesh)
    vf  = df.assemble(s_phi*df.dx(mesh)) / vol
    vprint(f"  volume fraction (≈⟨s(φ)⟩): {vf:.4f}")

    if save_outputs:
        with df.XDMFFile(mesh.mpi_comm(), f"{out}_sphi.xdmf") as xdmf:
            xdmf.parameters["functions_share_mesh"] = True
            xdmf.parameters["flush_output"] = False
            xdmf.write(mesh); xdmf.write(s_phi)
        vprint(f"  ✓ Saved: {out}_sphi.xdmf")

    E_void  = df.Constant(args.E_void)
    E_solid = df.Constant(args.E_solid)
    nu      = df.Constant(args.nu)
    E_field = E_void + (E_solid - E_void) * s_phi
    lam, mu = lame_from_E_nu(E_field, nu)

    # ---------- DG0 space and container for maps ----------
    want_energy = bool(int(getattr(args, "return_energy", 0)))
    V0 = df.FunctionSpace(mesh, "DG", 0) if want_energy else None
    energy_maps = {}  # label -> DG0 Function

    rho0 = None
    if want_energy:
        rho0 = df.project(s_phi, V0)      # DG0 (cell-wise) version of material field
        rho0.rename("rho0", "rho0")
        energy_maps["rho0"] = rho0



    vprint(f"  smoothing/material done in {time.perf_counter()-t2:.2f}s")

    # --- If engine is provided, reuse cached FE structures for Step 3+4 ---
    if engine is not None:
        # IMPORTANT: engine must be built with same (nx,ny,nz)
        # Compute C and (optionally) energy maps using engine
        C_sym, mapsE = engine.homogenize(lam, mu, want_energy=want_energy, timings=None)

        # Add energy maps from engine (E11/E22/E33). Also keep rho0 from this run.
        if want_energy:
            # rho0 already computed in this run (project(s_phi,V0)) and saved to energy_maps
            # merge energies from engine:
            energy_maps["E11"] = mapsE.get("E11")
            energy_maps["E22"] = mapsE.get("E22")
            energy_maps["E33"] = mapsE.get("E33")

        total_dt = time.perf_counter() - t_all
        # Return early (skip old Step 3+4 below)
        return C_sym, None, float(vf), float(total_dt), (energy_maps if want_energy else None)


    # ---------- Step 3: Homogenization cell problem (SPD + nullspace) ----------
    vprint("=== Step 3: Homogenization cell problem (SPD + nullspace) ===")
    t3 = time.perf_counter()

    pbc = PeriodicBoundary(mesh)
    debug_pbc_faces(mesh)

    Vp = df.VectorFunctionSpace(mesh, "CG", 1, constrained_domain=pbc)
    w = df.TrialFunction(Vp)
    v = df.TestFunction(Vp)
    x = df.SpatialCoordinate(mesh)

    a_spd = df.inner(stress(w, lam, mu), strain(v)) * df.dx(mesh)
    vprint("  assembling SPD A (once)...", end="", flush=True)
    Aop = df.assemble(a_spd)
    vprint(" done.")

    rb = []
    for i in range(3):
        e = [0.0, 0.0, 0.0]; e[i] = 1.0
        tvec = df.interpolate(df.Constant(tuple(e)), Vp)
        rb.append(tvec.vector())
    nullspace = df.VectorSpaceBasis(rb)
    nullspace.orthonormalize()
    df.as_backend_type(Aop).set_nullspace(nullspace)

    from dolfin import PETScOptions
    PETScOptions.set("ksp_type", "cg")
    PETScOptions.set("ksp_rtol", args.rtol)
    if args.pc == "hypre":
        PETScOptions.set("pc_type", "hypre")
        PETScOptions.set("pc_hypre_type", "boomeramg")
        PETScOptions.set("pc_hypre_boomeramg_coarsen_type", "HMIS")
        PETScOptions.set("pc_hypre_boomeramg_interp_type", "ext+i")
        PETScOptions.set("pc_hypre_boomeramg_strong_threshold", 0.25)
        PETScOptions.set("pc_hypre_boomeramg_agg_nl", 1)
    else:
        PETScOptions.set("pc_type", "gamg")
        PETScOptions.set("pc_gamg_threshold", 0.02)
        PETScOptions.set("pc_gamg_square_graph", 2)
        PETScOptions.set("mg_levels_ksp_type", "chebyshev")
        PETScOptions.set("mg_levels_pc_type", "jacobi")

    solver = df.PETScKrylovSolver(mesh.mpi_comm(), "cg")
    solver.parameters["monitor_convergence"] = False
    solver.set_operator(Aop)

    b = df.Vector(mesh.mpi_comm())
    try:
        Aop.init_vector(b, 0)
    except Exception:
        df.as_backend_type(Aop).init_vector(b, 0)

    w_sol = df.Function(Vp)

    def solve_rhs(u_aff, label, save_w=False):
        L = - df.inner(stress(u_aff, lam, mu), strain(v)) * df.dx(mesh)
        b.zero()
        df.assemble(L, tensor=b)
        nullspace.orthogonalize(b)

        t0 = time.perf_counter()
        solver.solve(w_sol.vector(), b)
        dt = time.perf_counter() - t0

        sig = stress(w_sol + u_aff, lam, mu)

        # --- optional: cell-wise strain energy for semi-adjoint ---
        Ecell = None
        if want_energy and label in ("E11", "E22", "E33"):
            u_tot = w_sol + u_aff
            sig_tot = stress(u_tot, lam, mu)
            eps_tot = strain(u_tot)

            # energy density (per volume): 0.5 * eps:sigma
            # For ranking/sensitivity, 0.5 factor is irrelevant, but keep it.
            edens = 0.5 * df.inner(sig_tot, eps_tot)

            Ecell = df.Function(V0)
            # project into DG0 (cell-wise constant)
            Ecell.assign(df.project(edens, V0))
            Ecell.rename("Ecell", f"Ecell_{label}")


        Sbar = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                Sbar[i,j] = df.assemble(sig[i,j] * df.dx(mesh)) / vol

        mw  = mean_vector(w_sol, mesh)
        wL2 = df.assemble(df.inner(w_sol, w_sol)*df.dx(mesh))**0.5
        vprint(f"  {label}: CG+AMG {dt:.2f}s | ⟨σ⟩ diag="
              + " ".join(f"{Sbar[d,d]:.4e}" for d in range(3))
              + f" | ||w||={wL2:.3e} | ⟨w⟩=({mw[0]:+.1e},{mw[1]:+.1e},{mw[2]:+.1e})")

        if save_outputs and save_w:
            with df.XDMFFile(mesh.mpi_comm(), f"{out}_w_{label}.xdmf") as xdmf:
                xdmf.parameters["functions_share_mesh"] = True
                xdmf.parameters["flush_output"] = False
                xdmf.write(mesh)
                w_sol.rename("w", f"w_{label}")
                xdmf.write(w_sol)
        return w_sol, Sbar, Ecell

    # E11 first
    u_aff_E11 = affine_field("E11", x)
    # _, Sbar_E11 = solve_rhs(u_aff_E11, "E11", save_w=True)
    # w11, Sbar_E11, Ecell_E11 = solve_rhs(u_aff_E11, "E11", save_w=True)
    _, Sbar_E11, Ecell_E11 = solve_rhs(u_aff_E11, "E11", save_w=True)
    if want_energy and Ecell_E11 is not None:
        energy_maps["E11"] = Ecell_E11
    vprint(f"  SPD E11 done in {time.perf_counter()-t3:.2f}s")

    # ---------- Step 4: Full homogenized stiffness C̄ ----------
    vprint("=== Step 4: Full homogenized stiffness C̄ (Voigt 6×6, tensorial shear) ===")
    labels = ["E11","E22","E33","E12","E13","E23"]
    Ccols  = [np.array([Sbar_E11[0,0], Sbar_E11[1,1], Sbar_E11[2,2],
                        Sbar_E11[0,1], Sbar_E11[0,2], Sbar_E11[1,2]], float)]
    for lab in ["E22","E33","E12","E13","E23"]:
        u_aff = affine_field(lab, x)
        save_this = bool(args.save_w_all)

        _, Sbar, Ecell = solve_rhs(u_aff, lab, save_w=save_this)

        if want_energy and (Ecell is not None) and lab in ("E22", "E33"):
            energy_maps[lab] = Ecell

        Ccols.append(np.array([Sbar[0,0], Sbar[1,1], Sbar[2,2],
                               Sbar[0,1], Sbar[0,2], Sbar[1,2]], float))


    C_ten = np.column_stack(Ccols)
    C_sym = 0.5*(C_ten + C_ten.T)
    asym  = np.abs(C_ten - C_ten.T).max()

    vprint("  Max |C - C^T| (symmetry defect):", f"{asym:.3e}")
    vprint("  C̄ (tensorial Voigt [11,22,33,12,13,23]):")
    for i in range(6):
        vprint("   ", " ".join(f"{C_sym[i,j]: .4e}" for j in range(6)))

    C_eng = None
    if int(args.voigt_eng) == 1:
        Tinv = np.diag([1,1,1,0.5,0.5,0.5])
        C_eng = C_sym @ Tinv
        vprint("  C̄ (engineering Voigt; shears use γ=2ε):")
        for i in range(6):
            vprint("   ", " ".join(f"{C_eng[i,j]: .4e}" for j in range(6)))

    if save_outputs:
        np.savetxt(f"{out}_Cbar_tensor.csv", C_sym, delimiter=",")
        np.save(   f"{out}_Cbar_tensor.npy", C_sym)
        if int(args.voigt_eng) == 1 and C_eng is not None:
            np.savetxt(f"{out}_Cbar_engineering.csv", C_eng, delimiter=",")
            vprint(f"  ✓ Saved: {out}_Cbar_engineering.csv")
        vprint(f"  ✓ Saved: {out}_Cbar_tensor.csv (+ .npy)")

        # Summary (same as before)
        total_dt = time.perf_counter() - t_all
        C21 = [C_sym[i, j] for i in range(6) for j in range(i, 6)]
        label = (
            f"Voronoi(N={args.n_points}, rmin={args.r_min}, rmax={args.r_max}, seed={args.seed}, "
            f"blocks={args.blocks}, repl={args.repl_mode}, cut={int(args.cut_to_unit)}) | "
            f"lvlset(d={args.beam_diam}, k={args.k}, h={args.hash_cell}, beta={args.beta}, "
            f"thr={args.thresh}, vf={vf:.4f}) | mesh={args.nx}x{args.ny}x{args.nz} | q={args.qorder}"
            + ("" if int(args.fixed_abc)==0 else
               f" | ABC(A={tuple(mod1(np.array(args.A,float)))}, B={tuple(mod1(np.array(args.B,float)))}, C={tuple(mod1(np.array(args.C,float)))}, epsR={args.eps_rest})")
        )
        vprint(label)
        vprint("= " + ",".join(f"{c:.6e}" for c in C21) + f", time={total_dt:.2f}s")

        summary_path = f"{out}_summary.csv"
        header_cols = [
            "nx","ny","nz","qorder",
            "N_POINTS","R_MIN","R_MAX","SEED","BLOCKS","REPL_MODE","CUT_TO_UNIT",
            "BEAM_DIAM","K","HASH_CELL","BETA","THRESH","VF",
            "E_SOLID","E_VOID","NU",
            "FIXED_ABC","AX","AY","AZ","BX","BY","BZ","CX","CY","CZ","EPS_REST"
        ] + [f"C{i+1}" for i in range(21)] + ["TIME_S"]

        if int(args.fixed_abc)==1:
            AX,AY,AZ = (mod1(np.array(args.A,float))).tolist()
            BX,BY,BZ = (mod1(np.array(args.B,float))).tolist()
            CX,CY,CZ = (mod1(np.array(args.C,float))).tolist()
            FIXED_FLAG = 1
        else:
            AX=AY=AZ=BX=BY=BZ=CX=CY=CZ=""
            FIXED_FLAG = 0

        row_vals = [
            args.nx, args.ny, args.nz, args.qorder,
            args.n_points, args.r_min, args.r_max, args.seed, args.blocks, args.repl_mode, int(args.cut_to_unit),
            args.beam_diam, args.k, args.hash_cell, args.beta, args.thresh, float(vf),
            args.E_solid, args.E_void, args.nu,
            FIXED_FLAG, AX, AY, AZ, BX, BY, BZ, CX, CY, CZ, args.eps_rest
        ] + [float(v) for v in C21] + [float(total_dt)]

        write_header = not os.path.exists(summary_path)
        with open(summary_path, "a") as f:
            if write_header:
                f.write(",".join(map(str, header_cols)) + "\n")
            f.write(",".join(map(str, row_vals)) + "\n")
        vprint(f"  ✓ Appended summary row → {summary_path}")

    total_dt = time.perf_counter() - t_all
    if want_energy:
        return C_sym, C_eng, float(vf), float(total_dt), energy_maps
    else:
        return C_sym, C_eng, float(vf), float(total_dt), None






def compute_C_from_ABC(A, B, C, *, out_prefix="inprocess", save_outputs=False, engine=None, **kwargs):
    """
    Convenience wrapper for optimization:
      A,B,C: iterable length-3
      kwargs: any CLI-style settings, e.g. nx=20, ny=20, ...
    Returns:
      C_sym (6x6), vf (float)
    """
    ap = build_argparser()
    args = ap.parse_args([])  # start from defaults without CLI
    args.out_prefix = out_prefix
    args.save_outputs = 1 if save_outputs else 0

    # set A/B/C and force fixed-abc
    args.fixed_abc = 1
    args.A = list(map(float, A))
    args.B = list(map(float, B))
    args.C = list(map(float, C))

    # override any provided kwargs (nx, beta, etc.)
    for k, v in kwargs.items():
        if not hasattr(args, k.replace("-", "_")) and not hasattr(args, k):
            # allow both "n-points" and "n_points" style keys
            pass
        key = k.replace("-", "_")
        if hasattr(args, key):
            setattr(args, key, v)
        elif hasattr(args, k):
            setattr(args, k, v)
        else:
            raise AttributeError(f"Unknown setting '{k}' for homog args.")

    C_sym, _, vf, _, energy_maps = run_homog(args, engine=engine)
    return C_sym, vf, energy_maps

def compute_rho0_from_ABC(
    A, B, C, *,
    nx=20, ny=20, nz=20,
    n_points=3, eps_rest=0.03,
    beam_diam=0.06,
    beta=10.0, thresh=0.5,
    k=100.0, hash_cell=0.08,
    qorder=1,
    seed=8, blocks=4, repl_mode=26, cut_to_unit=True,
    no_progress=True,
    verbose=0,
    reload_vor=False,          # <-- NOVÉ: defaultně nere-loaduje
    r_min=0.1, r_max=0.2       # <-- NOVÉ: explicitně nastavujeme radii
):
    """
    Geometry-only pipeline:
    returns (rho0_dg0_numpy, mesh, V0) without homogenization solves.
    """

    # mesh (fixed)
    mesh = df.UnitCubeMesh(nx, ny, nz)

    # --- Voronoi config ---
    if reload_vor:
        importlib.reload(VOR)

    # nastav vždy parametry (bez ohledu na reload)
    VOR.N_POINTS    = n_points
    VOR.R_MIN       = r_min
    VOR.R_MAX       = r_max
    VOR.SEED        = seed
    VOR.BLOCKS      = blocks
    VOR.REPL_MODE   = str(repl_mode)
    VOR.CUT_TO_UNIT = bool(cut_to_unit)

    # tichý režim + žádný OBJ export (pro optimalizaci)
    VOR.VERBOSE = bool(verbose)
    VOR.EXPORT_OBJ = False
    VOR.CHECK_PERIODICITY = False

    # --- fixed seeds A/B/C + optional rest ---
    A = mod1(np.array(A, float))
    B = mod1(np.array(B, float))
    C = mod1(np.array(C, float))
    pts = [A, B, C]

    if n_points > 3:
        rng = np.random.default_rng(seed + 123)
        while len(pts) < n_points:
            x = rng.random(3)
            if near_enough(x, pts, float(eps_rest)):
                pts.append(mod1(x))

    VOR.USER_POINTS = np.vstack(pts)

    # Voronoi edges -> beams
    _verts, _edges = VOR.get_periodic_edges()
    beams = _beams_from_voronoi_module(VOR, beam_diam=beam_diam)

    # phi
    phi = indicator_from_voronoi_edges_numpy_L2(
        mesh, beams, k=k, cell=hash_cell, qorder=qorder,
        progress=(not no_progress)
    )
    phi.rename("phi", "levelset")

    # s(phi)
    Vphi  = phi.function_space()
    s_phi = df.Function(Vphi)
    arr = phi.vector().get_local()
    arr = 1.0 / (1.0 + np.exp(-float(beta)*(arr - float(thresh))))
    arr = np.clip(arr, 0.0, 1.0)
    s_phi.vector().set_local(arr)
    s_phi.vector().apply("insert")

    # DG0 projection
    V0 = df.FunctionSpace(mesh, "DG", 0)
    rho0 = df.project(s_phi, V0)

    return rho0.vector().get_local(), mesh, V0




# ================================ main ===================================
def main():
    ap = build_argparser()
    args = ap.parse_args()
    run_homog(args)


if __name__ == "__main__":
    main()