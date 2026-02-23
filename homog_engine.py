# homog_engine.py
from __future__ import annotations
import numpy as np
import dolfin as df

from timing_utils import Timings
from contextlib import contextmanager
# importuj svoje helpery: PeriodicBoundary, strain, stress, affine_field, mean_vector, domain_volume, lame_from_E_nu
# importuj svoje geometry: compute_phi_and_sphi_from_ABC nebo aspoň existující části

class _NullTimings:
    @contextmanager
    def section(self, name: str):
        yield

# ------------------- helpers: FE math -------------------
def strain(u): return df.sym(df.grad(u))

def stress(u, lam, mu):  # Hooke isotropic
    e = strain(u)
    return lam*df.tr(e)*df.Identity(3) + 2.0*mu*e

def affine_field(label, x):
    if   label == "E11": return df.as_vector([x[0], 0*x[1], 0*x[2]])
    elif label == "E22": return df.as_vector([0*x[0], x[1], 0*x[2]])
    elif label == "E33": return df.as_vector([0*x[0], 0*x[1], x[2]])
    elif label == "E12": return df.as_vector([0.5*x[1], 0.5*x[0], 0*x[2]])  # ε12=1
    elif label == "E13": return df.as_vector([0.5*x[2], 0*x[1], 0.5*x[0]])  # ε13=1
    elif label == "E23": return df.as_vector([0*x[0], 0.5*x[2], 0.5*x[1]])  # ε23=1
    else: raise ValueError(label)

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

class HomogEngine:
    def __init__(self, *, nx, ny, nz, pc="hypre", rtol=1e-8, verbose=False):
        self.verbose = bool(verbose)
        self.pc = pc
        self.rtol = rtol

        self.mesh = df.UnitCubeMesh(nx, ny, nz)
        self.vol = df.assemble(df.Constant(1.0) * df.dx(self.mesh))

        self.pbc = PeriodicBoundary(self.mesh)
        self.Vp = df.VectorFunctionSpace(self.mesh, "CG", 1, constrained_domain=self.pbc)
        self.V0 = df.FunctionSpace(self.mesh, "DG", 0)

        self.w = df.TrialFunction(self.Vp)
        self.v = df.TestFunction(self.Vp)
        self.x = df.SpatialCoordinate(self.mesh)

        # Placeholders set each iteration
        self.lam = None
        self.mu = None

        # solver & buffers
        self.Aop = None
        self.solver = None
        self.b = df.Vector(self.mesh.mpi_comm())
        self.w_sol = df.Function(self.Vp)

        self._setup_nullspace()
        self._setup_solver()

    def _setup_nullspace(self):
        rb = []
        for i in range(3):
            e = [0.0, 0.0, 0.0]; e[i] = 1.0
            tvec = df.interpolate(df.Constant(tuple(e)), self.Vp)
            rb.append(tvec.vector())
        self.nullspace = df.VectorSpaceBasis(rb)
        self.nullspace.orthonormalize()

    def _setup_solver(self):
        from dolfin import PETScOptions
        PETScOptions.set("ksp_type", "cg")
        PETScOptions.set("ksp_rtol", self.rtol)

        if self.pc == "hypre":
            PETScOptions.set("pc_type", "hypre")
            PETScOptions.set("pc_hypre_type", "boomeramg")
            PETScOptions.set("pc_hypre_boomeramg_coarsen_type", "HMIS")
            PETScOptions.set("pc_hypre_boomeramg_interp_type", "ext+i")
            PETScOptions.set("pc_hypre_boomeramg_strong_threshold", 0.25)
        else:
            PETScOptions.set("pc_type", "gamg")
            PETScOptions.set("pc_gamg_threshold", 0.02)
            PETScOptions.set("mg_levels_ksp_type", "chebyshev")
            PETScOptions.set("mg_levels_pc_type", "jacobi")

        self.solver = df.PETScKrylovSolver(self.mesh.mpi_comm(), "cg")
        self.solver.parameters["monitor_convergence"] = False

    def assemble_A(self, lam, mu, timings: Timings):
        """Assemble SPD matrix for current lam,mu. Reuse solver/nullspace/buffers."""
        self.lam, self.mu = lam, mu
        with timings.section("assemble_A"):
            a_spd = df.inner(stress(self.w, lam, mu), strain(self.v)) * df.dx(self.mesh)
            self.Aop = df.assemble(a_spd)
            df.as_backend_type(self.Aop).set_nullspace(self.nullspace)
            # init RHS vector compatibly
            try:
                self.Aop.init_vector(self.b, 0)
            except Exception:
                df.as_backend_type(self.Aop).init_vector(self.b, 0)
            self.solver.set_operator(self.Aop)

    def solve_case(self, u_aff, timings: Timings):
        """Solve one RHS, return (w_sol copy, Sbar 3x3)."""
        with timings.section("assemble_rhs"):
            L = - df.inner(stress(u_aff, self.lam, self.mu), strain(self.v)) * df.dx(self.mesh)
            self.b.zero()
            df.assemble(L, tensor=self.b)
            self.nullspace.orthogonalize(self.b)

        with timings.section("solve"):
            self.solver.solve(self.w_sol.vector(), self.b)

        # postproc sigma-bar
        with timings.section("post_sigma"):
            u_tot = self.w_sol + u_aff
            sig = stress(u_tot, self.lam, self.mu)
            Sbar = np.zeros((3,3))
            for i in range(3):
                for j in range(3):
                    Sbar[i,j] = df.assemble(sig[i,j] * df.dx(self.mesh)) / self.vol
        return self.w_sol, Sbar

    def energy_DG0(self, u_aff, timings: Timings):
        """Cell-wise energy density DG0 for this case (uses current w_sol already solved)."""
        with timings.section("energy_proj"):
            u_tot = self.w_sol + u_aff
            sig = stress(u_tot, self.lam, self.mu)
            eps = strain(u_tot)
            edens = 0.5 * df.inner(sig, eps)
            return df.project(edens, self.V0)  # later we can LocalSolver-optimize

    def homogenize(self, lam, mu, want_energy: bool, timings=None):
        """Run 6 loadcases, return C(6x6) and optional maps."""
        if timings is None:
            timings = _NullTimings()

        self.assemble_A(lam, mu, timings)

        maps = {}
        # E11
        u_aff = affine_field("E11", self.x)
        _, S11 = self.solve_case(u_aff, timings)
        if want_energy:
            maps["E11"] = self.energy_DG0(u_aff, timings)

        cols = [np.array([S11[0,0],S11[1,1],S11[2,2], S11[0,1],S11[0,2],S11[1,2]])]

        for lab in ["E22","E33","E12","E13","E23"]:
            u_aff = affine_field(lab, self.x)
            _, S = self.solve_case(u_aff, timings)
            if want_energy and lab in ("E22","E33"):
                maps[lab] = self.energy_DG0(u_aff, timings)
            cols.append(np.array([S[0,0],S[1,1],S[2,2], S[0,1],S[0,2],S[1,2]]))

        C = np.column_stack(cols)
        C = 0.5*(C + C.T)
        return C, maps
