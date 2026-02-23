# optimize_nodes_true_semiadjoint.py
from __future__ import annotations
import argparse
import numpy as np
import time
import csv

from homog_251007 import compute_C_from_ABC, compute_rho0_from_ABC
from objective_minimax import J_minimax_Cii
from homog_engine import HomogEngine


def mod1(x): return np.mod(x, 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", type=float, nargs=3, required=True)
    ap.add_argument("--B", type=float, nargs=3, required=True)
    ap.add_argument("--C", type=float, nargs=3, required=True)

    ap.add_argument("--t", type=float, default=0.002)
    ap.add_argument("--delta", type=float, default=0.002)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=10)

    # fixed pipeline params
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--nz", type=int, default=20)
    ap.add_argument("--n_points", type=int, default=3)
    ap.add_argument("--beam_diam", type=float, default=0.06)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--k", type=float, default=100.0)
    ap.add_argument("--hash_cell", type=float, default=0.08)
    ap.add_argument("--qorder", type=int, default=1)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--repl_mode", type=int, default=26)
    ap.add_argument("--cut_to_unit", type=int, default=1)
    ap.add_argument("--eps_rest", type=float, default=0.03)

    args = ap.parse_args()

    engine = HomogEngine(nx=args.nx, ny=args.ny, nz=args.nz, pc="hypre", rtol=1e-8, verbose=False)


    geom_kwargs = dict(
        nx=args.nx, ny=args.ny, nz=args.nz,
        n_points=args.n_points, eps_rest=args.eps_rest,
        beam_diam=args.beam_diam,
        beta=args.beta, thresh=args.thresh,
        k=args.k, hash_cell=args.hash_cell,
        qorder=args.qorder,
        seed=args.seed, blocks=args.blocks, repl_mode=args.repl_mode, cut_to_unit=bool(args.cut_to_unit),
        no_progress=True, verbose=0
    )

    homog_kwargs = dict(
        out_prefix="opt_tmp",
        save_outputs=False,
        nx=args.nx, ny=args.ny, nz=args.nz,
        n_points=args.n_points,
        beam_diam=args.beam_diam,
        beta=args.beta, thresh=args.thresh,
        k=args.k, hash_cell=args.hash_cell,
        qorder=args.qorder,
        seed=args.seed, blocks=args.blocks, repl_mode=args.repl_mode, cut_to_unit=bool(args.cut_to_unit),
        eps_rest=args.eps_rest,
        verbose=0,
        no_progress=True,
        return_energy=1
    )

    p = mod1(np.array(list(args.A)+list(args.B)+list(args.C), float))
    best = None

    hist_path = "opt_history.csv"
    with open(hist_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter","J","vf","C11","C22","C33",
                    "Ax","Ay","Az","Bx","By","Bz","Cx","Cy","Cz",
                    "is_best"])

    for it in range(1, args.iters+1):
        t0 = time.perf_counter()
        A, B, C = p[0:3], p[3:6], p[6:9]

        # 1) one expensive run: C + maps
        Cmat, vf, maps = compute_C_from_ABC(A, B, C, engine=engine, **homog_kwargs)

        C11, C22, C33 = float(Cmat[0,0]), float(Cmat[1,1]), float(Cmat[2,2])
        J, alpha = J_minimax_Cii(C11, C22, C33, t=args.t)

        rho_base = maps["rho0"].vector().get_local()
        E11 = maps["E11"].vector().get_local()
        E22 = maps["E22"].vector().get_local()
        E33 = maps["E33"].vector().get_local()

        # 2) combined weight map (semi-adjoint)
        w_e = -(alpha[0]*E11 + alpha[1]*E22 + alpha[2]*E33)

        # 3) cheap geometry perturbations
        grad = np.zeros(9)
        for i in range(9):
            pp = p.copy()
            pp[i] += args.delta
            pp = mod1(pp)
            Ap, Bp, Cp = pp[0:3], pp[3:6], pp[6:9]

            rho_p, _, _ = compute_rho0_from_ABC(Ap, Bp, Cp, **geom_kwargs, reload_vor=False)
            drho = (rho_p - rho_base) / args.delta
            grad[i] = np.dot(w_e, drho)

        # normalize and step
        gn = np.linalg.norm(grad)
        step = (args.eta * grad/gn) if gn > 0 else 0.0*grad
        p_new = mod1(p - step)

        dt = time.perf_counter() - t0
        print(f"iter {it}/{args.iters}  dt={dt:.1f}s  J={J:.8g}  vf={vf:.5f}  C=[{C11:.6g},{C22:.6g},{C33:.6g}]  |g|={gn:.3e}")

        if best is None or J < best["J"]:
            best = dict(J=J, p=p.copy(), C=(C11,C22,C33), vf=vf, alpha=alpha.copy())

        is_best = int(best is not None and J == best["J"])

        with open(hist_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([it, J, vf, C11, C22, C33,
                        p[0], p[1], p[2],
                        p[3], p[4], p[5],
                        p[6], p[7], p[8],
                        is_best])


        p = p_new

    A, B, C = best["p"][0:3], best["p"][3:6], best["p"][6:9]
    print("\n=== BEST ===")
    print("J_best=", best["J"])
    print("C_best=", best["C"], "vf=", best["vf"], "alpha=", best["alpha"])
    print("A_best=", A, "B_best=", B, "C_best=", C)
    print("  A_new=", p_new[0:3], "B_new=", p_new[3:6], "C_new=", p_new[6:9])

if __name__ == "__main__":
    main()
