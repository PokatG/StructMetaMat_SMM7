# optimize_nodes_fd.py
from __future__ import annotations
import argparse
import numpy as np
import time

from homog_251007 import compute_C_from_ABC
from objective_minimax import J_minimax_Cii

def mod1(x: np.ndarray) -> np.ndarray:
    """Map coordinates to [0,1) periodically."""
    return np.mod(x, 1.0)

def evaluate(A, B, C, *, t, homog_kwargs):
    Cmat, vf = compute_C_from_ABC(A, B, C, out_prefix="opt_tmp", save_outputs=False, **homog_kwargs)
    C11, C22, C33 = float(Cmat[0,0]), float(Cmat[1,1]), float(Cmat[2,2])
    J, alpha = J_minimax_Cii(C11, C22, C33, t=t)
    return J, np.array([C11, C22, C33]), alpha, vf

def fd_gradient(p: np.ndarray, base_J: float, *, delta: float, t: float, homog_kwargs):
    """
    Forward finite-difference gradient for 9 params (A,B,C coords).
    p shape (9,).
    """
    g = np.zeros_like(p)
    # unpack
    A0, B0, C0 = p[0:3], p[3:6], p[6:9]

    for i in range(9):
        pp = p.copy()
        pp[i] += delta
        pp = mod1(pp)
        A, B, C = pp[0:3], pp[3:6], pp[6:9]
        Ji, _, _, _ = evaluate(A, B, C, t=t, homog_kwargs=homog_kwargs)
        g[i] = (Ji - base_J) / delta
    return g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", type=float, nargs=3, required=True)
    ap.add_argument("--B", type=float, nargs=3, required=True)
    ap.add_argument("--C", type=float, nargs=3, required=True)

    ap.add_argument("--t", type=float, default=0.002)
    ap.add_argument("--delta", type=float, default=0.002, help="FD step in design space (in [0,1])")
    ap.add_argument("--eta", type=float, default=0.05, help="Gradient descent step size")
    ap.add_argument("--iters", type=int, default=5)

    # keep homog settings fixed
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--nz", type=int, default=20)
    ap.add_argument("--n_points", type=int, default=3)
    ap.add_argument("--beam_diam", type=float, default=0.06)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--k", type=float, default=100.0)
    ap.add_argument("--hash_cell", type=float, default=0.08)

    args = ap.parse_args()

    homog_kwargs = dict(
        nx=args.nx, ny=args.ny, nz=args.nz,
        n_points=args.n_points,
        beam_diam=args.beam_diam,
        beta=args.beta, thresh=args.thresh,
        k=args.k, hash_cell=args.hash_cell,
        verbose=0,
    )

    p = np.array(list(args.A) + list(args.B) + list(args.C), dtype=float)
    p = mod1(p)

    best = None

    for it in range(1, args.iters + 1):
        t0 = time.perf_counter()
        A, B, C = p[0:3], p[3:6], p[6:9]
        J0, Cdiag, alpha, vf = evaluate(A, B, C, t=args.t, homog_kwargs=homog_kwargs)

        g = fd_gradient(p, J0, delta=args.delta, t=args.t, homog_kwargs=homog_kwargs)

        # Normalize gradient to avoid crazy steps (simple safeguard)
        gn = np.linalg.norm(g)
        if gn > 0:
            step = args.eta * g / gn
        else:
            step = 0.0 * g

        p_new = mod1(p - step)

        dt = time.perf_counter() - t0

        print(f"\n>>> starting iter {it}/{args.iters}")
        print(f"\n--- iter {it}/{args.iters} ---  ({dt:.2f}s)")
        print(f"J={J0:.8g}  vf={vf:.6f}")
        print(f"C11,C22,C33={Cdiag}")
        print(f"alpha={alpha}")
        print(f"|grad|={gn:.3e}  step_norm={np.linalg.norm(step):.3e}")
        print(f"A={A}  B={B}  C={C}")

        if best is None or J0 < best["J"]:
            best = dict(J=J0, p=p.copy(), Cdiag=Cdiag.copy(), alpha=alpha.copy(), vf=vf)

        p = p_new

    print("\n=== BEST FOUND ===")
    A, B, C = best["p"][0:3], best["p"][3:6], best["p"][6:9]
    print(f"J_best={best['J']:.8g}  vf={best['vf']:.6f}")
    print(f"C11,C22,C33_best={best['Cdiag']}")
    print(f"alpha_best={best['alpha']}")
    print(f"A_best={A}  B_best={B}  C_best={C}")

if __name__ == "__main__":
    main()
