# evaluate_minimax_inprocess.py
from __future__ import annotations
import argparse
import numpy as np

from homog_251007 import compute_C_from_ABC
from objective_minimax import J_minimax_Cii

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", type=float, nargs=3, required=True)
    ap.add_argument("--B", type=float, nargs=3, required=True)
    ap.add_argument("--C", type=float, nargs=3, required=True)
    ap.add_argument("--t", type=float, default=0.002)

    # a pár běžných nastavení, co chceš v iteracích držet konstantní:
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

    Cmat, vf, maps = compute_C_from_ABC(
        args.A, args.B, args.C,
        out_prefix="inprocess_tmp",
        save_outputs=False,      # important for speed
        nx=args.nx, ny=args.ny, nz=args.nz,
        n_points=args.n_points,
        beam_diam=args.beam_diam,
        beta=args.beta, thresh=args.thresh,
        k=args.k, hash_cell=args.hash_cell,
        verbose=0,
        no_progress=True,
        return_energy=1,
    )

    C11, C22, C33 = float(Cmat[0,0]), float(Cmat[1,1]), float(Cmat[2,2])
    J, alphas = J_minimax_Cii(C11, C22, C33, t=args.t)

    print("\n=== In-process results ===")
    print(f"vf≈{vf:.6f}")
    print(f"C11={C11:.7g}, C22={C22:.7g}, C33={C33:.7g}")
    print(f"alpha=[{alphas[0]:.6f},{alphas[1]:.6f},{alphas[2]:.6f}]")
    print(f"J={J:.7g}")
    # print(maps.keys())

    print("maps keys:", None if maps is None else list(maps.keys()))
    if maps is not None:
        for name, f in maps.items():
            print(name, "len =", len(f.vector().get_local()) if f is not None else None)

if __name__ == "__main__":
    main()
