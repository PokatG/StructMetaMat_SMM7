# eval_minimax.py
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np

from objective_minimax import J_minimax_Cii

def load_C_tensor(out_prefix: str) -> np.ndarray:
    """
    homog_251007.py writes:
      <out>_Cbar_tensor.npy
    which should be 6x6 Voigt (tensorial shear ordering).
    """
    npy_path = Path(f"{out_prefix}_Cbar_tensor.npy")
    if not npy_path.exists():
        raise FileNotFoundError(f"Expected C tensor file not found: {npy_path.resolve()}")
    C = np.load(npy_path)
    C = np.asarray(C, dtype=float)
    if C.shape != (6, 6):
        raise ValueError(f"Unexpected C shape {C.shape}, expected (6,6). File: {npy_path}")
    return C

def run_homog(
    homog_path: Path,
    out_prefix: str,
    A: list[float], B: list[float], Cpt: list[float],
    nx: int, ny: int, nz: int,
    extra_args: list[str],
) -> None:
    """
    Calls homog_251007.py as a subprocess so we don't need to refactor it yet.
    """
    cmd = [
        sys.executable, str(homog_path),
        "--out-prefix", out_prefix,
        "--fixed-abc", "1",
        "--A", *map(str, A),
        "--B", *map(str, B),
        "--C", *map(str, Cpt),
        "--nx", str(nx),
        "--ny", str(ny),
        "--nz", str(nz),
    ] + extra_args

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser(description="Evaluate minimax objective based on C11,C22,C33 from homog_251007.py")
    ap.add_argument("--homog", type=str, default="homog_251007.py", help="Path to homog_251007.py")
    ap.add_argument("--out-prefix", type=str, default="eval_demo", help="Output prefix passed to homog")
    ap.add_argument("--A", type=float, nargs=3, default=[0.1,0.1,0.1])
    ap.add_argument("--B", type=float, nargs=3, default=[0.9,0.9,0.9])
    ap.add_argument("--C", type=float, nargs=3, default=[0.5,0.55,0.5])

    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--nz", type=int, default=20)

    ap.add_argument("--t", type=float, default=0.02, help="Smooth-min temperature (smaller -> closer to min)")
    ap.add_argument("--extra", type=str, nargs="*", default=[], help="Extra args forwarded to homog_251007.py")

    args = ap.parse_args()

    homog_path = Path(args.homog)
    if not homog_path.exists():
        raise FileNotFoundError(f"homog script not found: {homog_path.resolve()}")

    # 1) Run homogenization (produces out-prefix_Cbar_tensor.npy)
    run_homog(
        homog_path=homog_path,
        out_prefix=args.out_prefix,
        A=list(args.A), B=list(args.B), Cpt=list(args.C),
        nx=args.nx, ny=args.ny, nz=args.nz,
        extra_args=args.extra,
    )

    # 2) Load C and compute objective
    Cmat = load_C_tensor(args.out_prefix)
    C11, C22, C33 = float(Cmat[0,0]), float(Cmat[1,1]), float(Cmat[2,2])

    J, alphas = J_minimax_Cii(C11, C22, C33, t=args.t)

    print("\n=== Results ===")
    print(f"C11={C11:.6g}, C22={C22:.6g}, C33={C33:.6g}")
    print(f"smooth-min weights alpha = [a11,a22,a33] = {alphas}")
    print(f"Objective J = {-1} * smooth_min(C11,C22,C33) = {J:.6g}")
    print(f"(Lower J is better; maximizing worst-direction stiffness means minimizing J.)")

if __name__ == "__main__":
    main()
