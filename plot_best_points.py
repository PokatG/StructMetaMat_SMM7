# plot_best_points.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def draw_cube(ax):
    # 12 edges of unit cube
    edges = [
        ((0,0,0),(1,0,0)), ((0,1,0),(1,1,0)), ((0,0,1),(1,0,1)), ((0,1,1),(1,1,1)),
        ((0,0,0),(0,1,0)), ((1,0,0),(1,1,0)), ((0,0,1),(0,1,1)), ((1,0,1),(1,1,1)),
        ((0,0,0),(0,0,1)), ((1,0,0),(1,0,1)), ((0,1,0),(0,1,1)), ((1,1,0),(1,1,1)),
    ]
    for (x0,y0,z0),(x1,y1,z1) in edges:
        ax.plot([x0,x1],[y0,y1],[z0,z1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="opt_history.csv")
    ap.add_argument("--only-best", action="store_true", help="Plot only rows marked is_best=1")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.only_best:
        df = df[df["is_best"] == 1].copy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    draw_cube(ax)

    
    Ax, Ay, Az = df["Ax"].to_numpy(), df["Ay"].to_numpy(), df["Az"].to_numpy()
    Bx, By, Bz = df["Bx"].to_numpy(), df["By"].to_numpy(), df["Bz"].to_numpy()
    Cx, Cy, Cz = df["Cx"].to_numpy(), df["Cy"].to_numpy(), df["Cz"].to_numpy()

    # trajectories (lines)
    ax.plot(Ax, Ay, Az, label="A path")
    ax.plot(Bx, By, Bz, label="B path")
    ax.plot(Cx, Cy, Cz, label="C path")

    ax.scatter(Ax, Ay, Az, label="A")
    ax.scatter(Bx, By, Bz, label="B")
    ax.scatter(Cx, Cy, Cz, label="C")


    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(0,1)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("A,B,C trajectory in unit cube")

    plt.show()

if __name__ == "__main__":
    main()
