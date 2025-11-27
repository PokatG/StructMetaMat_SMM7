import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis, norm, lognorm, probplot, kstest
import os
from datetime import datetime
import pandas as pd

# ---------- Nastavení výstupních složek ----------
output_folder_png = "Figures/Analyzing_Input_Data/png"
output_folder_svg = "Figures/Analyzing_Input_Data/svg"
os.makedirs(output_folder_png, exist_ok=True)
os.makedirs(output_folder_svg, exist_ok=True)

# Aktuální čas do názvů souborů
current_time = datetime.now().strftime("%y%m%d%H%M")

# ---------- Cesta k datům ----------
file_path = '/home/gb/Documents/PhD/Projects/StructMetaMat/Implicit_03/sweepC_summary.csv'

# ---------- Načtení CSV: sloupce 30 až předposlední ----------
def load_data(file_path: str) -> np.ndarray:
    """
    Načte CSV s hlavičkou; vrátí pouze sloupce 30..předposlední (0-index 29:-1)
    jako numpy float32 pole tvaru (N, 21) — typicky C11..C66 (Voigt blok).
    """
    df = pd.read_csv(file_path)
    if df.shape[1] < 32:
        raise ValueError(
            f"Soubor má jen {df.shape[1]} sloupců — potřebuji min. 32, "
            "aby existoval rozsah 30..předposlední."
        )
    Y = df.iloc[:, 31:-1].copy()
    for c in Y.columns:
        Y[c] = pd.to_numeric(Y[c], errors="coerce")
    Y = Y.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return Y.values.astype(np.float32)

data = load_data(file_path)

# ---------- Pomocné funkce (normal/lognormal fit + grafy) ----------
def plot_normal_fit_evaluation(column_data, row, col, time_str):
    column_data = np.asarray(column_data, dtype=float)
    column_data = column_data[np.isfinite(column_data)]
    if column_data.size < 5:
        return

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist(column_data, bins=30, density=True, alpha=0.6, edgecolor='black')

    mean, std = float(np.mean(column_data)), float(np.std(column_data, ddof=0))
    if std <= 0:
        std = 1e-12
    x = np.linspace(column_data.min(), column_data.max(), 200)
    pdf = norm.pdf(x, mean, std)
    ax[0].plot(x, pdf, 'r-', label='Normal PDF')

    num_samples = len(column_data)
    ax[0].legend([f'Normal PDF',
                  f'{num_samples} samples, Mean: {mean:.3g}, Std Dev: {std:.3g}'],
                 loc='upper right')
    ax[0].set_title(r'$\mathbb{{C}}_{{{},{}}}$: Histogram and Normal PDF'.format(row+1, col+1), fontsize=10)
    ax[0].grid(False)

    try:
        ks_stat, p_value = kstest(column_data, 'norm', args=(mean, std))
    except Exception:
        ks_stat, p_value = np.nan, np.nan

    probplot(column_data, dist="norm", sparams=(mean, std), plot=ax[1])
    ax[1].grid(True)
    ax[1].legend([f'K-S stat: {ks_stat:.4f}, p-value: {p_value:.4f}', 'Normal distribution'],
                 loc='upper left')
    ax[1].set_title(r'$\mathbb{{C}}_{{{},{}}}$: Normal Q-Q Plot'.format(row+1, col+1), fontsize=10)

    plt.tight_layout()
    file_name = f'Normal_Hist_C_{row+1}{col+1}_{time_str}'
    plt.savefig(f'{output_folder_png}/{file_name}.png', format='png')
    plt.savefig(f'{output_folder_svg}/{file_name}.svg', format='svg')
    plt.close(fig)


def plot_lognormal_with_fit_evaluation(column_data, row, col, time_str):
    column_data = np.asarray(column_data, dtype=float)
    column_data = column_data[np.isfinite(column_data)]
    column_data = column_data[column_data > 0]  # lognorm vyžaduje > 0
    if column_data.size < 5:
        return

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist(column_data, bins=30, density=True, alpha=0.6, edgecolor='black')

    shape, loc, scale = lognorm.fit(column_data, floc=0)
    x = np.linspace(column_data.min(), column_data.max(), 200)
    pdf = lognorm.pdf(x, shape, loc=loc, scale=scale)
    ax[0].plot(x, pdf, 'r-', label='Lognormal PDF')

    mean = float(np.mean(column_data))
    std_dev = float(np.std(column_data, ddof=0))
    num_samples = len(column_data)
    ax[0].legend([f'Lognormal PDF',
                  f'{num_samples} samples, Mean: {mean:.3g}, Std Dev: {std_dev:.3g}'],
                 loc='upper right')
    ax[0].set_title(r'$\mathbb{{C}}_{{{},{}}}$: Histogram and Lognormal PDF'.format(row+1, col+1), fontsize=10)
    ax[0].grid(False)

    try:
        ks_stat, p_value = kstest(column_data, 'lognorm', args=(shape, loc, scale))
    except Exception:
        ks_stat, p_value = np.nan, np.nan

    probplot(column_data, dist="lognorm", sparams=(shape,), plot=ax[1])
    ax[1].grid(True)
    ax[1].legend([f'K-S stat: {ks_stat:.4f}, p-value: {p_value:.4f}', 'Lognormal distribution'],
                 loc='upper left')
    ax[1].set_title(r'$\mathbb{{C}}_{{{},{}}}$: Lognormal Q-Q Plot'.format(row+1, col+1), fontsize=10)

    plt.tight_layout()
    file_name = f'Lognormal_Hist_C_{row+1}{col+1}_{time_str}'
    plt.savefig(f'{output_folder_png}/{file_name}.png', format='png')
    plt.savefig(f'{output_folder_svg}/{file_name}.svg', format='svg')
    plt.close(fig)

# ---------- 6×6 mřížka (21 pozic, horní trojúhelník včetně diagonály) ----------
histogram_positions = [
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (0, 2), (1, 3), (2, 4), (3, 5),
    (0, 3), (1, 4), (2, 5),
    (0, 4), (1, 5),
    (0, 5)
]

if data.shape[1] < len(histogram_positions):
    raise ValueError(f"Očekávám {len(histogram_positions)} sloupců (C_ij), ale mám {data.shape[1]}.")

data_targets = data

# ---------- Společný grid histogramů ----------
fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(15, 15))
axes = axes.flatten()

for idx, (row, col) in enumerate(histogram_positions):
    ax = axes[row * 6 + col]
    hist_data = data_targets[:, idx]
    hist_data = hist_data[np.isfinite(hist_data)]

    if hist_data.size == 0:
        ax.axis('off')
        continue

    skew_val = skew(hist_data)
    kurt_val = kurtosis(hist_data)

    ax.hist(hist_data, bins=30, edgecolor='black')
    ax.set_title(r'$\mathbb{{C}}_{{{},{}}}$'.format(row+1, col+1), fontsize=10)
    ax.legend([f'Skewness: {skew_val:.2f}\nKurtosis: {kurt_val:.2f}'], loc='upper right')

for i in range(len(axes)):
    if (i // 6, i % 6) not in histogram_positions:
        axes[i].axis('off')

plt.tight_layout()
combined_file_name = f'Hist_Combined_C_{current_time}'
plt.savefig(f'{output_folder_png}/{combined_file_name}.png', format='png')
plt.savefig(f'{output_folder_svg}/{combined_file_name}.svg', format='svg')
plt.show()

# ---------- Individuální fitování ----------
for idx, (row, col) in enumerate(histogram_positions):
    col_data = data_targets[:, idx]
    col_data = col_data[np.isfinite(col_data)]
    if col_data.size == 0:
        continue

    positive_data = col_data[col_data > 0]
    if positive_data.size >= 5:
        plot_lognormal_with_fit_evaluation(positive_data, row, col, current_time)

    if col_data.size >= 5:
        plot_normal_fit_evaluation(col_data, row, col, current_time)
