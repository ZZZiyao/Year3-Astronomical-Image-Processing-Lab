import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial import distance

# read cateloge
filename = r"D:\aip\astronomical-image-processing\loop18_catalog.txt"
data = np.loadtxt(filename, skiprows=1)  # skip 1st row

# find x and y coordinates
filtered_data = data[data[:, 1] < 1200]
x_coords = filtered_data[:, 1]
y_coords = filtered_data[:, 2]

# random galaxy generation
x_min, x_max = x_coords.min(), x_coords.max()
y_min, y_max = y_coords.min(), y_coords.max()

# random coordinate generation
num_random = len(x_coords)
random_x_coords = np.random.uniform(x_min, x_max, num_random)
random_y_coords = np.random.uniform(y_min, y_max, num_random)

# calculate distance between galaxies
coords_data = np.column_stack((x_coords, y_coords))
coords_random = np.column_stack((random_x_coords, random_y_coords))

pairwise_distances_data = distance.cdist(coords_data, coords_data, 'euclidean')
pairwise_distances_random = distance.cdist(coords_random, coords_random, 'euclidean')
pairwise_distances_data_random = distance.cdist(coords_data, coords_random, 'euclidean')

# stats
bins = np.logspace(np.log10(0.1), np.log10(np.max(pairwise_distances_data)), num=1000)  
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# stats of num
DD_hist, _ = np.histogram(pairwise_distances_data[np.triu_indices(len(pairwise_distances_data), k=1)], bins=bins)
RR_hist, _ = np.histogram(pairwise_distances_random[np.triu_indices(len(pairwise_distances_random), k=1)], bins=bins)
DR_hist, _ = np.histogram(pairwise_distances_data_random, bins=bins)

# Landy-Szalay estimator
n_data_pairs = len(coords_data) * (len(coords_data) - 1) / 2
n_random_pairs = len(coords_random) * (len(coords_random) - 1) / 2
n_data_random_pairs = len(coords_data) * len(coords_random)

RR_density = RR_hist / n_random_pairs
DR_density = DR_hist / n_data_random_pairs
DD_density = DD_hist / n_data_pairs

# tpfc
xi = (DD_density - 2 * DR_density + RR_density) / RR_density

# plotting
plt.figure(figsize=(10, 6))
plt.loglog(bin_centers, xi, marker='x', linestyle='None', color='green')
plt.xlabel('Separation Distance (pixels)')
plt.ylabel('Two-Point Correlation Function (Landy-Szalay)')
plt.title('Two-Point Correlation Function for Filtered Galaxy Catalog (x < 1200)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# Peebles
def peebles_model(r, r0, A):
    gamma = 1.8  # 固定 gamma 为 1.8
    return A * (r / r0) ** (-gamma)


initial_guess = [10, 1]  # 初始猜测值，r0=10, A=1
valid_indices = ~np.isnan(xi) & (xi > 0.001) & (bin_centers > 10) & ~np.isinf(xi) & ~np.isinf(bin_centers)

popt, pcov = curve_fit(peebles_model, bin_centers[valid_indices], xi[valid_indices], p0=initial_guess)

# fit
r0_fit, A_fit = popt

# plot tpfc
r_values = np.logspace(np.log10(10), np.log10(np.max(pairwise_distances_data)), num=500)
xi_fitted = peebles_model(r_values, r0_fit, A_fit)

plt.figure(figsize=(10, 6))
plt.loglog(bin_centers, xi, marker='x', linestyle='None', color='green', label='Observed Data (Landy-Szalay)')
plt.plot(r_values, xi_fitted, linestyle='-', color='red', label=f'Fitted Peebles Model (r0={r0_fit:.2f})')
plt.xlabel('Separation Distance (pixels)')
plt.ylabel('Two-Point Correlation Function')
plt.title('Fitting Observed Two-Point Correlation Function with Peebles Model')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()


print(f"Fitted value of r0: {r0_fit:.2f}")
