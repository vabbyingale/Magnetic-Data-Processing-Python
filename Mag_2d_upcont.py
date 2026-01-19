import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from numpy.fft import fft2, fftshift
from math import ceil
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import Mag_guspi_upward

# --------------------------------------------------
# Load files
# --------------------------------------------------
data_mag = np.loadtxt("File_forward_anom.txt")
data_depth = np.loadtxt("File_forward_depth.txt")
data_forward = np.loadtxt("File_forward_model.txt")

# --------------------------------------------------
# Create meshgrid from unique x, y
# --------------------------------------------------
x = np.unique(data_mag[:, 0])
y = np.unique(data_mag[:, 1])
X, Y = np.meshgrid(x, y)

nx = len(x)
ny = len(y)

# --------------------------------------------------
# Interpolate to grid (nearest)
# --------------------------------------------------
Z_mag = griddata(
    (data_mag[:, 0], data_mag[:, 1]),
    data_mag[:, 2],
    (X, Y),
    method="nearest"
)

Z_depth = griddata(
    (data_depth[:, 0], data_depth[:, 1]),
    data_depth[:, 2],
    (X, Y),
    method="nearest"
)

Z_forward = griddata(
    (data_forward[:, 0], data_forward[:, 1]),
    data_forward[:, 2],
    (X, Y),
    method="nearest"
)

# --------------------------------------------------
# Grid spacing and parameters
# --------------------------------------------------
dx = abs(x[1] - x[0])
dy = abs(y[1] - y[0])
zlev = ceil(np.nanmax(Z_depth))
pad_factor = 0.1

# --------------------------------------------------
# Estimate short wavelength cutoff ws
# --------------------------------------------------
lambda_cutoff = np.nanmax(Z_depth) + abs(np.nanmin(Z_depth))
ws_x = lambda_cutoff / dx
ws_y = lambda_cutoff / dy
ws_guess = 3 * np.mean([ws_x, ws_y])

# --------------------------------------------------
# Estimate long wavelength cutoff wl from dominant FFT wavelength
# --------------------------------------------------
F = np.abs(fft2(Z_mag))
F_shift = fftshift(F)

kx = np.arange(-nx/2, nx/2) / (nx * dx)
ky = np.arange(-ny/2, ny/2) / (ny * dy)
kxx, kyy = np.meshgrid(kx, ky)

idx_max = np.unravel_index(np.argmax(F_shift), F_shift.shape)
ky_idx, kx_idx = idx_max

lambda_dom = 1.0 / np.sqrt(
    kxx[ky_idx, kx_idx]**2 + kyy[ky_idx, kx_idx]**2
)
wl_guess = lambda_dom / np.mean([dx, dy])

print(f"Estimated wl ~ {wl_guess:.0f}, ws ~ {ws_guess:.0f}")

# --------------------------------------------------
# Upward continuation
# --------------------------------------------------
m_lev = Mag_guspi_upward(
    Z_mag,
    Z_depth,
    dx,
    dy,
    wl_guess,
    ws_guess,
    zlev,
    pad_factor
)

# --------------------------------------------------
# Residual analysis
# --------------------------------------------------
diff = Z_forward - m_lev
diff_flat = diff.ravel()

mu = np.mean(diff_flat)
sigma = np.std(diff_flat)

print(f"Residual mean: {mu:.3e}, std: {sigma:.3e}")

# --------------------------------------------------
# Plotting with aligned colorbars
# --------------------------------------------------
zmin = round(np.nanmin(Z_mag) / 100) * 100
zmax = round(np.nanmax(Z_mag) / 100) * 100
contour_levels = np.arange(zmin, zmax + 1, 200)

fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(2, 3)

# ---- Plot 1: Depth ----
ax1 = fig.add_subplot(gs[0, 0])
cf = ax1.contourf(X, Y, np.flipud(Z_depth), cmap="viridis")
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf, cax=cax)
ax1.set_title("Depth (km)")
ax1.set_aspect("equal")
ax1.invert_yaxis()

# ---- Plot 2: Original field ----
ax2 = fig.add_subplot(gs[0, 1])
cf = ax2.contourf(X, Y, np.flipud(Z_mag), 50, cmap="jet")
cs = ax2.contour(X, Y, np.flipud(Z_mag), contour_levels, colors="k", linewidths=1)
ax2.clabel(cs, fontsize=10)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf, cax=cax)
ax2.set_title("Original Magnetic Field")
ax2.set_aspect("equal")
ax2.invert_yaxis()

# ---- Plot 3: Forward model ----
ax3 = fig.add_subplot(gs[0, 2])
cf = ax3.contourf(X, Y, np.flipud(Z_forward), 50, cmap="jet")
cs = ax3.contour(X, Y, np.flipud(Z_forward), contour_levels, colors="k", linewidths=1)
ax3.clabel(cs, fontsize=10)
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf, cax=cax)
ax3.set_title("Forward Model")
ax3.set_aspect("equal")
ax3.invert_yaxis()

# ---- Plot 4: Spatial residuals ----
ax5 = fig.add_subplot(gs[1, 1])
cn = np.max(np.abs(Z_forward - m_lev))
cf = ax5.contourf(X, Y, np.flipud(Z_forward - m_lev), 50,
                  cmap="seismic", vmin=-cn, vmax=cn)
divider = make_axes_locatable(ax5)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf, cax=cax)
ax5.set_title("Difference: Forward - Upcont")
ax5.set_aspect("equal")
ax5.invert_yaxis()

# ---- Plot 5: Upward continuation ----
ax6 = fig.add_subplot(gs[1, 2])
cf = ax6.contourf(X, Y, np.flipud(m_lev), 50, cmap="jet")
cs = ax6.contour(X, Y, np.flipud(m_lev), contour_levels, colors="k", linewidths=1)
ax6.clabel(cs, fontsize=10)
divider = make_axes_locatable(ax6)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf, cax=cax)
ax6.set_title(
    f"Upcont at zlev={zlev:.1f} (wl={wl_guess:.2f}, ws={ws_guess:.2f})"
)
ax6.set_aspect("equal")
ax6.invert_yaxis()

plt.tight_layout()
plt.show(block=True)
plt.close()
