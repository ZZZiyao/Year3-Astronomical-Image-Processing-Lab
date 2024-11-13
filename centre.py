import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, LinearStretch
from scipy.ndimage import median_filter
from matplotlib.path import Path

def create_triangle_mask(data, vertices):
    """
    Create a mask for a triangle given vertices.
    """
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    points = np.transpose((x.ravel(), y.ravel()))
    path = Path(vertices)
    grid = path.contains_points(points)
    grid = grid.reshape((ny, nx))
    return grid

def create_rectangle_mask(data, center, width, height):
    """
    Create a mask for a rectangle given the center coordinates, width, and height.
    """
    ny, nx = data.shape
    x_center, y_center = center
    x_start = x_center - width // 2
    y_start = y_center - height // 2
    mask = np.zeros_like(data, dtype=bool)
    mask[y_start:y_start+height, x_start:x_start+width] = True
    return mask

def load_and_process_fits(file_path):
    """
    Load FITS file, apply masks to triangles, rectangles, a central star, and a line, 
    and return processed data.
    """
    with fits.open(file_path) as hdul:
        # Convert data to float to allow for NaN values
        data = hdul[0].data.astype(float)
        height, width = data.shape
        mask = np.zeros_like(data, dtype=bool)
        
        # Mask central star and line
        center_x, center_y = 1441, 3197
        radius = 300
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask[dist_from_center <= radius] = True
        mask[:, center_x-25:center_x+25] = True

        # Define triangles to mask
        triangles = [
            [(1429, 60.7), (1273, 0), (1538, 0)], 
            [(1289,121.6), (1428.4,171.6), (1528.6,121.6)],
            [(1392.67,213.80), (1472.13,213.80), (1438.0,238.11)],
            [(1286.40,231.83), (1525.89,231.83), (1428.44,270.01)],
            [(1013.5,309.6), (1704.2,309.6), (1440.9,363.1)],
            [(1095.4,423.2), (1653.4,423.2), (1438.9,474.7)]
        ]

        # Apply triangle masks
        for vertices in triangles:
            triangle_mask = create_triangle_mask(data, vertices)
            mask |= triangle_mask

        # Rectangles to mask (center coordinates, width, height)
        rectangles = [
            ((905, 2288), 90, 135),
            ((972, 2770), 90, 135),
            ((775, 3313), 100, 230),
            ((2133, 3759), 60, 100),
            ((2456, 3413), 60, 100),
            ((2131, 2308), 60, 100),
            ((2088, 1425), 60, 100)
        ]

        # Apply rectangle masks
        for center, width, height in rectangles:
            rectangle_mask = create_rectangle_mask(data, center, width, height)
            mask |= rectangle_mask

        # Set masked areas to NaN
        data[mask] = np.nan

    return data

def display_fits(data):
    """
    Display FITS image data using ZScale normalization, replacing NaN with the median value for visualization.
    """
    # Replace NaN values with median for display purposes
    median_value = np.nanmedian(data)
    display_data = np.where(np.isnan(data), median_value, data)

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(display_data)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())

    plt.figure(figsize=(10, 8))
    plt.imshow(display_data, cmap='cool', norm=norm)
    plt.colorbar()
    plt.title('Processed FITS Image with ZScale Stretch')
    plt.xlabel('X Pixels')
    plt.ylabel('Y Pixels')
    plt.show()

if __name__ == "__main__":
    file_path = r"D:/aip/Astro/Fits_Data/mosaic.fits"
    processed_data = load_and_process_fits(file_path)
    display_fits(processed_data)
