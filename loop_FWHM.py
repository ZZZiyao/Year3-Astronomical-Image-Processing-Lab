import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize, LinearStretch
from sklearn.cluster import DBSCAN
from centre import load_and_process_fits  # Assuming this module and function are correctly defined
import os
import warnings
from astropy.utils.metadata import MergeConflictWarning
from scipy.ndimage import median_filter

# Suppress specific warnings
warnings.simplefilter('ignore', MergeConflictWarning)


def display_fits(data, annotations=None):
    """
    Display function that includes data and annotations.
    Displays FITS image data using ZScale normalization and optional overlays.
    """
    # Calculate the median of non-NaN values
    median_value = np.nanmedian(data)
    # Replace NaN values with the median for display purposes
    display_data = np.where(np.isnan(data), median_value, data)

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(display_data)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())

    plt.figure(figsize=(10, 8))
    im = plt.imshow(display_data, cmap='cool', norm=norm)
    plt.colorbar(im, label='Original Data Intensity')

    if annotations:
        for x, y in annotations:
            plt.plot(x, y, 'o', markersize=2, alpha=0.8, color='yellow')  # Yellow markers for detected sources

    plt.title('FITS Image with ZScale Stretch and Annotations')
    plt.xlabel('X Pixels')
    plt.ylabel('Y Pixels')
    plt.show()

def process_and_detect_stars(fits_file_path):
    # Load and process the image using 'centre.py' for initial adjustments
    processed_data = load_and_process_fits(fits_file_path)
    height, width = processed_data.shape

    # Apply a median filter to the edge regions to reduce salt-and-pepper noise
    edge_margin = 1
    filtered_data = processed_data.copy()
    edge_region = (
        (slice(0, edge_margin), slice(0, width)),  # Top edge
        (slice(height - edge_margin, height), slice(0, width)),  # Bottom edge
        (slice(0, height), slice(0, edge_margin)),  # Left edge
        (slice(0, height), slice(width - edge_margin, width))  # Right edge
    )
    
    for region in edge_region:
        filtered_data[region] = median_filter(processed_data[region], size=1)  # Apply median filter with a 3x3 kernel

    # Display the image after median filtering
    display_fits(filtered_data)

    # Apply background subtraction using Background2D for the entire image, ignoring zeros
    mask = (filtered_data == 0)  # Create a mask for zero values
    bkg_estimator = MedianBackground()
    bkg = Background2D(filtered_data, (50, 50), bkg_estimator=bkg_estimator, mask=mask)
    data_subtracted = filtered_data - bkg.background

    all_sources = []

    # Star detection using DAOStarFinder on the globally background-subtracted data, ignoring zero regions
    for fwhm_value in np.arange(1, 17, 1):
        median = np.median(data_subtracted[data_subtracted > 0])  # Calculate median ignoring zeros
        for i in range(0, height, 100):  # Increase the step size to reduce the number of regions
            for j in range(0, width, 100):
                # Define the coordinates of the current region
                x_start, x_end = j, min(j + 100, width)
                y_start, y_end = i, min(i + 100, height)
                region_data = data_subtracted[y_start:y_end, x_start:x_end]

                # Apply the zero mask to the region data
                region_data[region_data == 0] = median

                # Star detection in the current region
                star_finder = DAOStarFinder(fwhm=fwhm_value, threshold=4 * mad_std(region_data), exclude_border=True)
                try:
                    sources = star_finder.find_stars(region_data)
                except Exception as e:
                    print(f"Error in DAOStarFinder for FWHM {fwhm_value}, Region ({i//100},{j//100}): {e}")
                    sources = None

                if sources is not None:
                    sources['xcentroid'] += x_start
                    sources['ycentroid'] += y_start
                    sources = sources[~np.isnan(sources['sharpness'])]  # Remove rows with NaN in `sharpness`
                    all_sources.append(sources)
                    print(f"FWHM {fwhm_value}, Region ({i//100},{j//100}): Found {len(sources)} stars")
                else:
                    print(f"FWHM {fwhm_value}, Region ({i//100},{j//100}): No stars found")

    # Concatenate all detected sources into a single table
    if all_sources:
        from astropy.table import vstack
        concatenated_sources = vstack(all_sources)

        # Use the positions to create a clustering
        positions = np.array([concatenated_sources['xcentroid'], concatenated_sources['ycentroid']]).transpose()
        clustering = DBSCAN(eps=4, min_samples=2).fit(positions)
        labels = clustering.labels_

        # Create a catalog of final star properties after clustering
        unique_labels = set(labels)
        final_sources = []

        for label in unique_labels:
            if label == -1:
                # -1 label means noise, skip it
                continue
            cluster_sources = concatenated_sources[labels == label]
            
            # Select the source with the maximum npix within each cluster
            max_npix_source = cluster_sources[np.argmax(cluster_sources['npix'])]
            final_sources.append(max_npix_source)

        # Create a new table with the final sources
        from astropy.table import Table
        final_sources_table = Table(rows=final_sources, names=concatenated_sources.colnames)

        # Define image edge margin
        edge_margin = 2

        # Apply different filtering conditions based on whether the star is on the edge
        filtered_sources = []
        for source in final_sources_table:
            x, y = source['xcentroid'], source['ycentroid']
            # Check if the source is at the edge of the image
            if (x < edge_margin) or (x > width - edge_margin) or (y < edge_margin) or (y > height - edge_margin):
                # Only apply npix filter for sources on the edge
                if (source['npix'] > 36) and (source['sharpness'] > 0.2) and (source['sharpness'] < 0.8):
                    filtered_sources.append(source)
            else:
                # Apply full filter conditions for sources not on the edge
                if (source['sharpness'] > 0.05) and (source['sharpness'] < 0.95):
                    filtered_sources.append(source)

        # Convert the filtered sources back to an astropy table
        filtered_sources_table = Table(rows=filtered_sources, names=final_sources_table.colnames)

        print(f"Number of stars after clustering and final filtering: {len(filtered_sources_table)}")

        # Save the catalog to a text file
        catalog_file_path = r"D:\aip\astronomical-image-processing\loop18_catalog.txt"
        with open(catalog_file_path, 'w') as f:
            f.write("# id xcentroid ycentroid flux sharpness roundness1 roundness2 npix sky peak mag\n")
            for idx, source in enumerate(filtered_sources_table):
                f.write(f"{idx} {source['xcentroid']} {source['ycentroid']} {source['flux']} "
                        f"{source['sharpness']} {source['roundness1']} {source['roundness2']} "
                        f"{source['npix']} {source['peak']} {source['mag']}\n")
        print(f"Catalog saved to {catalog_file_path}")

        # Display the original image with only the filtered source annotations
        unique_points = [(int(source['xcentroid']), int(source['ycentroid'])) for source in filtered_sources_table]
        display_fits(data_subtracted, unique_points)
    else:
        display_fits(data_subtracted)

if __name__ == "__main__":
    fits_file_path = r"D:\aip\Astro\Fits_Data\mosaic.fits"
    process_and_detect_stars(fits_file_path)
