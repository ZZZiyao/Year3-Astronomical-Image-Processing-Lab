import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.background import MedianBackground

def load_catalog(catalog_file_path):
    """
    Load star catalog from a text file and filter out sources near image edges.
    """
    # Load the catalog using Astropy's Table
    catalog = Table.read(catalog_file_path, format='ascii')

    # Print column names to verify
    print("Available columns in catalog:", catalog.colnames)

    # Determine image boundaries from data
    x_min, x_max = np.min(catalog['col2']), np.max(catalog['col2'])
    y_min, y_max = np.min(catalog['col3']), np.max(catalog['col3'])

    # Define edge margin
    edge_margin = 200

    # Filter out stars that are too close to the image edge (within 200 pixels)
    filtered_catalog = catalog[(catalog['col2'] > x_min + edge_margin) &
                               (catalog['col2'] < x_max - edge_margin) &
                               (catalog['col3'] > y_min + edge_margin) &
                               (catalog['col3'] < y_max - edge_margin)]
    return filtered_catalog

def calculate_flux_with_aperture(image_data, sources):
    """
    Calculate flux for each star using aperture photometry with background subtraction.
    
    Parameters:
    - image_data: 2D numpy array, the image data from which to calculate flux.
    - sources: Astropy Table, containing xcentroid and ycentroid of detected stars.
    
    Returns:
    - fluxes: numpy array, the calculated fluxes for each source.
    """
    positions = np.transpose((sources['col2'], sources['col3']))
    
    # Adjust aperture radius based on the npix value in the catalog
    aperture_radii = np.sqrt(sources['col8']) + 2  # Adjust radius based on npix

    fluxes = []
    for i, pos in enumerate(positions):
        # Define circular aperture for the star and annulus for the background
        aperture = CircularAperture(pos, r=aperture_radii[i])
        annulus_aperture = CircularAnnulus(pos, r_in=aperture_radii[i] + 5, r_out=aperture_radii[i] + 10)

        # Perform aperture photometry
        phot_table = aperture_photometry(image_data, aperture)
        annulus_table = aperture_photometry(image_data, annulus_aperture)

        # Calculate background per pixel in the annulus using median
        annulus_mask = annulus_aperture.to_mask(method='center')
        annulus_data = annulus_mask.multiply(image_data)
        annulus_data_1d = annulus_data[annulus_data > 0]  # Flatten and remove zero values
        bkg_median = np.median(annulus_data_1d)

        # Calculate total background in the aperture
        bkg_sum = bkg_median * aperture.area

        # Calculate flux by subtracting background from aperture sum
        flux = phot_table['aperture_sum'][0] - bkg_sum
        fluxes.append(flux)

    return np.array(fluxes)

def process_and_plot_flux(fits_file_path, catalog_file_path):
    """
    Process the FITS data and star catalog to calculate magnitudes and plot the log(N) vs. magnitude.
    
    Parameters:
    - fits_file_path: str, path to the FITS file.
    - catalog_file_path: str, path to the star catalog file.
    """
    # Load the FITS data
    with fits.open(fits_file_path) as hdul:
        image_data = hdul[0].data
        # Read the zero point from the FITS header
        ZP_inst = hdul[0].header.get('MAGZPT',25.3)  # Default to 25.0 if MAGZPT is not available
        print(f"Zero point (MAGZPT) from FITS header: {ZP_inst}")
        ZP_error = hdul[0].header.get('MAGZRR',0.02)  # Default to 0.0 if MAGZRR is not available
        print(f"Zero point error (MAGZRR) from FITS header: {ZP_error}")


    # Load and filter the catalog
    filtered_catalog = load_catalog(catalog_file_path)

    # Calculate the fluxes using aperture photometry
    fluxes = calculate_flux_with_aperture(image_data, filtered_catalog)

    # Add flux to catalog and calculate magnitudes using the provided zero-point constant
    filtered_catalog['col4'] = fluxes
    counts = filtered_catalog['col4']
    
    # Avoid issues with non-positive flux values when taking the logarithm
    valid_flux_indices = counts > 0
    valid_fluxes = counts[valid_flux_indices]
    valid_catalog = filtered_catalog[valid_flux_indices]

    magnitudes = ZP_inst - 2.5 * np.log10(valid_fluxes)
    valid_catalog['magnitude'] = magnitudes

    # Sort magnitudes to create cumulative count
    sorted_magnitudes = np.sort(magnitudes)
    N_values = np.arange(1, len(sorted_magnitudes) + 1)

    # Plot log(N) vs magnitude
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_magnitudes, np.log10(N_values), 'o', markersize=4, alpha=0.7)
    plt.xlabel('Magnitude (m)')
    plt.ylabel('log(N(m))')
    plt.title('Number of Sources Detected vs. Magnitude')
    plt.grid()
    plt.show()

    # Print the original pixel counts of the 10 brightest sources (smallest magnitude)
    sorted_catalog = valid_catalog[np.argsort(valid_catalog['magnitude'])]
    print("Original pixel counts of the 10 brightest sources (smallest magnitude):")
    for i in range(min(10, len(sorted_catalog))):
        x = int(sorted_catalog['col2'][i])
        y = int(sorted_catalog['col3'][i])
        original_pixel_value = image_data[y, x]
        print(f"Source {i + 1}: xcentroid={x}, ycentroid={y}, original pixel value={original_pixel_value}, magnitude={sorted_catalog['magnitude'][i]}")

if __name__ == "__main__":
    fits_file_path = r"D:\aip\Astro\Fits_Data\mosaic.fits"
    catalog_file_path = r"D:\aip\astronomical-image-processing\loop8_catalog.txt"

    # Process and plot the flux data
    process_and_plot_flux(fits_file_path, catalog_file_path)
