# from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def reconstruct_image_from_png(
    png_path: str,
    wavelength: float,
    plate_resolution: float,
    reconstruction_distance: float,
    dtype: type = np.complex64,
) -> np.ndarray:
    """
    Reconstruct a flat image from a hologram saved as a PNG.

    Parameters
    ----------
    png_path : str
        Path to the saved PNG hologram.
    wavelength : float
        Wavelength of the reference light in mm.
    plate_resolution : float
        Size of each pixel in mm.
    reconstruction_distance : float
        Distance between the hologram and the reconstructed image plane (mm).
    dtype : type, default np.complex64
        Data type for calculations.

    Returns
    -------
    np.ndarray
        Reconstructed intensity image.
    """
    # Load the hologram from the 16-bit grayscale PNG
    hologram_image = Image.open(png_path).convert('I;16')  # Load as grayscale
    hologram_array = np.array(hologram_image, dtype=np.float32)

    # Map the grayscale values [0, 255] to phase values [-π, π]
    phase = (hologram_array / 65535.0) * (2 * np.pi) - np.pi

    # Construct the hologram's complex field (uniform amplitude assumed)
    hologram_field = np.exp(1j * phase).astype(dtype)

    # Call the original reconstruct_image function
    return reconstruct_image(
        hologram=hologram_field,
        wavelength=wavelength,
        plate_resolution=plate_resolution,
        reconstruction_distance=reconstruction_distance,
    )


def reconstruct_image(
    hologram: np.ndarray,
    wavelength: float,
    plate_resolution: float,
    reconstruction_distance: float,
) -> np.ndarray:
    """
    Reconstruct a flat image from a hologram.

    Parameters
    ----------
    hologram : np.ndarray
        The computed interference pattern (hologram).
    wavelength : float
        Wavelength of the reference light in mm.
    pixel_size : float
        Size of each pixel in mm.
    reconstruction_distance : float
        Distance between the hologram and the reconstructed image plane (mm).

    Returns
    -------
    np.ndarray
        Reconstructed intensity image.
    """
    k = 2 * np.pi / wavelength  # Wave number

    # Compute the Fourier Transform of the hologram
    spectrum = np.fft.fftshift(np.fft.fft2(hologram))

    # Create spatial frequency coordinates
    ny, nx = hologram.shape
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=plate_resolution))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=plate_resolution))
    FX, FY = np.meshgrid(fx, fy)

    # Propagator in the Fourier domain
    H = np.exp(1j * k * reconstruction_distance) * np.exp(
        -1j * np.pi * wavelength * reconstruction_distance * (FX**2 + FY**2)
    )

    # Backpropagation (multiply the spectrum with the propagator)
    reconstructed_field = np.fft.ifft2(np.fft.ifftshift(spectrum * H))

    # Compute the intensity of the reconstructed field
    reconstructed_intensity = np.abs(reconstructed_field) ** 2

    return reconstructed_intensity
