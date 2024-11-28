import argparse
from enum import Enum
from pathlib import Path
import sys

import numpy as np
from PIL import Image


from .hologram import HologramParameters, compute_hologram
from .image import create_hologram_image
from .reconstruction import reconstruct_image_from_png
from .utilities import show_grid_memory_requirements, Timer


class NormalizationMethod(str, Enum):
    """Enumeration of supported normalization methods."""
    MINMAX = 'minmax'
    LOG = 'log'
    SIGMOID = 'sigmoid'


class OutputType(str, Enum):
    """Enumeration of supported output types."""
    PREVIEW = 'preview'
    HIGH_QUALITY = 'high_quality'


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Transmission Hologram Simulator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--stl_path',
        type=Path,
        default="cgh/stls/dodecahedron.stl",
        help='Path to input STL file'
    )

    # Output configuration
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default="hologram.png",
        help='Output file path (extension determines format)'
    )

    parser.add_argument(
        '--output-type',
        type=OutputType,
        choices=list(OutputType),
        default=OutputType.PREVIEW,
        help='Output type (preview = 16-bit PNG, high_quality = 32-bit TIFF)'
    )

    # Optical parameters
    parser.add_argument(
        '--wavelength', '-w',
        type=float,
        default=0.532,
        help='Wavelength of coherent light in millimeters (e.g., 0.532 for 532nm)'
    )
    parser.add_argument(
        '--plate-size',
        type=float,
        default=25.4,
        help='Size of virtual recording plate in millimeters'
    )

    parser.add_argument(
        '--plate-resolution', '-z',
        type=float,
        default=11.811,  # 300 dpi
        help='Recording resolution in dots per millimeter'
    )

    parser.add_argument(
        '--light-source-distance',
        type=float,
        default=100.0,
        help='Distance of coherent source in millimeters'
    )

    # Object processing parameters
    parser.add_argument(
        '--scale-factor', '-s',
        type=float,
        default=5.0,
        help='Scale factor for STL object'
    )

    parser.add_argument(
        '--rotation-factors', '-r',
        type=float,
        default=(0.0, 0.0, 0.0),
        nargs=3,
        help='Rotational transform in degrees for X, Y, and Z'
    )

    parser.add_argument(
        '--translation-factors', '-t',
        type=float,
        default=(0.0, 0.0, 0.0),
        nargs=3,
        help='Rotational transform in degrees for X, Y, and Z'
    )

    parser.add_argument(
        '--subdivision-factor', '-d',
        type=int,
        default=4,
        help='Number of times to subdivide triangles'
    )

    # Image processing parameters
    parser.add_argument(
        '--normalization',
        type=NormalizationMethod,
        choices=list(NormalizationMethod),
        default=NormalizationMethod.MINMAX,
        help='Method for normalizing intensity values'
    )

    # Optional debug output
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output and save intermediate results'
    )

    parser.add_argument(
        '--high-precision',
        action='store_true',
        help='If set, will use np.float128 and np.complex256, if available'
    )
    return parser.parse_args()


def main() -> None:
    """Main function with command line interface."""
    args = parse_args()

    if args.high_precision:
        try:
            dtype = np.float128
            complex_dtype = np.complex256
        except Exception:
            dtype = np.float64
            complex_dtype = np.complex128
    else:
        dtype = np.float32
        complex_dtype = np.complex64

    print(f"Using numpy.{dtype.__name__} and numpy.{complex_dtype.__name__} precision.")

    # Create parameter object from arguments
    params = HologramParameters(
        wavelength=args.wavelength,
        plate_size=args.plate_size,
        plate_resolution=args.plate_resolution,
        scale_factor=args.scale_factor,
        rotation_factors=args.rotation_factors,
        translation_factors=args.translation_factors,
        subdivision_factor=args.subdivision_factor,
        dtype=dtype,
        complex_dtype=complex_dtype,
    )

    show_grid_memory_requirements(params)

    try:
        # Validate input file
        if not args.stl_path.exists():
            raise FileNotFoundError(f"STL file not found: {args.stl_path}")

        # Create output directory if needed
        args.output.parent.mkdir(parents=True, exist_ok=True)

        # Simulate hologram
        print(f"Processing STL file: {args.stl_path}")
        with Timer("Computation required"):
            interference_pattern, phase = compute_hologram(args.stl_path, params)

        # Save output
        print(f"Saving hologram to: {args.output}")
        create_hologram_image(
            interference_pattern=interference_pattern,
            phase=phase,
            output_path=str(args.output),
            output_type=args.output_type,
            normalization_method=args.normalization
        )

        # Show an image reconstructed from the hologram
        reconstruction = reconstruct_image_from_png(
            png_path=args.output,
            wavelength=params.wavelength,
            plate_resolution=params.plate_resolution,
            reconstruction_distance=20.0
        )
        reconstruction = 255 * (reconstruction - np.min(reconstruction)) / (np.max(reconstruction) - np.min(reconstruction))
        reconstruction = reconstruction.astype(np.uint8)
        reconstruction_img = Image.fromarray(reconstruction, mode='L')
        reconstruction_img.save("reconstruction.png", format='PNG')

        # # Save additional debug information
        # debug_dir = args.output.parent / 'debug'
        # debug_dir.mkdir(exist_ok=True)

        # # Save parameters
        # with open(debug_dir / 'parameters.txt', 'w') as f:
        #     for key, value in vars(args).items():
        #         f.write(f"{key}: {value}\n")

        print("Processing complete!")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
