import argparse
from enum import Enum
from pathlib import Path
import sys

from . import FLOAT, COMPLEX
from .hologram import HologramParameters, compute_hologram
from .image import create_hologram_image
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
        '-o', '--output',
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
        '--wavelength',
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
        '--plate-resolution', '-r',
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

    parser.add_argument(
        '--object-distance',
        type=float,
        default=100.0,
        help='Distance of object behind plate in millimeters'
    )

    # Object processing parameters
    parser.add_argument(
        '--scale-factor',
        type=float,
        default=12.0,
        help='Scale factor for STL object'
    )

    parser.add_argument(
        '--subdivision-factor', '-s',
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

    return parser.parse_args()


def main() -> None:
    """Main function with command line interface."""
    print(f"Using numpy.{FLOAT.__name__} and numpy.{COMPLEX.__name__} precision.")
    args = parse_args()

    # Create parameter object from arguments
    params = HologramParameters(
        wavelength=args.wavelength,
        plate_size=args.plate_size,
        plate_resolution=args.plate_resolution,
        light_source_distance=args.light_source_distance,
        object_distance=args.object_distance,
        scale_factor=args.scale_factor,
        subdivision_factor=args.subdivision_factor
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

        if args.debug:
            # Save additional debug information
            debug_dir = args.output.parent / 'debug'
            debug_dir.mkdir(exist_ok=True)

            # Save parameters
            with open(debug_dir / 'parameters.txt', 'w') as f:
                for key, value in vars(args).items():
                    f.write(f"{key}: {value}\n")

            # Could add more debug outputs here if needed

        print("Processing complete!")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
