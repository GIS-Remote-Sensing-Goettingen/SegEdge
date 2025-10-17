"""
This scipts load Satlas SR from allen institute  and superresolvee Sentinel 2 images
"""
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Superresolve Sentinel 2 images using Satlas SR model from Allen Institute"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input Sentinel 2 image",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output superresolved image",
    )

    args = parser.parse_args()

    from sege.pipelines import satlas_sr

