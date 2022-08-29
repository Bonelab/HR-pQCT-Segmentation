from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def create_parser():
    parser = ArgumentParser(
        description='HRpQCT Segmentation 2D UNet Segmenting Script',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "image-directory", type=str, metavar="STR",
        help="directory containing AIM images to segment"
    )
    parser.add_argument(
        "--image-pattern"
    )
    parser.add_argument(
        "--masks-subdirectory", "-ms", type=str, default="masks", metavar="STR",
        help="subdirectory, inside of `image-directory`, to save the masks to"
    )
    parser.add_argument(
        ""
    )


def main():
    args = create_parser().parse_args()


if __name__ == "__main__":
    main()