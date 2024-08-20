"""Help info here."""

import argparse
import sys

def main():
    """Do the thing."""
    if any(s in sys.argv for s in ["-h", "--help", "help"]):
        print(__doc__)
    else:
        pass


if __name__ == "__main__":
    main()
