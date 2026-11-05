"""Freeze only a new source release; never reseal a started pilot."""
import argparse
from common import ROOT, sha, source_files, verify_source, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create",action="store_true")
    args = parser.parse_args()
    if args.create:
        if (ROOT/"outputs").exists():
            raise RuntimeError("An output tree exists; use a new experiment version instead of resealing")
        write_json(ROOT/"source_manifest.json",dict(schema=1,files=source_files()))
    print(verify_source())


if __name__ == "__main__":
    main()
