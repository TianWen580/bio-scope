from __future__ import annotations

import argparse
import sys

from bioclip_model import export_tol_species_list


def main() -> int:
    parser = argparse.ArgumentParser(description='Export BioCLIP TreeOfLife taxa/species list')
    parser.add_argument(
        '--taxa-csv',
        default='./data/bioclip_tol_taxa.csv',
        help='Path to output taxa CSV generated from TreeOfLife labels',
    )
    parser.add_argument(
        '--species-txt',
        default='./data/bioclip_tol_species.txt',
        help='Path to output deduplicated species text list',
    )
    parser.add_argument(
        '--model',
        default=None,
        help='Optional pybioclip model string; defaults to BIOCLIP_TOL_MODEL_ID/BIOCLIP_MODEL_ID',
    )
    args = parser.parse_args()

    count, error = export_tol_species_list(
        output_csv_path=args.taxa_csv,
        output_species_txt_path=args.species_txt,
        model_str=args.model,
    )

    if error:
        print(f'Failed: {error}', file=sys.stderr)
        return 1

    print(f'Exported {count} species labels to {args.species_txt}')
    print(f'Exported full taxa table to {args.taxa_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
