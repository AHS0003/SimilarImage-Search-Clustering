"""Shared command-line validation for the experiment entry points."""
import argparse
import json

from src.evaluation import CONFIGURATIONS, DESCRIPTORS, run_experiments


def cluster_argument(value):
    if value == 'auto':
        return value
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('--clusters must be auto or a positive integer') from exc
    if number < 1:
        raise argparse.ArgumentTypeError('--clusters must be auto or a positive integer')
    return number


def main(mode, argv=None):
    parser = argparse.ArgumentParser(description={
        'clustering': 'Evaluate gallery K-Means without using query images for fitting or K selection.',
        'retrieval': 'Evaluate fixed feature/rank fusion on held-out validation and test queries.',
        'compare': 'Compare all seven classical descriptor configurations on one shared split.',
    }[mode])
    parser.add_argument('--dataset', required=True, help='Directory containing category subdirectories, or unlabelled root images')
    parser.add_argument('--output', required=True, help='Directory for CSV reports, plots and reproducibility metadata')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clusters', type=cluster_argument, default='auto', metavar='auto|INT')
    parser.add_argument('--k-values', type=int, nargs='+', help='Explicit gallery K grid; invalid values are rejected. Default: 2..min(10,N).')
    parser.add_argument('--weights', type=float, nargs=3, default=(1., 1., 1.),
                        metavar=('COLOR', 'SHAPE', 'TEXTURE'), help='Fixed a-priori descriptor weights; never tuned on test data')
    parser.add_argument('--strategy', choices=('feature', 'rank', 'both'), default='both')
    parser.add_argument('--dataset-kind', choices=('real', 'synthetic'), default='real',
                        help='Use synthetic for generated smoke data; results are explicitly tagged')
    if mode != 'compare':
        parser.add_argument('--descriptors', choices=DESCRIPTORS, nargs='+', default=list(DESCRIPTORS))
    args = parser.parse_args(argv)
    try:
        metadata = run_experiments(
            dataset=args.dataset, output=args.output, seed=args.seed,
            configurations=CONFIGURATIONS if mode == 'compare' else (tuple(args.descriptors),),
            strategies=('feature', 'rank') if args.strategy == 'both' else (args.strategy,),
            weights=dict(zip(DESCRIPTORS, args.weights)), clusters=args.clusters,
            k_values=args.k_values,
            tasks=('clustering', 'retrieval') if mode == 'compare' else (mode,),
            dataset_kind=args.dataset_kind,
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: metadata[key] for key in
                      ('results_scope', 'readable_unique_images', 'split_counts', 'cluster_selections')}, indent=2))
    return 0
