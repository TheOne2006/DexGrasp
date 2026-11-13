"""DexGrasp v1 command line: data, train and test, without legacy model entries."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare-data')
    prepare.add_argument('--root', required=True)
    prepare.add_argument('--seed', type=int, default=0)
    prepare.add_argument('--grasps-per-shard', type=int, default=4096)
    audit = sub.add_parser('audit-data')
    audit.add_argument('--root', required=True)
    audit.add_argument('--coord-scale-m', type=float, default=0.2)
    for name in ('plan', 'train'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        if name == 'train':
            p.add_argument('--execute', action='store_true', help='Start optimizer updates (default only prints plan)')
            p.add_argument('--resume')
            p.add_argument('--stop-after', type=int, help='Save and exit after this optimizer step; resume with unchanged config')
    sample = sub.add_parser('sample')
    sample.add_argument('--checkpoint', required=True)
    sample.add_argument('--dataset-root', required=True)
    sample.add_argument('--output', required=True)
    sample.add_argument('--split', choices=('train', 'test'), default='test')
    sample.add_argument('--steps', type=int)
    sample.add_argument('--candidates', type=int, default=8)
    sample.add_argument('--batch-size', type=int, default=8)
    sample.add_argument('--device', default='cpu')
    sample.add_argument('--seed', type=int, default=0)
    sample.add_argument('--palm-gap-m', type=float)
    sample.add_argument('--direction', type=float, nargs=3)
    decode = sub.add_parser('decode')
    decode.add_argument('--samples', required=True)
    decode.add_argument('--output', required=True)
    decode.add_argument('--tolerance-m', type=float, default=0.003)
    decode.add_argument('--max-nfev', type=int, default=100)
    evaluate = sub.add_parser('test')
    evaluate.add_argument('--decoded', required=True)
    evaluate.add_argument('--verdicts')
    evaluate.add_argument('--gap-tolerance-m', type=float, default=0.005)
    evaluate.add_argument('--direction-tolerance-deg', type=float, default=10.)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare-data':
            from .data.prepare import prepare_dataset
            result = prepare_dataset(args.root, seed=args.seed, grasps_per_shard=args.grasps_per_shard)
        elif args.command == 'audit-data':
            from .data.dataset import audit_dataset
            result = audit_dataset(args.root, args.coord_scale_m)
        elif args.command in ('plan', 'train'):
            from .config import load_config
            from .train.runner import plan, run
            cfg = load_config(args.config)
            result = plan(cfg)
            if args.command == 'train' and args.execute:
                if result['blockers']:
                    raise ValueError('; '.join(result['blockers']))
                result = run(cfg, resume=args.resume, stop_after=args.stop_after)
        elif args.command == 'sample':
            from .test.pipeline import generate
            options = vars(args).copy()
            options.pop('command')
            result = generate(**options)
        elif args.command == 'decode':
            from .test.pipeline import decode_file
            result = decode_file(args.samples, args.output, tolerance_m=args.tolerance_m, max_nfev=args.max_nfev)
        else:
            from .test.pipeline import evaluate
            result = evaluate(args.decoded, verdicts=args.verdicts, gap_tolerance_m=args.gap_tolerance_m,
                              direction_tolerance_deg=args.direction_tolerance_deg)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 2 if result.get('status') == 'blocked' else 0
    except (ValueError, OSError, ImportError, KeyError, TypeError, RuntimeError, IndexError) as exc:
        print(json.dumps(dict(status='error', type=type(exc).__name__, message=str(exc)), ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
