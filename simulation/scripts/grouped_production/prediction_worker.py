"""Prediction evaluator launched by run.py after its PID challenge and locks."""
import json
from pathlib import Path
import sys
from dexonomy.prediction_eval import load_request, evaluate
from dexonomy.tabletop_protocol import serial

if __name__ == '__main__':
    config = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=False)
    request = load_request(config['prediction_request'])
    result = evaluate(request, backend='warp', batch_size=config.get('eval_batch', 6))
    (out/'physics.json').write_text(json.dumps(result, default=serial, indent=2, allow_nan=False))
    if result['counts']['simulation_error'] or result['counts']['invalid_input']:
        raise SystemExit(2)
