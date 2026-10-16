"""Run a recorded demo through the production fit and MC5 reliability owners."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from logic.conformer_workflow import snapshot_from_payload, execute_fit, validate_result
from logic.conformer_reliability import ReliabilityOptions, evaluate_reliability


def run(folder, output, *, replicates=100, folds=5, method="parametric"):
    data = json.loads((Path(folder)/"inputs.json").read_text(encoding="utf-8"))
    snapshot = snapshot_from_payload(data["snapshot"])
    result = execute_fit(snapshot, data["options"], input_rows=snapshot.conformers)
    if not result["success"]:
        raise RuntimeError("Demo fit failed; inspect recorded solver messages")
    def progress(stage, done, total):
        if done == total or done % 10 == 0:
            print(f"{stage}: {done}/{total}", flush=True)
    result["reliability"] = evaluate_reliability(result, ReliabilityOptions(method=method, replicates=replicates, folds=folds), progress=progress)
    validate_result(result)
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    diagnostics = result["reliability"]
    print("Recovered:", result["best"]["populations"], result["best"]["dchi_ax"], result["best"]["dchi_rh"])
    print("Bootstrap successful:", diagnostics["bootstrap"]["successful"], "/", replicates)
    print("CV:", diagnostics["cross_validation"]["rmsd_ppm"], "ppm; complete:", diagnostics["cross_validation"]["complete"])
    print("Suggested groups:", [g["conformer_ids"] for g in diagnostics["suggested_groups"]])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("informative", "ambiguous"), default="informative")
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--method", choices=("parametric", "cases"), default="parametric")
    parser.add_argument("--output", type=Path, required=True, help="Result JSON path; does not modify bundled inputs")
    args = parser.parse_args()
    run(Path(__file__).resolve().parent/args.scenario, args.output,
        replicates=args.replicates, folds=args.folds, method=args.method)
