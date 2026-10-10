"""Save audit-trained models so predictions and gain profiles remain independently reviewable."""

def save_checkpoint(model, config, stage, representation, fold, params):
    import hashlib
    import json
    import re
    label = re.sub(r"[^a-zA-Z0-9_-]+", "_", representation).strip("_")
    label += "_" + hashlib.sha256(representation.encode()).hexdigest()[:8]
    directory = config.output_dir / "training_models" / stage
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{label}_fold{fold}.ubj"
    best = int(model.best_iteration)
    rounds = model.num_boosted_rounds()
    if not 0 <= best < rounds:
        raise ValueError("Invalid early-stopping iteration")
    model.save_model(target)
    metadata = {"representation": representation, "fold": fold, "params": params,
                "num_features": model.num_features(), "best_iteration": best,
                "best_validation_auc": float(model.best_score), "retained_boosting_rounds": rounds,
                "gain_note": "gain statistics use all retained trees, including early-stopping patience trees",
                "model_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
