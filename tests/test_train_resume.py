import json

from train import find_latest_best_model


def _write_run(save_dir, log_dir, timestamp, approach):
    model_path = save_dir / f"best_model_{timestamp}.pth"
    model_path.write_bytes(b"stub")
    config_path = log_dir / f"config_{timestamp}.json"
    config_path.write_text(json.dumps({"args": {"approach": approach}}), encoding='utf-8')
    return model_path


def test_find_latest_best_model_uses_selected_approach(tmp_path):
    save_dir = tmp_path / "saved_models"
    log_dir = tmp_path / "logs"
    save_dir.mkdir()
    log_dir.mkdir()

    _write_run(save_dir, log_dir, "20260101_120000", "dqn")
    jepa_model = _write_run(save_dir, log_dir, "20260102_120000", "jepa")
    dqn_model = _write_run(save_dir, log_dir, "20260103_120000", "dqn")

    assert find_latest_best_model(str(save_dir), str(log_dir), "dqn") == str(dqn_model)
    assert find_latest_best_model(str(save_dir), str(log_dir), "jepa") == str(jepa_model)
    assert find_latest_best_model(str(save_dir), str(log_dir), "evo") is None
