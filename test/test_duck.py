import os
import tempfile
from lib.train import create_initial_state
import lib.render_duck as duck
import sys
from pathlib import Path

# Import create_train_run from test/conftest.py
sys.path.insert(0, str(Path(__file__).parent))
from conftest import create_train_run


def test_db():
    duck.ensure_duck(None, True)

    train_run = create_train_run()
    state = create_initial_state(train_run, None, "cpu")
    model_id = duck.insert_model(train_run)
    duck.insert_model_parameter(model_id, train_run.run_id, "test", 0)
    duck.insert_model_parameter(model_id, train_run.run_id, "test", 0.5)
    duck.insert_model_parameter(model_id, train_run.run_id, "test", "test")
    duck.insert_train_step(model_id, train_run.run_id, 0, "dataset", [0])
    duck.insert_train_step_metric(model_id, train_run.run_id, "metric", 1, 0.5)

    duck.render_duck(train_run, state)
    # duck.sync()  # Deprecated - use start_periodic_export() instead
    duck.insert_train_step_metric(model_id, train_run.run_id, "metric", 2, 0.5)

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "tempfile.bin")
        with open(file_path, "wb") as f:
            for i in range(1024 * 1024):
                f.write(b"\x00\xff\xaa\xbb")

        artifact_id = duck.insert_artifact(model_id, "art_test", file_path)

        bytes = duck.get_artifact(artifact_id)
        assert bytes == open(file_path, "rb").read()


if __name__ == "__main__":
    test_db()
    print("All tests passed!")
