import os
import tempfile
from lib.train import create_initial_state
import lib.render_duck as duck


def test_db():
    duck.ensure_duck(None, True)
    from test import create_train_run

    train_run = create_train_run()
    state = create_initial_state(train_run, None, "cpu")
    model_id = duck.insert_model(train_run)
    train_id = train_run.serialize_human()["train_id"]
    duck.insert_model_parameter(train_id, "test", 0)
    duck.insert_model_parameter(train_id, "test", 0.5)
    duck.insert_model_parameter(train_id, "test", "test")
    duck.insert_train_step(model_id, 0, 1, "dataset", [0])
    duck.insert_train_step_metric(model_id, 0, "metric", 1, 0.5)
    duck.insert_train_step_metric(model_id, 0, "metric", 1, [0.5])

    duck.render_duck(train_run, state)
    # duck.sync()
    duck.insert_train_step_metric(model_id, 0, "metric", 2, 0.5)

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "tempfile.bin")
        with open(file_path, "wb") as f:
            for i in range(1024 * 1024):
                f.write(b"\x00\xff\xaa\xbb")

        artifact_id = duck.insert_artifact(model_id, "art_test", file_path)

        bytes = duck.get_artifact(artifact_id)
        assert bytes == open(file_path, "rb").read()


def test_sync():
    duck.ensure_duck(None, True)
    from test import create_train_run

    train_run = create_train_run()
    state = create_initial_state(train_run, None, "cpu")
    model_id = duck.insert_model(train_run)
    duck.insert_model_parameter(model_id, "test", 0)
    duck.insert_model_parameter(model_id, "test", 0.5)
    duck.insert_model_parameter(model_id, "test", "test")
    duck.insert_train_step(model_id, 0, 1, "dataset", [0])
    duck.insert_train_step_metric(model_id, 0, "metric", 1, 0.5)
    duck.insert_train_step_metric(model_id, 0, "metric", 1, [0.5])

    duck.render_duck(train_run, state)
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "tempfile.bin")
        with open(file_path, "wb") as f:
            for i in range(1024 * 1024):
                f.write(b"\x00\xff\xaa\xbb")

        artifact_id = duck.insert_artifact(model_id, "art_test", file_path)

        bytes = duck.get_artifact(artifact_id)
        assert bytes == open(file_path, "rb").read()
    duck.sync(db="test", clear_pg=True)
    duck.insert_train_step_metric(model_id, 0, "metric", 2, 0.5)
    duck.sync(db="test")


if __name__ == "__main__":
    test_db()
    test_sync()
