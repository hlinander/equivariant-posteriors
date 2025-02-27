from lib.train import create_initial_state
import lib.render_duck as duck


def test_db():
    duck.ensure_duck(None, True)
    from test import create_train_run

    train_run = create_train_run()
    state = create_initial_state(train_run, None, "cpu")
    model_id = duck.insert_model(train_run.train_config)
    duck.insert_model_parameter(model_id, "test", 0)
    duck.insert_model_parameter(model_id, "test", 0.5)
    duck.insert_model_parameter(model_id, "test", "test")
    duck.insert_train_step(model_id, 1, "dataset", [0])
    duck.insert_train_step_metric(model_id, "metric", 1, 0.5)
    duck.insert_train_step_metric(model_id, "metric", 1, [0.5])

    duck.render_duck(train_run, state)
    duck.sync()
    duck.insert_train_step_metric(model_id, "metric", 2, 0.5)
    print("Sync 2")
    duck.sync()


if __name__ == "__main__":
    test_db()
