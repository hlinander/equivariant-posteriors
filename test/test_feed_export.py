from dataclasses import dataclass
from pathlib import Path
import sys

import pytest

import lib.render_duck as duck
from lib.analytics_config import FeedTarget
from lib.export_feed import FeedExportError, FeedExporter

sys.path.insert(0, str(Path(__file__).parent))
from conftest import create_train_run


@dataclass(frozen=True)
class _Report:
    delivered: int = 0
    filtered: int = 0
    dropped: int = 0
    pending: int = 0
    complete: bool = True

    @property
    def successful(self):
        return self.complete and self.dropped == 0


class _FakeRun:
    id = "fake-session"

    def __init__(self, reports=None):
        self.events = []
        self.reports = list(reports or [])
        self.finished = False

    def log_wait(self, stream_name, record, *, timeout):
        self.events.append(
            {
                "stream_name": stream_name,
                "data": dict(record),
                "timeout": timeout,
            }
        )
        return True

    def flush(self, timeout):
        if self.reports:
            return self.reports.pop(0)
        return _Report()

    def finish(self, timeout):
        self.finished = True
        return _Report()


@pytest.fixture
def local_duck():
    if duck.CONN is not None:
        duck.CONN.close()
    duck.CONN = None
    duck.SCHEMA_ENSURED = False
    duck.ensure_duck(None, True)
    yield
    if duck.CONN is not None:
        duck.CONN.close()
    duck.CONN = None
    duck.SCHEMA_ENSURED = False


def _local_run():
    train_run = create_train_run()
    model_id = duck.insert_model(train_run)
    duck.insert_run(train_run.run_id, model_id)
    return train_run, model_id


def test_train_runs_receive_distinct_default_ids():
    assert create_train_run().run_id != create_train_run().run_id


def test_exports_local_tables_and_advances_dedicated_cursors(local_duck):
    train_run, model_id = _local_run()
    duck.insert_model_parameter(model_id, train_run.run_id, "learning_rate", 0.001)
    duck.insert_train_step_metric(model_id, train_run.run_id, "loss", 1, 0.5)
    duck.insert_train_epoch_metric(
        model_id,
        train_run.run_id,
        1,
        1,
        "loss",
        "TestDataset",
        "train",
        0.5,
        0.4,
        0.6,
        10,
    )
    duck.insert_checkpoint_sample_metric(
        model_id,
        1,
        "accuracy",
        "TestDataset",
        [10, 11],
        0.75,
        [1.0, 0.5],
    )
    duck.insert_train_step(model_id, train_run.run_id, 1, "TestDataset", [10, 11])
    duck.insert_checkpoint(model_id, 1, None)

    run = _FakeRun()
    exporter = FeedExporter(
        train_run,
        FeedTarget(project="org/project", chunk_size=2),
        duck.CONN,
        run=run,
    )

    assert exporter.export_pending() == 6
    names = [event["stream_name"] for event in run.events]
    assert "model_parameters" in names
    assert names.count("metric") == 3
    assert "epoch_metrics" in names
    assert "evaluation_samples" in names
    assert "train_steps" in names
    assert "checkpoints" in names

    metric = next(
        event
        for event in run.events
        if event["stream_name"] == "metric" and event["data"]["kind"] == "metric"
    )
    assert metric["data"]["model_id"] == model_id
    assert metric["data"]["eqp_run_id"] == train_run.run_id
    assert metric["data"]["metric"] == "loss"
    assert metric["data"]["value"] == 0.5

    parameter = next(
        event for event in run.events if event["stream_name"] == "model_parameters"
    )
    assert parameter["data"]["name"] == "learning_rate"
    assert parameter["data"]["value_type"] == "float"
    assert parameter["data"]["value_float"] == pytest.approx(0.001)

    sync_keys = {
        row[0]
        for row in duck.CONN.execute(
            "SELECT table_name FROM sync_state WHERE table_name LIKE 'feed:%'"
        ).fetchall()
    }
    assert sync_keys == {
        "feed:model_parameter",
        "feed:train_step_metric",
        "feed:train_epoch_metric",
        "feed:checkpoint_sample_metric",
        "feed:train_steps",
        "feed:checkpoints",
    }

    event_count = len(run.events)
    assert exporter.export_pending() == 0
    assert len(run.events) == event_count


def test_timeout_reuses_pending_events_instead_of_reemitting_rows(local_duck):
    train_run, model_id = _local_run()
    duck.insert_train_step_metric(model_id, train_run.run_id, "loss", 1, 0.5)
    run = _FakeRun(
        reports=[
            _Report(pending=1, complete=False),
            _Report(delivered=1),
        ]
    )
    exporter = FeedExporter(
        train_run,
        FeedTarget(project="org/project"),
        duck.CONN,
        run=run,
    )

    with pytest.raises(FeedExportError, match="original wire identities"):
        exporter.export_pending()
    assert len(run.events) == 1
    assert (
        duck.CONN.execute(
            "SELECT COUNT(*) FROM sync_state WHERE table_name = 'feed:train_step_metric'"
        ).fetchone()[0]
        == 0
    )

    assert exporter.export_pending() == 1
    assert len(run.events) == 1
    assert (
        duck.CONN.execute(
            "SELECT COUNT(*) FROM sync_state WHERE table_name = 'feed:train_step_metric'"
        ).fetchone()[0]
        == 1
    )


def test_chunk_boundary_keeps_all_rows_with_the_same_timestamp(local_duck):
    train_run, model_id = _local_run()
    for step in range(3):
        duck.insert_train_step_metric(
            model_id, train_run.run_id, "loss", step, 1.0 / (step + 1)
        )
    duck.CONN.execute(
        """
        UPDATE train_step_metric
        SET timestamp = TIMESTAMPTZ '2026-01-01 00:00:00+00'
        WHERE run_id = ?
        """,
        (train_run.run_id,),
    )

    run = _FakeRun()
    exporter = FeedExporter(
        train_run,
        FeedTarget(project="org/project", chunk_size=2),
        duck.CONN,
        run=run,
    )

    assert exporter.export_pending() == 3
    assert len(run.events) == 3
    assert exporter.export_pending() == 0
    assert len(run.events) == 3
