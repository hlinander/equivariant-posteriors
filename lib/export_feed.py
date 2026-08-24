"""Recurring export of local EQP analytics through Feed.

The local DuckDB remains the training process's recorder and checkpoint source.
This module owns one Feed session per EQP run, streams rows added since its
own sync cursor, and advances that cursor only after an acknowledged flush.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from lib.analytics_config import FeedTarget
from lib.log import log, log_next_in
from lib.train_dataclasses import TrainRun


class FeedExportError(RuntimeError):
    """The exporter could not establish acknowledged delivery."""


@dataclass(frozen=True)
class _TableSpec:
    name: str
    filter_column: str
    filter_value: int


@dataclass
class _PendingExport:
    table_name: str
    boundary: float
    row_count: int = 0


class FeedExporter:
    """Stateful bridge between one local EQP run and one Feed session."""

    def __init__(
        self,
        train_run: TrainRun,
        target: FeedTarget,
        cursor,
        *,
        run=None,
        run_factory: Optional[Callable[..., object]] = None,
    ) -> None:
        self.train_run = train_run
        self.target = target
        if target.chunk_size <= 0:
            raise ValueError("FeedTarget.chunk_size must be positive")
        self.cursor = cursor
        self.cursor.execute("USE local")
        self.model_id = self._resolve_model_id()
        self._lock = threading.Lock()
        self._pending: Optional[_PendingExport] = None
        self._failure: Optional[str] = None
        self._closed = False
        self.run = run or self._create_run(run_factory)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def reference(self) -> str:
        return f"feed://{self.target.project}/{self.run.id}"

    def export_pending(self, *, finish: bool = False, cursor=None) -> int:
        """Publish new rows and acknowledge them before advancing sync cursors."""
        with self._lock:
            if cursor is not None:
                self.cursor = cursor
                self.cursor.execute("USE local")
            if self._closed:
                return 0
            if self._failure is not None:
                raise FeedExportError(self._failure)

            delivered_rows = self._settle_pending()
            cutoff = time.time()
            specs = self._table_specs()
            for spec in specs:
                delivered_rows += self._emit_table(spec, cutoff)

            # A fresh run may only have its run-metadata event. Flush it even if
            # none of the local metric tables contains rows yet.
            if delivered_rows == 0:
                self._flush_without_cursor()

            if finish:
                report = self.run.finish(self.target.flush_timeout_seconds)
                if not report.successful:
                    raise FeedExportError(
                        _report_error("Feed final flush failed", report)
                    )
                self._closed = True

            return delivered_rows

    def _settle_pending(self) -> int:
        if self._pending is None:
            return 0
        report = self.run.flush(self.target.flush_timeout_seconds)
        if not report.complete:
            raise FeedExportError(
                _report_error(
                    "Feed flush timed out; events remain pending with their "
                    "original wire identities",
                    report,
                )
            )
        if not report.successful:
            self._failure = _report_error(
                "Feed permanently dropped an export batch", report
            )
            raise FeedExportError(self._failure)

        pending = self._pending
        self.cursor.execute(
            """
            INSERT INTO sync_state (table_name, last_synced_timestamp)
            VALUES (?, ?)
            ON CONFLICT (table_name)
            DO UPDATE SET last_synced_timestamp = EXCLUDED.last_synced_timestamp
            """,
            (self._sync_key(pending.table_name), pending.boundary),
        )
        self._pending = None
        return pending.row_count

    def _flush_without_cursor(self) -> None:
        report = self.run.flush(self.target.flush_timeout_seconds)
        if not report.complete:
            raise FeedExportError(
                _report_error("Feed metadata flush timed out", report)
            )
        if not report.successful:
            self._failure = _report_error(
                "Feed permanently dropped run metadata", report
            )
            raise FeedExportError(self._failure)

    def _emit_table(self, spec: _TableSpec, cutoff: float) -> int:
        delivered_rows = 0
        while True:
            rows, columns, boundary = self._next_chunk(spec, cutoff)
            if not rows:
                break
            assert boundary is not None
            self._pending = _PendingExport(
                table_name=spec.name,
                boundary=boundary,
            )
            try:
                for values in rows:
                    row = dict(zip(columns, values))
                    self._emit_row(spec.name, row)
                    self._pending.row_count += 1
            except Exception as error:
                if self._pending.row_count > 0:
                    self._failure = (
                        "Feed export stopped after a partial enqueue; the local "
                        f"cursor was retained: {error}"
                    )
                raise
            delivered_rows += self._settle_pending()
        return delivered_rows

    def _next_chunk(self, spec: _TableSpec, cutoff: float):
        last_synced = self._last_synced(spec.name)
        base_sql = f"""
            SELECT * FROM {spec.name}
            WHERE {spec.filter_column} = ?
              AND EPOCH(timestamp) > ?
              AND EPOCH(timestamp) <= ?
        """
        preview = self.cursor.execute(
            f"{base_sql} ORDER BY timestamp LIMIT ?",
            (spec.filter_value, last_synced, cutoff, self.target.chunk_size),
        )
        columns = [column[0] for column in preview.description]
        rows = preview.fetchall()
        if not rows:
            return [], columns, None

        timestamp_index = columns.index("timestamp")
        boundary = max(_timestamp(row[timestamp_index]) for row in rows)
        if len(rows) == self.target.chunk_size:
            # Include every row tied at the boundary timestamp. Advancing a
            # timestamp-only cursor after a partial tie would otherwise lose rows.
            complete_chunk = self.cursor.execute(
                f"{base_sql} AND EPOCH(timestamp) <= ? ORDER BY timestamp",
                (spec.filter_value, last_synced, cutoff, boundary),
            )
            columns = [column[0] for column in complete_chunk.description]
            rows = complete_chunk.fetchall()
        return rows, columns, boundary

    def _emit_row(self, table_name: str, row: dict) -> None:
        from lib.render_duck import (
            CHECKPOINT_SAMPLE_METRIC,
            CHECKPOINTS_TABLE_NAME,
            MODEL_PARAMETER,
            TRAIN_EPOCH_METRIC,
            TRAIN_STEP_METRIC,
            TRAIN_STEPS_TABLE_NAME,
        )

        if table_name == MODEL_PARAMETER:
            self._emit_fields(
                "model_parameters",
                "run",
                lambda builder: (
                    builder.add_int("model_id", row["model_id"])
                    .add_int("eqp_run_id", row["run_id"])
                    .add_string("name", row["name"])
                    .add_variant("value", _typed_value(row, "value"))
                    .add_float("source_timestamp", _timestamp(row["timestamp"]))
                ),
            )
            return

        if table_name == TRAIN_STEP_METRIC:
            value = _typed_value(row, "value")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._emit_metric(
                    metric=row["name"],
                    value=float(value),
                    step=row["step"],
                    kind="metric",
                    context="train",
                    row=row,
                )
            else:
                self._emit_variant_metric(row, value)
            return

        if table_name == TRAIN_EPOCH_METRIC:
            context = f"{row['dataset_split']}:{row['dataset']}"
            self._emit_metric(
                metric=row["name"],
                value=float(row["mean"]),
                step=row["step"],
                kind="epoch",
                context=context,
                row=row,
            )
            self._emit_fields(
                "epoch_metrics",
                "evaluation" if row["dataset_split"] != "train" else "train",
                lambda builder: (
                    builder.add_int("model_id", row["model_id"])
                    .add_int("eqp_run_id", row["run_id"])
                    .add_int("epoch", row["epoch"])
                    .add_int("step", row["step"])
                    .add_string("metric", row["name"])
                    .add_string("dataset", row["dataset"])
                    .add_string("dataset_split", row["dataset_split"])
                    .add_float("mean", row["mean"])
                    .add_float("min", row["min"])
                    .add_float("max", row["max"])
                    .add_int("count", row["count"])
                    .add_float("source_timestamp", _timestamp(row["timestamp"]))
                ),
            )
            return

        if table_name == CHECKPOINT_SAMPLE_METRIC:
            mean = _typed_value(row, "mean")
            per_sample = _typed_value(row, "value_per_sample")
            if isinstance(mean, (int, float)) and not isinstance(mean, bool):
                self._emit_metric(
                    metric=row["name"],
                    value=float(mean),
                    step=row["step"],
                    kind="evaluation",
                    context=row["dataset"],
                    row=row,
                )
            self._emit_fields(
                "evaluation_samples",
                "evaluation",
                lambda builder: (
                    builder.add_int("model_id", row["model_id"])
                    .add_int("step", row["step"])
                    .add_string("metric", row["name"])
                    .add_string("dataset", row["dataset"])
                    .add_int_array("sample_ids", row["sample_ids"] or [])
                    .add_variant("mean", mean)
                    .add_variant("values", per_sample)
                    .add_float("source_timestamp", _timestamp(row["timestamp"]))
                ),
            )
            return

        if table_name == TRAIN_STEPS_TABLE_NAME:
            self._emit_fields(
                "train_steps",
                "train",
                lambda builder: (
                    builder.add_int("model_id", row["model_id"])
                    .add_int("eqp_run_id", row["run_id"])
                    .add_int("step", row["step"])
                    .add_string("dataset", row["dataset"])
                    .add_int_array("sample_ids", row["sample_ids"] or [])
                    .add_float("source_timestamp", _timestamp(row["timestamp"]))
                ),
            )
            return

        if table_name == CHECKPOINTS_TABLE_NAME:
            self._emit_fields(
                "checkpoints",
                "checkpoint",
                lambda builder: (
                    builder.add_int("model_id", row["model_id"])
                    .add_int("step", row["step"])
                    .add_optional_string("path", row["path"])
                    .add_float("source_timestamp", _timestamp(row["timestamp"]))
                ),
            )
            return

        raise FeedExportError(f"Unsupported Feed export table: {table_name}")

    def _emit_metric(
        self,
        *,
        metric: str,
        value: float,
        step: Optional[int],
        kind: str,
        context: Optional[str],
        row: dict,
    ) -> None:
        self._emit_fields(
            "metric",
            "evaluation" if kind in {"evaluation", "epoch"} else "train",
            lambda builder: (
                builder.add_string("metric", metric)
                .add_float("value", value)
                .add_optional_int("step", step)
                .add_string("kind", kind)
                .add_optional_string("context", context)
                .add_int("model_id", row["model_id"])
                .add_int("eqp_run_id", row.get("run_id", self.train_run.run_id))
                .add_float("source_timestamp", _timestamp(row["timestamp"]))
            ),
        )

    def _emit_variant_metric(self, row: dict, value) -> None:
        self._emit_fields(
            "metric_values",
            "train",
            lambda builder: (
                builder.add_int("model_id", row["model_id"])
                .add_int("eqp_run_id", row["run_id"])
                .add_string("metric", row["name"])
                .add_int("step", row["step"])
                .add_variant("value", value)
                .add_float("source_timestamp", _timestamp(row["timestamp"]))
            ),
        )

    def _emit_fields(self, schema_name: str, channel: str, build_fields) -> None:
        from feed import EventBuilder

        fields = build_fields(EventBuilder()).build()
        accepted = self.run.client.emit_on_wait(
            self.run.client.channel(channel),
            schema_name,
            fields,
            self.target.enqueue_timeout_seconds,
        )
        if not accepted:
            raise FeedExportError(
                f"Feed did not accept {schema_name!r} within "
                f"{self.target.enqueue_timeout_seconds}s"
            )

    def _last_synced(self, table_name: str) -> float:
        result = self.cursor.execute(
            "SELECT last_synced_timestamp FROM sync_state WHERE table_name = ?",
            (self._sync_key(table_name),),
        ).fetchone()
        if result is not None:
            return float(result[0])

        # Resumed analytics loaded from checkpoint have already been published by
        # the previous execution. Start after that checkpoint watermark.
        checkpoint_result = self.cursor.execute(
            "SELECT last_synced_timestamp FROM sync_state WHERE table_name = ?",
            (f"ckpt_{table_name}",),
        ).fetchone()
        return float(checkpoint_result[0]) if checkpoint_result is not None else 0.0

    def _sync_key(self, table_name: str) -> str:
        return f"feed:{table_name}"

    def _resolve_model_id(self) -> int:
        row = self.cursor.execute(
            "SELECT model_id FROM runs WHERE id = ? ORDER BY timestamp DESC LIMIT 1",
            (self.train_run.run_id,),
        ).fetchone()
        if row is None:
            raise FeedExportError(
                f"No local run metadata found for run_id={self.train_run.run_id}"
            )
        return int(row[0])

    def _create_run(self, run_factory: Optional[Callable[..., object]]):
        if run_factory is None:
            try:
                from feed import init as run_factory
            except ImportError as error:
                raise FeedExportError(
                    "FeedTarget requires feed-python; install the EQP "
                    "feed optional dependency"
                ) from error

        serialized = self.train_run.serialize_human()
        config = {
            "eqp": {
                "project": self.train_run.project,
                "run_id": self.train_run.run_id,
                "model_id": self.model_id,
                "train_id": serialized["train_id"],
                "ensemble_id": serialized["ensemble_id"],
            },
            "train_config": serialized["train_config"],
            "train_eval": serialized["train_eval"],
            "epochs": serialized["epochs"],
            "git_rev": serialized["git_rev"],
            "slurm_jobid": serialized["slurm_jobid"],
        }
        return run_factory(
            project=self.target.project,
            server_url=self.target.server_url,
            name=f"{self.train_run.project}-{self.model_id}",
            config=config,
            tags=["eqp"],
            group=self.train_run.project,
            max_retries=0,
            max_retry_queue_depth=0,
        )

    def _table_specs(self) -> list[_TableSpec]:
        from lib.render_duck import (
            CHECKPOINT_SAMPLE_METRIC,
            CHECKPOINTS_TABLE_NAME,
            MODEL_PARAMETER,
            TRAIN_EPOCH_METRIC,
            TRAIN_STEP_METRIC,
            TRAIN_STEPS_TABLE_NAME,
        )

        return [
            _TableSpec(MODEL_PARAMETER, "run_id", self.train_run.run_id),
            _TableSpec(TRAIN_STEP_METRIC, "run_id", self.train_run.run_id),
            _TableSpec(TRAIN_EPOCH_METRIC, "run_id", self.train_run.run_id),
            _TableSpec(CHECKPOINT_SAMPLE_METRIC, "model_id", self.model_id),
            _TableSpec(TRAIN_STEPS_TABLE_NAME, "run_id", self.train_run.run_id),
            _TableSpec(CHECKPOINTS_TABLE_NAME, "model_id", self.model_id),
        ]


_EXPORTERS: dict[int, FeedExporter] = {}
_EXPORTERS_LOCK = threading.Lock()


def get_feed_exporter(
    train_run: TrainRun, target: FeedTarget, cursor
) -> FeedExporter:
    with _EXPORTERS_LOCK:
        exporter = _EXPORTERS.get(train_run.run_id)
        if exporter is None or exporter.closed:
            exporter = FeedExporter(train_run, target, cursor)
            _EXPORTERS[train_run.run_id] = exporter
        return exporter


def finish_feed_export(
    train_run: TrainRun, target: FeedTarget, cursor
) -> Optional[str]:
    exporter = get_feed_exporter(train_run, target, cursor)
    count = exporter.export_pending(finish=True, cursor=cursor)
    with _EXPORTERS_LOCK:
        _EXPORTERS.pop(train_run.run_id, None)
    log("export", f"Delivered {count} local analytics rows to {exporter.reference}")
    return exporter.reference


def export_periodic_feed(
    train_run: TrainRun,
    target: FeedTarget,
    interval_seconds: float,
):
    import lib.render_duck as duck

    def export_loop():
        if duck.CONN is None:
            log("export", "Error: DuckDB connection not initialized")
            return
        cursor = duck.CONN.cursor()
        exporter = get_feed_exporter(train_run, target, cursor)

        while not exporter.closed:
            try:
                count = exporter.export_pending(cursor=cursor)
                if count:
                    log_next_in(
                        "export",
                        f"Delivered {count} analytics rows to Feed",
                        interval_seconds,
                    )
            except Exception as error:
                log("export", f"Feed export pending: {error}")

            try:
                _flush_checkpoint_analytics(train_run, cursor)
            except Exception as error:
                log("export", f"Error during checkpoint export: {error}")

            time.sleep(interval_seconds)

    thread = threading.Thread(target=export_loop, daemon=True)
    thread.start()
    return thread


def _flush_checkpoint_analytics(train_run: TrainRun, cursor) -> None:
    from lib.paths import get_or_create_checkpoint_path
    from lib.staging_filesystem import flush_all_to_checkpoint

    checkpoint_path = get_or_create_checkpoint_path(train_run.train_config)
    flush_all_to_checkpoint(train_run, checkpoint_path, cursor)


def _typed_value(row: dict, prefix: str):
    value_type = row["type"]
    return row[f"{prefix}_{value_type}"]


def _timestamp(value) -> float:
    if hasattr(value, "timestamp"):
        return float(value.timestamp())
    return float(value)


def _report_error(prefix: str, report) -> str:
    return (
        f"{prefix}: delivered={report.delivered}, filtered={report.filtered}, "
        f"dropped={report.dropped}, pending={report.pending}"
    )
