from pathlib import Path
from typing import Optional
import pandas
import json
import duckdb
import os
from typing import List
from dataclasses import dataclass
from lib.train_dataclasses import TrainRun, TrainEpochState, TrainConfig
from lib.paths import (
    get_or_create_checkpoint_path,
)
from lib.random_util import random_positive_i64
from lib.stable_hash import stable_hash_str
from lib.compute_env import env
import time

CONN = None
LAST_MODEL_ID = None
LAST_RUN_CONFIG = None
SCHEMA_ENSURED = False


INT = "int"
FLOAT = "float"
TEXT = "text"

TYPES = [INT, FLOAT, TEXT]

MODEL_PARAMETER = "model_parameter"
TRAIN_STEP_METRIC = "train_step_metric"
TRAIN_STATE = "train_state"
CHECKPOINT_SAMPLE_METRIC = "checkpoint_sample_metric"
CHECKPOINT_PARAMETER = "checkpoint_parameter"

MODELS_TABLE_NAME = "models"
TRAIN_STEPS_TABLE_NAME = "train_steps"
CHECKPOINTS_TABLE_NAME = "checkpoints"
ARTIFACTS_TABLE_NAME = "artifacts"
ARTIFACT_CHUNKS_TABLE_NAME = "artifact_chunks"
LOG_TABLE_NAME = "logs"

RUNS_TABLE_NAME = "runs"


@dataclass
class TypeDef:
    name: str
    sql_type: str
    python_type: type


TYPE_DEFS = [
    TypeDef(INT, "BIGINT", int),
    TypeDef(FLOAT, "FLOAT", float),
    TypeDef(TEXT, "TEXT", str),
]

PYTHON_TYPE_TO_TYPE_DEF = {typedef.python_type: typedef for typedef in TYPE_DEFS}


def table_name(kind, type_name):
    return f"{kind}_{type_name}"


ALL_TABLES = [
    MODELS_TABLE_NAME,
    RUNS_TABLE_NAME,
    TRAIN_STEPS_TABLE_NAME,
    CHECKPOINTS_TABLE_NAME,
    ARTIFACTS_TABLE_NAME,
    ARTIFACT_CHUNKS_TABLE_NAME,
    MODEL_PARAMETER,
    TRAIN_STEP_METRIC,
    CHECKPOINT_SAMPLE_METRIC,
]


def sql_create_table_runs():
    return f"""
                CREATE TABLE IF NOT EXISTS {RUNS_TABLE_NAME} (
                    id BIGINT,
                    model_id BIGINT,
                    timestamp TIMESTAMPTZ,
                )
    """


def insert_run(run_id, model_id):
    sql_insert_model = """
    INSERT INTO runs (id, model_id, timestamp) VALUES (?, ?, now())
    """
    execute(sql_insert_model, (run_id, model_id))


def sql_create_table_artifacts():
    return """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id BIGINT,
                    timestamp TIMESTAMPTZ,
                    model_id BIGINT,
                    name text,
                    path text,
                    type text,
                    size int
                )
    """


def sql_create_table_logs():
    return f"""
                CREATE TABLE IF NOT EXISTS {LOG_TABLE_NAME}(
                    run_id BIGINT,
                    timestamp TIMESTAMPTZ,
                    context text,
                    level text,
                    msg text
                )
    """


def insert_log(run_id: int, context: str, level: str, msg: str):
    sql_insert_log = f"""
        INSERT INTO {LOG_TABLE_NAME} (run_id, timestamp, context, level, msg) 
        VALUES (?, ?, ?, ?, ?)
    """
    execute(sql_insert_log, (run_id, time.time(), context, level, msg))


def sql_create_table_artifact_chunks():
    return """
                CREATE TABLE IF NOT EXISTS artifact_chunks (
                    artifact_id bigint,
                    seq_num int,
                    data bytea,
                    size int,
                    timestamp TIMESTAMPTZ
                )
                """


def insert_artifact(
    model_id: int, name: str, path: Path, type: Optional[str] = None
) -> int:
    try:
        return _insert_artifact(model_id, name, path, type)
    except duckdb.duckdb.ConstraintException as e:
        print(e)
        print(f"Artifact {name} already present for {model_id}")
        return None


def _insert_artifact(
    model_id: int, name: str, path: Path, type: Optional[str] = None
) -> int:
    path = Path(path) if not isinstance(path, Path) else path
    size_bytes = path.stat().st_size
    path_str = path.absolute().as_posix()
    if type is None:
        type = path.suffix

    artifact_id = random_positive_i64()

    execute(
        """
            INSERT INTO artifacts (id, timestamp, model_id, name, path, type, size)
            VALUES (?, now(), ?, ?, ?, ?, ?)
            """,
        (artifact_id, model_id, name, path_str, type, size_bytes),
    )
    print("[db] Uploading artifact")
    with path.open("rb") as file:
        seq_num = 0
        while chunk := file.read(1024 * 1024):
            execute(
                "INSERT INTO artifact_chunks (artifact_id, seq_num, data, size, timestamp) VALUES (?, ?, ?, ?, now())",
                (artifact_id, seq_num, chunk, len(chunk)),
            )
            seq_num += 1
            print(f"[db] Chunk {seq_num}")

    return artifact_id


def get_artifact(artifact_id):
    chunks = execute_and_fetch(
        "SELECT data FROM artifact_chunks WHERE artifact_id = ? ORDER BY seq_num",
        (artifact_id,),
    )
    artifact_data = b"".join(chunk for (chunk,) in chunks)
    return artifact_data


def sql_create_table_model_parameter():
    return f"""
        CREATE TABLE IF NOT EXISTS {MODEL_PARAMETER} (
            model_id BIGINT,
            run_id BIGINT,
            timestamp TIMESTAMPTZ,
            name TEXT,
            type TEXT,
            value_int BIGINT,
            value_float FLOAT,
            value_text TEXT
        )"""


def insert_model_parameter(model_id, run_id, name, value):
    value_type = type(value)
    if value_type in PYTHON_TYPE_TO_TYPE_DEF:
        type_def = PYTHON_TYPE_TO_TYPE_DEF[value_type]
    else:
        insert_model_parameter(model_id, run_id, name, str(value))
        return

    # Set the appropriate value column based on type
    value_int = value if type_def.name == INT else None
    value_float = value if type_def.name == FLOAT else None
    value_text = value if type_def.name == TEXT else None

    sql_insert_model_parameter = f"""
        INSERT INTO {MODEL_PARAMETER} (model_id, run_id, name, type, value_int, value_float, value_text, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, now())
    """
    execute(sql_insert_model_parameter, (model_id, run_id, name, type_def.name, value_int, value_float, value_text))


def sql_create_table_train_step_metric():
    return f"""
        CREATE TABLE IF NOT EXISTS {TRAIN_STEP_METRIC} (
            model_id BIGINT,
            run_id BIGINT,
            timestamp TIMESTAMPTZ,
            name TEXT,
            step INTEGER,
            type TEXT,
            value_int BIGINT,
            value_float FLOAT,
            value_text TEXT
        )"""


def insert_train_step_metric(model_id, run_id, name, step, value):
    value_type = type(value)
    if value_type in PYTHON_TYPE_TO_TYPE_DEF:
        type_def = PYTHON_TYPE_TO_TYPE_DEF[value_type]
    else:
        insert_train_step_metric(model_id, run_id, name, step, str(value))
        return

    # Set the appropriate value column based on type
    value_int = value if type_def.name == INT else None
    value_float = value if type_def.name == FLOAT else None
    value_text = value if type_def.name == TEXT else None

    sql_insert_train_step_metric = f"""
        INSERT INTO {TRAIN_STEP_METRIC} (model_id, run_id, timestamp, name, step, type, value_int, value_float, value_text)
        VALUES (?, ?, now(), ?, ?, ?, ?, ?, ?)
    """
    execute(sql_insert_train_step_metric, (model_id, run_id, name, step, type_def.name, value_int, value_float, value_text))


def select_train_step_metric_float(model_id, name):
    sql_select = f"""
        SELECT * FROM (
        SELECT * FROM (SELECT step, value_float as value FROM {TRAIN_STEP_METRIC}
        WHERE model_id=? AND name=? AND type=? ORDER BY step)
        USING SAMPLE 1000 ROWS)
        ORDER BY step
        """
    return execute_and_fetch(sql_select, (model_id, name, FLOAT))


def sql_create_table_checkpoint_sample_metric():
    return f"""
        CREATE TABLE IF NOT EXISTS {CHECKPOINT_SAMPLE_METRIC} (
            model_id BIGINT,
            timestamp TIMESTAMPTZ,
            step INTEGER,
            name TEXT,
            dataset TEXT,
            sample_ids INTEGER[],
            type TEXT,
            mean_int BIGINT,
            mean_float FLOAT,
            mean_text TEXT,
            value_per_sample_int BIGINT[],
            value_per_sample_float FLOAT[],
            value_per_sample_text TEXT[]
        )"""


def insert_checkpoint_sample_metric(
    model_id, step, name, dataset, sample_ids, mean, value_per_sample, db_prefix=""
):
    value_type = type(mean)
    if value_type in PYTHON_TYPE_TO_TYPE_DEF:
        type_def = PYTHON_TYPE_TO_TYPE_DEF[value_type]
    else:
        insert_checkpoint_sample_metric(
            model_id, step, name, dataset, sample_ids, str(mean), str(value_per_sample)
        )
        return

    # Set the appropriate value columns based on type
    mean_int = mean if type_def.name == INT else None
    mean_float = mean if type_def.name == FLOAT else None
    mean_text = mean if type_def.name == TEXT else None

    value_per_sample_int = value_per_sample if type_def.name == INT else None
    value_per_sample_float = value_per_sample if type_def.name == FLOAT else None
    value_per_sample_text = value_per_sample if type_def.name == TEXT else None

    sql_insert_checkpoint_sample_metric = f"""
        INSERT INTO {db_prefix}{CHECKPOINT_SAMPLE_METRIC} (model_id, step, name, dataset, sample_ids, type, mean_int, mean_float, mean_text, value_per_sample_int, value_per_sample_float, value_per_sample_text, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now())
    """
    execute(
        sql_insert_checkpoint_sample_metric,
        (model_id, step, name, dataset, sample_ids, type_def.name, mean_int, mean_float, mean_text, value_per_sample_int, value_per_sample_float, value_per_sample_text),
    )


def sql_create_table_models():
    return f"""
    CREATE TABLE IF NOT EXISTS {MODELS_TABLE_NAME} (
        id BIGINT,
        train_id TEXT,
        timestamp TIMESTAMPTZ,
    )"""


def insert_model(train_run: TrainRun):
    model_id = random_positive_i64()
    return insert_model_with_model_id(train_run, model_id)


def insert_model_with_model_id(train_run: TrainRun, model_id: int):
    ensure_duck(train_run)
    # insert_or_update_train_run(train_run, model_id)
    train_id = stable_hash_str(train_run.train_config)

    sql_insert_model = """
    INSERT INTO models (id, train_id, timestamp) VALUES (?, ?, now())
    """
    execute(sql_insert_model, (model_id, train_id))
    return model_id


def sql_create_table_train_steps():
    return f"""
    CREATE TABLE IF NOT EXISTS {TRAIN_STEPS_TABLE_NAME} (
        model_id BIGINT,
        run_id BIGINT,
        step INTEGER,
        dataset TEXT,
        sample_ids INTEGER[],
        timestamp TIMESTAMPTZ
    )
    """


def insert_train_step(
    model_id: int, run_id: int, step: int, dataset: str, sample_ids: List[int]
):
    sql_insert_train_step = """
        INSERT INTO train_steps (model_id, run_id, step, dataset, sample_ids, timestamp)
        VALUES (?, ?, ?, ?, ?, now())
    """
    # data = [(model_id, step, dataset, sample_id) for sample_id in sample_ids]
    execute(
        sql_insert_train_step,
        [model_id, run_id, step, dataset, sample_ids],
    )


def sql_create_table_checkpoints():
    return f"""
    CREATE TABLE IF NOT EXISTS {CHECKPOINTS_TABLE_NAME} (
        model_id BIGINT ,
        step INTEGER ,
        path TEXT,
        timestamp TIMESTAMPTZ
            )
    """


def insert_checkpoint(model_id: int, step: int, path: str, db_prefix=""):
    sql_insert_train_step = f"""
        INSERT INTO {db_prefix}{CHECKPOINTS_TABLE_NAME} (model_id, step, path, timestamp)
        VALUES (?, ?, ?, now())
       -- ON CONFLICT (model_id, step)
       -- DO UPDATE SET path=EXCLUDED.path
    """
    # data = [(model_id, step, dataset, sample_id) for sample_id in sample_ids]
    execute(
        sql_insert_train_step,
        [model_id, step, path],
    )


def get_checkpoints(model_id: int):
    sql_get_checkpoints = f"""
        SELECT * FROM {CHECKPOINTS_TABLE_NAME}
        WHERE model_id=?
    """
    # data = [(model_id, step, dataset, sample_id) for sample_id in sample_ids]
    return execute_and_fetch(
        sql_get_checkpoints,
        [model_id],
    )


def execute(sql, params=None):
    try:
        return CONN.execute(sql, params)
    except Exception as e:
        # print(sql)
        raise e


def execute_many(sql, params=None):
    try:
        return CONN.executemany(sql, params)
    except Exception as e:
        # print(sql)
        raise e


def execute_and_fetch(sql, params=None):
    try:
        return CONN.execute(sql, params).fetchall()
    except Exception as e:
        # print(sql)
        raise e


def _ensure_schema(executor=execute):
    executor(sql_create_table_models())
    executor(sql_create_table_runs())
    executor(sql_create_table_train_steps())
    executor(sql_create_table_checkpoints())
    executor(sql_create_table_artifacts())
    executor(sql_create_table_artifact_chunks())
    executor(sql_create_table_model_parameter())
    executor(sql_create_table_checkpoint_sample_metric())
    executor(sql_create_table_train_step_metric())


def ensure_duck(run_run: Optional[TrainRun] = None, in_memory=False):
    global CONN
    global SCHEMA_ENSURED

    if run_run is None or in_memory:
        db_path = ":memory:"
    else:
        db_path = (
            get_or_create_checkpoint_path(run_run.train_config)
            / f"duck_{run_run.run_id:x}.db"
        )
    if CONN is None:
        print("Connecting to duck...")
        CONN = duckdb.connect()
        CONN.sql(f"ATTACH '{db_path}' as local")
        CONN.sql("USE local")
        CONN.sql("LOAD icu")
        # CONN = duckdb.connect(db_path)
        # CONN = duckdb.connect()
        print("Connected.")

    if not SCHEMA_ENSURED:
        _ensure_schema()


def dict_to_normalized_json(input_dict):
    return json.loads(pandas.json_normalize(input_dict).to_json(orient="records"))[0]


def insert_or_update_train_run(train_run: TrainRun, model_id: int):
    train_run_flat = dict_to_normalized_json(train_run.serialize_human())
    insert_run(train_run.run_id, model_id)
    insert_model_parameter(
        model_id, train_run.run_id, "train_config_hash", train_run_flat["train_id"]
    )
    insert_model_parameter(model_id, train_run.run_id, "model_id", model_id)
    for key, value in train_run_flat.items():
        insert_model_parameter(
            model_id,
            train_run.run_id,
            key,
            value,
        )


def render_duck(
    train_run: TrainRun, train_epoch_state: TrainEpochState, in_memory=False
):
    global LAST_MODEL_ID
    ensure_duck(train_run, in_memory)
    insert_or_update_train_run(train_run, train_epoch_state.model_id)
    insert_model_parameter(
        train_epoch_state.model_id,
        train_run.run_id,
        "train_dataset_len",
        len(train_epoch_state.train_dataloader.dataset),
    )
    if train_epoch_state.val_dataloader is not None:
        insert_model_parameter(
            train_epoch_state.model_id,
            train_run.run_id,
            "val_dataset_len",
            len(train_epoch_state.val_dataloader.dataset),
        )
    LAST_MODEL_ID = train_epoch_state.model_id


def start_periodic_export(
    train_run: TrainRun,
    interval_seconds: int = 300,
    s3_bucket: Optional[str] = None,
):
    """
    Start a background thread that periodically exports data to S3

    Args:
        train_run: The training run context
        interval_seconds: How often to export (default: 300 = 5 minutes)
        s3_bucket: S3 bucket name (defaults to env().s3_bucket)

    Returns:
        The background thread handle

    Example:
        # In your training script
        export_thread = start_periodic_export(train_run, interval_seconds=300)
    """
    from lib.export_parquet import export_periodic
    return export_periodic(train_run, interval_seconds, s3_bucket)
