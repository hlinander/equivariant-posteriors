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
PG_SCHEMA_ENSURED = False


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


ALL_TABLES = (
    [
        MODELS_TABLE_NAME,
        RUNS_TABLE_NAME,
        TRAIN_STEPS_TABLE_NAME,
        CHECKPOINTS_TABLE_NAME,
        ARTIFACTS_TABLE_NAME,
        ARTIFACT_CHUNKS_TABLE_NAME,
    ]
    + [table_name(TRAIN_STEP_METRIC, type_def.name) for type_def in TYPE_DEFS]
    + [table_name(MODEL_PARAMETER, type_def.name) for type_def in TYPE_DEFS]
    + [table_name(CHECKPOINT_SAMPLE_METRIC, type_def.name) for type_def in TYPE_DEFS]
)


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
                    size int
                )
                """


def insert_artifact(
    model_id: int, name: str, path: Path, type: Optional[str] = None
) -> int:
    try:
        _insert_artifact(model_id, name, path, type)
    except duckdb.duckdb.ConstraintException as e:
        print(e)
        print(f"Artifact {name} already present for {model_id}")


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
            INSERT INTO artifacts (id, model_id, name, path, type, size)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
        (artifact_id, model_id, name, path_str, type, size_bytes),
    )
    print("[db] Uploading artifact")
    with path.open("rb") as file:
        seq_num = 0
        while chunk := file.read(1024 * 1024):
            execute(
                "INSERT INTO artifact_chunks (artifact_id, seq_num, data, size) VALUES (?, ?, ?, ?)",
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


def sql_create_table_model_parameter(type_def):
    return f"""
        CREATE TABLE IF NOT EXISTS {table_name(MODEL_PARAMETER, type_def.name)} (
            model_id BIGINT,
            run_id BIGINT,
            timestamp TIMESTAMPTZ,
            name TEXT,
            value {type_def.sql_type}
        )"""


def insert_model_parameter(model_id, run_id, name, value):
    value_type = type(value)
    if value_type in PYTHON_TYPE_TO_TYPE_DEF:
        type_def = PYTHON_TYPE_TO_TYPE_DEF[value_type]
    else:
        insert_model_parameter(model_id, run_id, name, str(value))
        return

    sql_insert_model_parameter = f"""
        INSERT INTO {table_name(MODEL_PARAMETER, type_def.name)} (model_id, run_id, name, value, timestamp) 
        VALUES (?, ?, ?, ?, now())
    """
    execute(sql_insert_model_parameter, (model_id, run_id, name, value))


def sql_create_table_train_step_metric(type_def):
    return f"""
        CREATE TABLE IF NOT EXISTS {table_name(TRAIN_STEP_METRIC, type_def.name)} (
            model_id BIGINT,
            timestamp TIMESTAMPTZ,
            run_id BIGINT,
            name TEXT,
            step INTEGER,
            value {type_def.sql_type}
        )"""


def insert_train_step_metric(model_id, run_id, name, step, value):
    value_type = type(value)
    if value_type in PYTHON_TYPE_TO_TYPE_DEF:
        type_def = PYTHON_TYPE_TO_TYPE_DEF[value_type]
    else:
        insert_train_step_metric(model_id, run_id, name, step, str(value))
        return

    # raise Exception(type_def)
    sql_insert_train_step_metric = f"""
        INSERT INTO {table_name(TRAIN_STEP_METRIC, type_def.name)} (model_id, run_id, name, step, value, timestamp) 
        VALUES (?, ?, ?, ?, ?, now())
    """
    execute(sql_insert_train_step_metric, (model_id, run_id, name, step, value))


def select_train_step_metric_float(model_id, name):
    sql_select = f"""
        SELECT * FROM (
        SELECT * FROM (SELECT step, value FROM {table_name(TRAIN_STEP_METRIC, PYTHON_TYPE_TO_TYPE_DEF[float].name)}
        WHERE model_id=? AND name=? ORDER BY step)
        USING SAMPLE 1000 ROWS)
        ORDER BY step
        """
    return execute_and_fetch(sql_select, (model_id, name))


def sql_create_table_checkpoint_sample_metric(type_def):
    return f"""
        CREATE TABLE IF NOT EXISTS {table_name(CHECKPOINT_SAMPLE_METRIC, type_def.name)} (
            model_id BIGINT,
            timestamp TIMESTAMPTZ,
            step INTEGER,
            name TEXT,
            dataset TEXT,
            sample_ids INTEGER[],
            mean {type_def.sql_type},
            value_per_sample {type_def.sql_type}[]
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

    sql_insert_train_step_metric = f"""
        INSERT INTO {db_prefix}{table_name(CHECKPOINT_SAMPLE_METRIC, type_def.name)} (model_id, step, name, dataset, sample_ids, mean, value_per_sample, timestamp) 
        VALUES (?, ?, ?, ?, ?, ?, ?, now())
    """
    execute(
        sql_insert_train_step_metric,
        (model_id, step, name, dataset, sample_ids, mean, value_per_sample),
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


def insert_checkpoint_pg(model_id: int, step: int, path: str, db_prefix=""):
    sql_insert_train_step = f"""
        INSERT INTO {db_prefix}{CHECKPOINTS_TABLE_NAME} (model_id, step, path, timestamp)
        VALUES (?, ?, ?, now())
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


def sql_create_table_sync():
    return """
    CREATE TABLE IF NOT EXISTS sync (
        table_name TEXT,
        row_id BIGINT
        -- synced_time TIMESTAMP WITHOUT TIME ZONE
    )
    """


def attach_pg(db="equiv_v2"):
    hostname = env().postgres_host  # "localhost"
    port = env().postgres_port  # 5432
    pw = env().postgres_password

    if "DUCKLAKE" in os.environ:
        lake_name = os.environ["DUCKLAKE"]
        CONN.sql(
            f"ATTACH IF NOT EXISTS 'ducklake:{lake_name}' as pg (data_path './{lake_name}_data')"
        )
        print(f"Attached local ducklake at {lake_name}")
        return

    CONN.sql("INSTALL ducklake")
    CONN.sql("INSTALL aws")
    CONN.sql(
        f"""CREATE OR REPLACE SECRET (
             TYPE s3,
    PROVIDER config,
    KEY_ID '{env().s3_key}',
    SECRET '{env().s3_secret}',
    REGION '{env().s3_region}',
    ENDPOINT '{env().s3_endpoint}'
             )
        """
    )
    CONN.sql(
        f"ATTACH IF NOT EXISTS 'ducklake:postgres:user=postgres password={pw} host={hostname} port={port} dbname=ducklake' as pg (data_path 's3://eqpducklake')"
    )


def sync(train_run: Optional[TrainRun] = None, db="equiv_v2", clear_pg=False):
    global PG_SCHEMA_ENSURED
    import time

    start = time.time()
    total_time = 0
    ensure_duck(train_run)
    attach_pg(db)

    if clear_pg:
        if "CLEAR_POSTGRES" not in os.environ:
            print("Not clearing postgres without CLEAR_POSTGRES env set")
        else:
            for table_name in ALL_TABLES:
                execute(f"DROP TABLE pg.{table_name} CASCADE")

    if not PG_SCHEMA_ENSURED:
        # pg_stime = time.time()
        _ensure_schema(execute_ducklake)
        # print(f"pg schema {time.time() - pg_stime}")
        # for table_name in ALL_TABLES:
        #     # alter_time = time.time()
        #     execute_ducklake(
        #         f"""
        #         ALTER TABLE {table_name}
        #             ADD COLUMN IF NOT EXISTS id_serial SERIAL;
        #        """
        #     )
        # print(f"alter {table_name} {time.time() - alter_time}")

        PG_SCHEMA_ENSURED = True

    for table_name in ALL_TABLES:
        # table_time = time.time()
        last_row_id = execute(
            "SELECT MAX(row_id) FROM sync WHERE table_name=?", (table_name,)
        ).fetchone()
        if last_row_id[0] is None:
            synced_row_id = execute_and_fetch(
                f"SELECT MIN(rowid) - 1 as min_row FROM {table_name}"
            )
            if len(synced_row_id) == 1 and synced_row_id[0] != (None,):
                synced_row_id = int(synced_row_id[0][0])
            else:
                synced_row_id = None
        else:
            synced_row_id = int(last_row_id[0])

        next_synced_row_id = execute_and_fetch(
            f"SELECT MAX(rowid) as max_row FROM {table_name}"
        )
        if len(next_synced_row_id) == 1 and next_synced_row_id[0] != (None,):
            next_synced_row_id = int(next_synced_row_id[0][0])

        if synced_row_id is not None:
            try:
                execute(
                    f"""
                    INSERT INTO pg.{table_name} BY NAME SELECT * FROM {table_name}
                    WHERE rowid > '{synced_row_id}'
                    AND rowid <= '{next_synced_row_id}';
                    INSERT INTO sync (row_id, table_name)
                    VALUES ('{next_synced_row_id}', '{table_name}')
                    -- ON CONFLICT (table_name)
                    -- DO UPDATE SET row_id='{next_synced_row_id}'
                    """,
                )
            except duckdb.duckdb.Error as e:
                print(e)

    print(
        execute(
            "SELECT table_name, MAX(row_id) from sync group by table_name"
        ).fetch_df()
    )

    # t = time.time() - table_time
    # print(f"{t} for table {table_name}")
    # total_time += t

    print("Sync time", time.time() - start, total_time)


def execute_pg(sql, params=None):
    try:
        return CONN.execute(f"CALL postgres_execute('pg', '{sql}')")
    except Exception as e:
        # print(sql)
        raise e


def execute_ducklake(sql, params=None):
    try:
        CONN.sql("USE pg")
        res = CONN.execute(sql)
        CONN.sql("USE local")
        return res
    except Exception as e:
        # print(sql)
        raise e


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
    executor(sql_create_table_sync())
    executor(sql_create_table_artifacts())
    executor(sql_create_table_artifact_chunks())

    for type_def in TYPE_DEFS:
        executor(sql_create_table_model_parameter(type_def))
        executor(sql_create_table_checkpoint_sample_metric(type_def))
        executor(sql_create_table_train_step_metric(type_def))


def ensure_duck(run_run: Optional[TrainRun], in_memory=False):
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
    insert_model_parameter(
        train_epoch_state.model_id,
        train_run.run_id,
        "val_dataset_len",
        len(train_epoch_state.val_dataloader.dataset),
    )
    LAST_MODEL_ID = train_epoch_state.model_id
