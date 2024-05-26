import psycopg
import pandas
import json
from threading import Thread
import os
import queue
from typing import Union
from pathlib import Path
from itertools import islice

from lib.train_dataclasses import TrainEpochState
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import EnsembleConfig

from lib.compute_env import env


# TODO explain analyze, psql \x, web explain analyze, CTE,
# TODO one table per value type


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def dict_to_normalized_json(input_dict):
    return json.loads(pandas.json_normalize(input_dict).to_json(orient="records"))[0]


def setup_psql():
    hostname = env().postgres_host  # "localhost"
    port = env().postgres_port  # 5432
    pw = env().postgres_password
    # if "EP_POSTGRES" in os.environ:
    #     hostname = os.environ.get("EP_POSTGRES")
    #     # print(f"PSQL: {hostname}")
    # if "EP_POSTGRES_PORT" in os.environ:
    #     try:
    #         port = int(os.environ.get("EP_POSTGRES_PORT"))
    #     except ValueError:
    #         pass
    print(f"[db] Connection to {hostname}:{port}")
    with psycopg.connect(
        f"dbname=postgres user=postgres password={pw}",
        host=hostname,
        port=port,
        autocommit=True,
    ) as conn:
        try:
            conn.execute(
                """
                        CREATE DATABASE equiv;
                    """
            )
        except psycopg.errors.DuplicateDatabase:
            pass

    with psycopg.connect(
        f"dbname=equiv user=postgres password={pw}",
        host=hostname,
        port=port,
        autocommit=True,
    ) as conn:
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS metrics (
                    id serial PRIMARY KEY,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
                    train_id text,
                    ensemble_id text,
                    x float,
                    xaxis text,
                    variable text,
                    value float,
                    value_text text,
                    UNIQUE(train_id, x, xaxis, variable)
                )
                """
        )
        conn.execute(
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS metrics_batch_order AS
            SELECT metrics.id,
               metrics.created_at,
               metrics.train_id,
               metrics.ensemble_id,
               metrics.x,
               metrics.xaxis,
               metrics.variable,
               metrics.value,
               metrics.value_text
              FROM metrics
             WHERE (metrics.xaxis = 'batch'::text)
             ORDER BY metrics.train_id, metrics.variable, metrics.xaxis, metrics.x;
            """
        )
        conn.execute(
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS metrics_batch_order_10 AS
            SELECT metrics.id, 
               metrics.created_at, 
               metrics.train_id, 
               metrics.ensemble_id, 
               metrics.x, 
               metrics.xaxis, 
               metrics.variable, 
               metrics.value, 
               metrics.value_text
              FROM metrics
             WHERE ((metrics.xaxis = 'batch'::text) AND (((metrics.x)::integer % 10) = 0))
             ORDER BY metrics.train_id, metrics.variable, metrics.xaxis, metrics.x;
             """
        )
        conn.execute(
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS metrics_batch_order_100 AS
            SELECT metrics.id, 
               metrics.created_at, 
               metrics.train_id, 
               metrics.ensemble_id, 
               metrics.x, 
               metrics.xaxis, 
               metrics.variable, 
               metrics.value, 
               metrics.value_text
              FROM metrics
             WHERE ((metrics.xaxis = 'batch'::text) AND (((metrics.x)::integer % 100) = 0))
             ORDER BY metrics.train_id, metrics.variable, metrics.xaxis, metrics.x;
             """
        )
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS runs (
                    id serial PRIMARY KEY,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
                    train_id text,
                    ensemble_id text,
                    variable text,
                    value_float float,
                    value_int integer,
                    value_text text,
                    UNIQUE(train_id, variable)
                )
                """
        )
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id serial PRIMARY KEY,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
                    train_id text,
                    ensemble_id text,
                    name text,
                    path text,
                    size int,
                    UNIQUE(train_id, name)
                )
                """
        )
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS artifact_chunks (
                    id serial PRIMARY KEY,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
                    artifact_id int references artifacts(id),
                    seq_num int,
                    data bytea,
                    size int
                )
                """
        )


def insert_param(conn, train_id, ensemble_id, variable, value):
    value_type = type(value)
    value_dispatch = {
        str: lambda v: dict(value_text=v, value_float=None, value_int=None),
        float: lambda v: dict(value_text=str(v), value_float=v, value_int=None),
        int: lambda v: dict(value_text=str(v), value_float=None, value_int=v),
        list: lambda v: dict(value_text=str(v), value_float=None, value_int=None),
        bool: lambda v: dict(value_text=str(v), value_float=None, value_int=None),
        type(None): lambda v: dict(value_text="None", value_float=None, value_int=None),
    }
    value_dict = dict(
        train_id=train_id,
        ensemble_id=ensemble_id,
        variable=variable,
    )
    if value_type not in value_dispatch:
        value_type = str
        value = str(value)
    value_dict.update(value_dispatch[value_type](value))
    conn.execute(
        """
        INSERT INTO runs (train_id, ensemble_id, variable, value_text, value_int, value_float)
        VALUES (%(train_id)s, %(ensemble_id)s, %(variable)s, %(value_text)s, %(value_int)s, %(value_float)s)
        ON CONFLICT (train_id, variable) DO UPDATE SET value_text=EXCLUDED.value_text, value_int=EXCLUDED.value_int, value_float=EXCLUDED.value_float
        """,
        value_dict,
    )


def insert_or_update_train_run(conn, train_run: TrainRun):
    train_run_flat = dict_to_normalized_json(train_run.serialize_human())
    for key, value in train_run_flat.items():
        insert_param(
            conn, train_run_flat["train_id"], train_run_flat["ensemble_id"], key, value
        )


_threads = []
_result_queue = queue.Queue()


def render_psql(train_run: TrainRun, train_epoch_state: TrainEpochState, block=False):
    # _render_psql(train_run, train_epoch_state)
    # return
    if len(_threads) > 0:
        thread = _threads.pop(0)
        train_epoch_state.timing_metric.start("psql_thread_join")
        thread.join()
        train_epoch_state.timing_metric.stop("psql_thread_join")

    thread = Thread(
        target=_render_psql,
        kwargs=dict(
            train_run=train_run,
            train_epoch_state=train_epoch_state,
            result_queue=_result_queue,
        ),
    )
    thread.start()
    _threads.append(thread)
    if block:
        thread.join()
    try:
        return _result_queue.get(block=False)
    except queue.Empty:
        return None


def get_url():
    # hostname = os.getenv("EP_POSTGRES", env().postgres_host)
    # port = int(os.getenv("EP_POSTGRES_PORT", env().postgres_port))
    hostname = env().postgres_host
    port = env().postgres_port
    pw = env().postgres_password
    return f"postgresql://postgres:{pw}@{hostname}:{port}/equiv"


def _render_psql(
    train_run: TrainRun, train_epoch_state: TrainEpochState, result_queue: queue.Queue
):
    try:
        result_queue.put(_render_psql_unchecked(train_run, train_epoch_state))
    except psycopg.errors.OperationalError:
        result_queue.put(False)


def _render_psql_unchecked(train_run: TrainRun, train_epoch_state: TrainEpochState):
    train_run_dict = train_run.serialize_human()

    try:
        setup_psql()
    except psycopg.errors.OperationalError as e:
        # print("Couldn't setup PSQL datasink.")
        # print(str(e))
        return (False, str(e))

    train_epoch_state.timing_metric.start("psql_connection")
    with psycopg.connect(
        f"dbname=equiv user=postgres password={env().postgres_password}",
        host=env().postgres_host,
        port=int(env().postgres_port),
        autocommit=False,
        prepare_threshold=None,
    ) as conn:
        train_epoch_state.timing_metric.stop("psql_connection")
        train_epoch_state.timing_metric.start("psql_queries")

        if train_epoch_state.psql_starting_xs is None:
            train_epoch_state.psql_starting_xs = dict()
            row_cur = conn.execute(
                """
                           SELECT
                                variable, x
                            FROM
                                metrics
                            WHERE train_id=%s AND
                                (variable, x) IN (
                                    SELECT
                                        variable,
                                        MAX(x)
                                    FROM
                                        metrics
                                    WHERE
                                        train_id=%s  
                                    GROUP BY
                                        variable
                                );  
                         """,
                (train_run_dict["train_id"], train_run_dict["train_id"]),
            )
            rows = row_cur.fetchall()
            for row in rows:
                variable, max_x = row
                train_epoch_state.psql_starting_xs[variable] = max_x
                print(variable, max_x)

        def is_commited(variable, x):
            if variable in train_epoch_state.psql_starting_xs:
                if x <= train_epoch_state.psql_starting_xs[variable]:
                    # print(variable, x, "commited")
                    return True
                # else:
                # print(variable, x, "value x not commited")
            # else:
            #     print(variable, x, "variable not in dict")
            return False

        # create_param_view(conn, train_run)

        # start_time = time.time()
        train_epoch_state.timing_metric.start("psql_queries_params")
        insert_or_update_train_run(conn, train_run)
        train_epoch_state.timing_metric.stop("psql_queries_params")
        train_epoch_state.timing_metric.start("psql_queries_epoch")
        with conn.pipeline():
            for epoch in range(train_epoch_state.epoch):
                for metric in train_epoch_state.train_metrics:
                    if is_commited(metric.name(), epoch):
                        continue
                    conn.execute(
                        """
                        INSERT INTO metrics (train_id, x, xaxis, variable, value)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (train_id, x, xaxis, variable) DO UPDATE SET value = EXCLUDED.value
                    """,
                        (
                            train_run_dict["train_id"],
                            epoch,
                            "epoch",
                            metric.name(),
                            metric.mean(epoch),
                        ),
                    )
                for metric in train_epoch_state.validation_metrics:
                    metric_name = f"val_{metric.name()}"
                    if is_commited(metric_name, epoch):
                        continue
                    if metric.mean(epoch) is not None:
                        conn.execute(
                            """
                            INSERT INTO metrics (train_id, x, xaxis, variable, value)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (train_id, x, xaxis, variable) DO UPDATE SET value = EXCLUDED.value
                        """,
                            (
                                train_run_dict["train_id"],
                                epoch,
                                "epoch",
                                metric_name,
                                metric.mean(epoch),
                            ),
                        )
        train_epoch_state.timing_metric.stop("psql_queries_epoch")
        train_epoch_state.timing_metric.start("psql_queries_batch")
        with conn.pipeline(), conn.cursor() as cur:
            for metric in train_epoch_state.train_metrics:
                metric_name = f"{metric.name()}_batch"
                for idx, value in enumerate(metric.mean_batches()):
                    if is_commited(metric_name, idx):
                        continue
                    group_data = (
                        train_run_dict["train_id"],
                        idx,
                        "batch",
                        metric_name,
                        value,
                    )
                    query_hash = hash(group_data)
                    if query_hash not in train_epoch_state.psql_query_cache:
                        cur.execute(
                            """
                            INSERT INTO metrics (train_id, x, xaxis, variable, value)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (train_id, x, xaxis, variable) DO UPDATE SET value = EXCLUDED.value
                        """,
                            group_data,
                        )
                        train_epoch_state.psql_query_cache.add(query_hash)

        train_epoch_state.timing_metric.stop("psql_queries_batch")
        train_epoch_state.timing_metric.start("psql_queries_timing")
        with conn.pipeline(), conn.cursor() as cur:
            train_epoch_state.timing_metric.reset_group("timing_hash")
            train_epoch_state.timing_metric.reset_group("timing_query")
            for (
                timing_name,
                timing_data,
            ) in train_epoch_state.timing_metric.data.items():
                metric_name = f"timing_{timing_name}"
                for idx, value in enumerate(timing_data):
                    if is_commited(metric_name, idx):
                        continue
                    group_data = (
                        train_run_dict["train_id"],
                        idx,
                        "batch",
                        metric_name,
                        value,
                    )
                    train_epoch_state.timing_metric.start("timing_hash")
                    query_hash = hash(group_data)
                    if query_hash not in train_epoch_state.psql_query_cache:
                        train_epoch_state.timing_metric.stop_subgroup("timing_hash")
                        train_epoch_state.timing_metric.start("timing_query")
                        cur.execute(
                            """
                            INSERT INTO metrics (train_id, x, xaxis, variable, value)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (train_id, x, xaxis, variable) DO UPDATE SET value = EXCLUDED.value
                        """,
                            group_data,
                        )
                        train_epoch_state.timing_metric.stop_subgroup("timing_query")
                        train_epoch_state.timing_metric.start("timing_hash")
                        train_epoch_state.psql_query_cache.add(query_hash)
                        train_epoch_state.timing_metric.stop_subgroup("timing_hash")
                    else:
                        train_epoch_state.timing_metric.stop_subgroup("timing_hash")
        train_epoch_state.timing_metric.accumulate_group("timing_hash")
        train_epoch_state.timing_metric.accumulate_group("timing_query")

        train_epoch_state.timing_metric.stop("psql_queries_timing")
        train_epoch_state.timing_metric.stop("psql_queries")
        train_epoch_state.timing_metric.start("psql_commit")
        train_epoch_state.timing_metric.start("psql_commit_with_conn")
        conn.commit()
        train_epoch_state.timing_metric.stop("psql_commit")

    train_epoch_state.timing_metric.stop("psql_commit_with_conn")

    return (True, "")
    # print(f"Updated psql {time.time() - start_time}s")


def add_train_run(train_run):
    with psycopg.connect(
        f"dbname=equiv user=postgres password={env().postgres_password}",
        host=env().postgres_host,
        port=int(env().postgres_port),
        autocommit=False,
        prepare_threshold=None,
    ) as conn:
        insert_or_update_train_run(conn, train_run)
        conn.commit()


def add_parameter(train_run: TrainRun, name: str, value: str):
    # train_run_dict = train_run.serialize_human()
    try:
        setup_psql()
    except psycopg.errors.OperationalError as e:
        print("[Database] Could not connect to database, artifact not added.")
        print(f"{e}")
        return (False, str(e))

    train_run_dict = train_run.serialize_human()

    with psycopg.connect(
        f"dbname=equiv user=postgres password={env().postgres_password}",
        host=env().postgres_host,
        port=int(env().postgres_port),
        autocommit=False,
    ) as conn:
        insert_param(
            conn, train_run_dict["train_id"], train_run_dict["ensemble_id"], name, value
        )
    print(
        f"[Database] Added parameter {name}: {value} for {train_run_dict['train_id']}"
    )


def connect_psql():
    return psycopg.connect(
        f"dbname=equiv user=postgres password={env().postgres_password}",
        host=env().postgres_host,
        port=int(env().postgres_port),
        autocommit=False,
    )


def add_metric_epoch_values(conn, train_run, metric_name, value):
    train_id = train_run.serialize_human()["train_id"]

    conn.execute(
        """
        INSERT INTO metrics (train_id, x, xaxis, variable, value)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (train_id, x, xaxis, variable) DO UPDATE SET value = EXCLUDED.value
    """,
        (
            train_id,
            train_run.epochs,
            "epoch",
            metric_name,
            value,
        ),
    )


def _add_artifact(train_id: str, ensemble_id: str, name: str, path: Path):
    path = Path(path) if not isinstance(path, Path) else path
    size_bytes = path.stat().st_size
    value_dict = dict(
        train_id=train_id,
        ensemble_id=ensemble_id,
        name=name,
        path=path.absolute().as_posix(),
        size=size_bytes,
    )

    with psycopg.connect(
        f"dbname=equiv user=postgres password={env().postgres_password}",
        host=env().postgres_host,
        port=int(env().postgres_port),
        autocommit=False,
    ) as conn:
        cursor = conn.execute(
            """
            INSERT INTO artifacts (train_id, ensemble_id, name, path, size)
            VALUES (%(train_id)s, %(ensemble_id)s, %(name)s, %(path)s, %(size)s)
            ON CONFLICT (train_id, name) DO UPDATE SET path=EXCLUDED.path, size=EXCLUDED.size
            RETURNING id
            """,
            value_dict,
        )
        artifact_id = cursor.fetchone()[0]
        conn.execute(
            "DELETE FROM artifact_chunks WHERE artifact_id = %s", (artifact_id,)
        )
        print("[db] Uploading artifact")
        with path.open("rb") as file:
            seq_num = 0
            while chunk := file.read(1024 * 1024):
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO artifact_chunks (artifact_id, seq_num, data, size) VALUES (%s, %s, %s, %s)",
                        (artifact_id, seq_num, chunk, len(chunk)),
                    )
                seq_num += 1
                print(f"[db] Chunk {seq_num}")


def add_artifact(train_run: TrainRun, name: str, path: Union[str, Path]):
    # train_run_dict = train_run.serialize_human()
    try:
        setup_psql()
    except psycopg.errors.OperationalError as e:
        print("[Database] Could not connect to database, artifact not added.")
        print(f"{e}")
        return (False, str(e))

    train_run_dict = train_run.serialize_human()

    # if isinstance(path, Path):
    # path = path.relative_to(env().paths.artifacts)
    # path = path.as_posix()

    _add_artifact(
        train_id=train_run_dict["train_id"],
        ensemble_id=train_run_dict["ensemble_id"],
        name=name,
        path=path,
    )

    print(f"[Database] Added artifact {name}: {path}")


def has_artifact(train_run: TrainRun, name: str):
    # train_run_dict = train_run.serialize_human()
    try:
        setup_psql()
    except psycopg.errors.OperationalError as e:
        print("[Database] Could not connect to database, artifact not added.")
        return (False, str(e))

    train_run_dict = train_run.serialize_human()
    with psycopg.connect(
        f"dbname=equiv user=postgres password={env().postgres_password}",
        host=env().postgres_host,
        port=int(env().postgres_port),
        autocommit=False,
    ) as conn:
        rows = conn.execute(
            """
            SELECT * FROM artifacts WHERE train_id=%(train_id)s AND name=%(name)s
            """,
            dict(train_id=train_run_dict["train_id"], name=name),
        )
        res = rows.fetchone()
        return res is not None


def get_parameter(train_run: TrainRun, name: str):
    # train_run_dict = train_run.serialize_human()
    try:
        setup_psql()
    except psycopg.errors.OperationalError as e:
        print("[Database] Could not connect to database, artifact not added.")
        return (False, str(e))

    train_run_dict = train_run.serialize_human()
    with psycopg.connect(
        f"dbname=equiv user=postgres password={env().postgres_password}",
        host=env().postgres_host,
        port=int(env().postgres_port),
        autocommit=False,
    ) as conn:
        rows = conn.execute(
            """
            SELECT value_text FROM runs WHERE train_id=%(train_id)s AND variable=%(name)s
            """,
            dict(train_id=train_run_dict["train_id"], name=name),
        )
        res = rows.fetchone()
        if res is not None:
            return res[0]
        else:
            return None


def add_ensemble_artifact(
    ensemble_config: EnsembleConfig, name: str, path: Union[str, Path]
):
    add_artifact(ensemble_config.members[0], name, path)
