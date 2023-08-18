import psycopg
import pandas
import json
import time
from threading import Thread

from lib.train_dataclasses import TrainEpochState
from lib.train_dataclasses import TrainRun


# TODO explain analyze, psql \x, web explain analyze, CTE,
# TODO one table per value type


def dict_to_normalized_json(input_dict):
    return json.loads(pandas.json_normalize(input_dict).to_json(orient="records"))[0]


def setup_psql():
    with psycopg.connect(
        "dbname=postgres user=postgres password=postgres",
        host="localhost",
        port=5432,
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
        "dbname=equiv user=postgres password=postgres",
        host="localhost",
        port=5432,
        autocommit=True,
    ) as conn:
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS metrics (
                    id serial PRIMARY KEY,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
                    train_id text,
                    ensemble_id text,
                    epoch int,
                    variable text,
                    value float,
                    value_text text,
                    UNIQUE(train_id, epoch, variable)
                )
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


def drop_views(conn):
    conn.execute("DROP VIEW IF EXISTS metrics_and_runs;")
    conn.execute("DROP VIEW IF EXISTS metrics_view;")


def create_metrics_view(conn):
    variables = list(
        conn.execute("SELECT DISTINCT variable FROM metrics ORDER BY variable")
    )
    keys_and_type = [f'"{key}" float' for key, in variables]

    properties = ", ".join(keys_and_type)
    properties = (
        f"(train_id_and_epoch TEXT, train_id TEXT, epoch INTEGER, {properties})"
    )

    create_view_sql2 = f"""
       CREATE OR REPLACE VIEW metrics_view AS
        SELECT *
        FROM crosstab(
            'SELECT train_id || epoch, train_id, epoch, variable, value
             FROM metrics ORDER BY train_id || epoch, variable',
            'SELECT DISTINCT variable FROM metrics ORDER BY variable'
        ) AS ct{properties};
    """
    # print(create_view_sql2)
    conn.execute("CREATE EXTENSION IF NOT EXISTS tablefunc")
    try:
        conn.execute(create_view_sql2)
    except psycopg.errors.InvalidTableDefinition:
        conn.cancel()
        conn.execute("DROP VIEW metrics_view CASCADE")
        conn.execute(create_view_sql2)


def create_metrics_and_run_info_view(conn):
    sql = """
            CREATE OR REPLACE VIEW metrics_and_runs AS 
        SELECT * FROM metrics_view JOIN runs_view USING(train_id)
        """
    conn.execute(sql)


def create_param_view(conn, train_run: TrainRun):
    insert_or_update_train_run(conn, train_run)
    variables = list(
        conn.execute("SELECT DISTINCT variable FROM runs ORDER BY variable")
    )
    keys_and_type = [f'"{key}" TEXT' for key, in variables]

    properties = ", ".join(keys_and_type)
    properties = f"(id text, {properties})"

    create_view_sql2 = f"""
       CREATE OR REPLACE VIEW runs_view AS
        SELECT *
        FROM crosstab(
            'SELECT train_id, variable, value_text
             FROM runs ORDER BY train_id, variable',
            'SELECT DISTINCT variable FROM runs ORDER BY variable'
        ) AS ct{properties};
    """
    # print(create_view_sql2)
    conn.execute("CREATE EXTENSION IF NOT EXISTS tablefunc")
    try:
        conn.execute(create_view_sql2)
    except psycopg.errors.InvalidTableDefinition:
        conn.commit()
        conn.execute("DROP VIEW runs_view CASCADE")
        conn.execute(create_view_sql2)


def insert_or_update_train_run(conn, train_run: TrainRun):
    train_run_flat = dict_to_normalized_json(train_run.serialize_human())
    for key, value in train_run_flat.items():
        insert_param(
            conn, train_run_flat["train_id"], train_run_flat["ensemble_id"], key, value
        )


_threads = []


def render_psql(train_run: TrainRun, train_epoch_state: TrainEpochState):
    _render_psql(train_run, train_epoch_state)
    return
    if len(_threads) > 3:
        thread = _threads.pop(0)
        thread.join()

    thread = Thread(
        target=_render_psql,
        kwargs=dict(train_run=train_run, train_epoch_state=train_epoch_state),
    )
    thread.start()
    _threads.append(thread)


def _render_psql(train_run: TrainRun, train_epoch_state: TrainEpochState):
    train_run_dict = train_run.serialize_human()
    try:
        setup_psql()
    except psycopg.errors.OperationalError as e:
        print("Couldn't setup PSQL datasink.")
        print(str(e))
        return

    with psycopg.connect(
        "dbname=equiv user=postgres password=postgres",
        host="localhost",
        port=5432,
        autocommit=False,
    ) as conn:
        create_param_view(conn, train_run)

        start_time = time.time()
        insert_or_update_train_run(conn, train_run)
        for epoch in range(train_epoch_state.epoch):
            for metric in train_epoch_state.train_metrics:
                conn.execute(
                    """
                    INSERT INTO metrics (train_id, epoch, variable, value)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (train_id, epoch, variable) DO UPDATE SET value = EXCLUDED.value
                """,
                    (
                        train_run_dict["train_id"],
                        epoch,
                        metric.name(),
                        metric.mean(epoch),
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO metrics (train_id, epoch, variable, value)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (train_id, epoch, variable) DO UPDATE SET value = EXCLUDED.value
                """,
                    (
                        train_run_dict["train_id"],
                        epoch,
                        f"val_{metric.name()}",
                        metric.mean(epoch),
                    ),
                )
        conn.commit()
        drop_views(conn)
        create_metrics_view(conn)
        create_metrics_and_run_info_view(conn)
        conn.commit()
        print(f"Updated psql {time.time() - start_time}s")
