import duckdb 
import seaborn as sns
import pandas as pd

db = duckdb.connect()
db.sql(open('secrets.sql').read())


def sql_escape(value: str) -> str:
    return value.replace("'", "''")


def build_typed_query(filter_params: dict, limit: int | None = None) -> str:
    text_filters = {}
    int_filters = {}

    for key, value in filter_params.items():
        if isinstance(value, bool):
            # optional: treat bools as text if your DB stores them as "True"/"False"
            text_filters[key] = str(value)
        elif isinstance(value, int):
            int_filters[key] = value
        else:
            text_filters[key] = str(value)

    ctes = []
    final_select = ""

    if text_filters:
        text_conditions = []
        for key, value in text_filters.items():
            key_esc = sql_escape(str(key))
            value_esc = sql_escape(str(value))
            text_conditions.append(f"(name = '{key_esc}' AND value = '{value_esc}')")

        text_where = " OR ".join(text_conditions)

        text_cte = f"""
        text_matches AS (
            SELECT model_id, run_id
            FROM eqp.model_parameter_text
            WHERE {text_where}
            GROUP BY model_id, run_id
            HAVING COUNT(DISTINCT name) = {len(text_filters)}
        )
        """
        ctes.append(text_cte)

    if int_filters:
        int_conditions = []
        for key, value in int_filters.items():
            key_esc = sql_escape(str(key))
            int_conditions.append(f"(name = '{key_esc}' AND value = {value})")

        int_where = " OR ".join(int_conditions)

        int_cte = f"""
        int_matches AS (
            SELECT model_id, run_id
            FROM eqp.model_parameter_int
            WHERE {int_where}
            GROUP BY model_id, run_id
            HAVING COUNT(DISTINCT name) = {len(int_filters)}
        )
        """
        ctes.append(int_cte)

    if text_filters and int_filters:
        final_select = """
        SELECT t.model_id, t.run_id
        FROM text_matches t
        INNER JOIN int_matches i
            ON t.model_id = i.model_id
           AND t.run_id = i.run_id
        """
    elif text_filters:
        final_select = """
        SELECT model_id, run_id
        FROM text_matches
        """
    elif int_filters:
        final_select = """
        SELECT model_id, run_id
        FROM int_matches
        """
    else:
        raise ValueError("filter_params cannot be empty")

    query = "WITH\n" + ",\n".join(ctes) + "\n" + final_select

    if limit is not None:
        query += f"\nLIMIT {limit}"

    return query

def get_model_min_loss(filter_params: dict) -> pd.DataFrame:
    """
    Returns the model_id related to the model with the highest value in 
    the column steps among the models matching the filter parameters.
    """

    query = build_typed_query(filter_params)
    df_filtered = db.sql(query).df()

    if df_filtered.empty:
        raise ValueError("No models found matching the filter parameters.")

    model_ids = df_filtered['model_id'].unique()
    model_ids_str = ", ".join(str(mid) for mid in model_ids)

    max_steps_query = f"""
    SELECT model_id, MIN(value) AS min_loss
    FROM eqp.train_step_metric_float
    WHERE model_id IN ({model_ids_str}) AND name = 'reg_loss'
    GROUP BY model_id
    """
    df_max_steps = db.sql(max_steps_query).df()

    return df_max_steps


if __name__ == '__main__':
    
    filter_params_pear = {
        'train_config.data.config.start_year': '2007',
        'train_config.data.config.end_year': '2017',
        'train_config.model.name': 'SwinHPPanguPad',
        
    }

    filter_params_conv = {
        'train_config.data.config.start_year': '2007',
        'train_config.data.config.end_year': '2017',
        'train_config.model.name': 'HEALPixPearConv',
    }

    pear_min_losses_df = get_model_min_loss(filter_params_pear)
    conv_min_losses_df = get_model_min_loss(filter_params_conv)



