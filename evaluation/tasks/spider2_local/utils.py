"""Spider2-lite local evaluation. 
Comparison logic from https://github.com/xlang-ai/Spider2"""

import os
import json
import sqlite3
import glob
import pandas as pd
import math


def load_tables_json():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    db_path = os.path.join(data_dir, "databases")
    tables_json = []

    for db_folder in os.listdir(db_path):
        db_folder_path = os.path.join(db_path, db_folder)
        if not os.path.isdir(db_folder_path):
            continue

        db_id = db_folder
        table_names_original = []
        column_names_original = []
        column_types = []
        column_descriptions = []
        sample_rows = {}

        ddl_path = os.path.join(db_folder_path, "DDL.csv")
        if os.path.exists(ddl_path):
            import csv

            with open(ddl_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (
                        row.get("table_name")
                        and row["table_name"] not in table_names_original
                    ):
                        table_names_original.append(row["table_name"])

        table_index = 0
        for json_file in sorted(glob.glob(os.path.join(db_folder_path, "*.json"))):
            with open(json_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    table_name = data.get(
                        "table_name", os.path.basename(json_file).replace(".json", "")
                    )

                    if table_name not in table_names_original:
                        table_names_original.append(table_name)
                        table_index = len(table_names_original) - 1
                    else:
                        table_index = table_names_original.index(table_name)

                    columns = data.get("column_names", [])
                    for col_name in columns:
                        column_names_original.append([table_index, col_name])

                    col_types = data.get("column_types", [])
                    column_types.extend(col_types)

                    descriptions = data.get("description", [])
                    for desc in descriptions:
                        column_descriptions.append([table_index, desc])

                    if "sample_rows" in data:
                        sample_rows[table_name] = data["sample_rows"]

                except json.JSONDecodeError:
                    continue

        tables_json.append(
            {
                "db_id": db_id,
                "table_names_original": table_names_original,
                "column_names_original": column_names_original,
                "column_types": column_types,
                "column_descriptions": column_descriptions,
                "sample_rows": sample_rows,
            }
        )

    return tables_json


def get_sql_for_database_from_tables_json(db_id, tables_json, use_column_desc=True):
    for db in tables_json:
        if db["db_id"] != db_id:
            continue

        table_names = db["table_names_original"]
        columns = db["column_names_original"]
        column_types = db.get("column_types", [])
        column_descs = db.get("column_descriptions", [])

        create_statements = []
        for table_index, table_name in enumerate(table_names):
            table_columns = []
            for col_idx, col in enumerate(columns):
                if col[0] == table_index:
                    col_name = col[1]
                    col_type = (
                        column_types[col_idx] if col_idx < len(column_types) else "TEXT"
                    )
                    col_desc = ""
                    if use_column_desc and col_idx < len(column_descs):
                        if column_descs[col_idx][0] == table_index:
                            col_desc = column_descs[col_idx][1]
                    table_columns.append((col_name, col_type, col_desc))

            column_defs = []
            for col_name, col_type, col_desc in table_columns:
                column_def = f'"{col_name}" {col_type.upper()}'
                if use_column_desc and col_desc:
                    column_def += f' COMMENT "{col_desc}"'
                column_defs.append(column_def)

            create_statement = f'CREATE TABLE "{table_name}"\n(\n    '
            create_statement += ",\n    ".join(column_defs)
            create_statement += "\n)"
            create_statements.append(create_statement)

        return create_statements

    raise ValueError(f"Database '{db_id}' not found in tables_json")


def get_sample_rows_for_database_from_tables_json(db_id, tables_json):
    for db in tables_json:
        if db["db_id"] != db_id:
            continue

        sample_rows = db.get("sample_rows", {})
        markdown_string = ""

        for table_name, data in sample_rows.items():
            if data:
                try:
                    import pandas as pd

                    df = pd.DataFrame(data)
                    df = df.head(1)
                    markdown_table = df.to_markdown(index=False)
                    markdown_string += f"table {table_name}:\n{markdown_table}\n\n"
                except:
                    pass

        return markdown_string

    return ""


def doc_to_text(doc):
    question = doc["question"]
    db_id = doc["db"]
    external_knowledge = doc.get("external_knowledge")

    db_path = os.path.join(os.path.dirname(__file__), "data", f"{db_id}.sqlite")
    sqls = [_get_schema_from_sqlite(db_path)]

    template_info = "/* Given the following database schema: */\n{}"
    prompt_info = template_info.format("\n\n".join(sqls))

    prompt_components = [prompt_info]

    if external_knowledge:
        doc_path = os.path.join(
            os.path.dirname(__file__), "data/documents", external_knowledge
        )
        if os.path.exists(doc_path):
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()
            external_knowledge_info = "/* External knowledge: */\n{}"
            prompt_components.append(external_knowledge_info.format(content))

    template_question = "/* Generate a {} SQL statement to answer the following question, ensuring that the syntax and functions are appropriate for {}. No explanation is required and don't use ``````: {} */"
    prompt_question = template_question.format("SQLite", "SQLite", question)

    prompt_components.append(prompt_question)

    prompt = "\n\n".join(prompt_components)
    return prompt


def _get_schema_from_sqlite(db_path):
    if not os.path.exists(db_path):
        return "/* Database file not found */"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    schemas = []
    for table in tables:
        if table[0] == "sqlite_sequence":
            continue
        cursor.execute(
            f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table[0]}'"
        )
        create_sql = cursor.fetchone()[0]
        schemas.append(create_sql)

    conn.close()
    return "\n\n".join(schemas)


def process_docs(dataset):
    def _process_doc(doc):
        if doc.get("instance_id", "").startswith("local"):
            return doc
        return None

    return dataset.filter(lambda x: x.get("instance_id", "").startswith("local"))


def doc_to_target(doc):
    instance_id = doc.get("instance_id", "")
    gold_sql_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data/gold/sql",
        f"{instance_id}.sql",
    )

    if os.path.exists(gold_sql_path):
        with open(gold_sql_path, "r") as f:
            return f.read().strip()
    return ""


def calculate_ex(predicted_res, ground_truth_res):
    return int(set(map(str, predicted_res)) == set(map(str, ground_truth_res)))


def calculate_ves(predicted_sql, ground_truth_sql, db_path, iterate_num=10):
    import time
    import numpy as np

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth_sql)
        ground_truth_res = cursor.fetchall()

        if set(map(str, predicted_res)) != set(map(str, ground_truth_res)):
            conn.close()
            return 0

        diff_list = []
        for _ in range(iterate_num):
            start_time = time.time()
            cursor.execute(predicted_sql)
            cursor.fetchall()
            pred_time = time.time() - start_time

            start_time = time.time()
            cursor.execute(ground_truth_sql)
            cursor.fetchall()
            truth_time = time.time() - start_time

            if pred_time > 0:
                diff_list.append(truth_time / pred_time)

        conn.close()

        if diff_list:
            diff_list = np.array(diff_list)
            mean = np.mean(diff_list)
            std = np.std(diff_list)
            diff_list = [x for x in diff_list if mean - 3 * std < x < mean + 3 * std]

        if not diff_list:
            return 0

        time_ratio = np.mean(diff_list)

        if time_ratio >= 2:
            return 1.25
        elif time_ratio >= 1:
            return 1
        elif time_ratio >= 0.5:
            return 0.75
        elif time_ratio >= 0.25:
            return 0.5
        else:
            return 0.25

    except Exception:
        return 0


def process_results(doc, results):
    pred_sql = results[0] if results else ""
    instance_id = doc.get("instance_id")
    db_id = doc.get("db")

    if not instance_id or not db_id:
        return {"ex": 0, "valid": 0}

    gold_sql_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data/gold/sql",
        f"{instance_id}.sql",
    )

    if not os.path.exists(gold_sql_path):
        return {"ex": 0, "valid": 0}

    with open(gold_sql_path, "r") as f:
        gold_sql = f.read().strip()

    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", f"{db_id}.sqlite"
    )

    if not os.path.exists(db_path):
        return {"ex": 0, "valid": 0}

    valid = 0
    ex = 0

    try:
        conn = sqlite3.connect(db_path)

        pred_df = pd.read_sql_query(pred_sql.strip(), conn)
        valid = 1

        gold_df = pd.read_sql_query(gold_sql.strip(), conn)

        ex = compare_pandas_table(
            pred_df, gold_df, condition_cols=[], ignore_order=False
        )

        conn.close()
    except Exception as e:
        valid = 0
        ex = 0

    return {"ex": ex, "valid": valid}


def compare_multi_pandas_table(
    pred, multi_gold, multi_condition_cols=[], multi_ignore_order=False
):
    print("multi_condition_cols", multi_condition_cols)

    if (
        multi_condition_cols == []
        or multi_condition_cols == [[]]
        or multi_condition_cols == [None]
        or multi_condition_cols == None
    ):
        multi_condition_cols = [[] for _ in range(len(multi_gold))]
    elif len(multi_gold) > 1 and not all(
        isinstance(sublist, list) for sublist in multi_condition_cols
    ):
        multi_condition_cols = [multi_condition_cols for _ in range(len(multi_gold))]
    multi_ignore_order = [multi_ignore_order for _ in range(len(multi_gold))]

    for i, gold in enumerate(multi_gold):
        if compare_pandas_table(
            pred, gold, multi_condition_cols[i], multi_ignore_order[i]
        ):
            return 1
    return 0


def compare_pandas_table(pred, gold, condition_cols=[], ignore_order=False):
    """_summary_

    Args:
        pred (Dataframe): _description_
        gold (Dataframe): _description_
        condition_cols (list, optional): _description_. Defaults to [].
        ignore_order (bool, optional): _description_. Defaults to False.

    """
    print("condition_cols", condition_cols)

    tolerance = 1e-2

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        if ignore_order_:
            v1, v2 = (
                sorted(
                    v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))
                ),
                sorted(
                    v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))
                ),
            )
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True

    if condition_cols != []:
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold
    pred_cols = pred

    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred_cols.transpose().values.tolist()
    score = 1
    for _, gold in enumerate(t_gold_list):
        if not any(
            vectors_match(gold, pred, ignore_order_=ignore_order)
            for pred in t_pred_list
        ):
            score = 0
        else:
            for j, pred in enumerate(t_pred_list):
                if vectors_match(gold, pred, ignore_order_=ignore_order):
                    break

    return score
