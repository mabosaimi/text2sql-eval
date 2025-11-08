"""BIRD mini-dev evaluation.
Adapted from https://github.com/bird-bench/mini_dev"""

import sqlite3
import os
import numpy as np
import time
import math


def nice_look_table(column_names: list, values: list):
    rows = []
    # Determine the maximum width of each column
    widths = [
        max(len(str(value[i])) for value in values + [column_names])
        for i in range(len(column_names))
    ]

    # Print the column names
    header = "".join(
        f"{column.rjust(width)} " for column, width in zip(column_names, widths)
    )
    # print(header)
    # Print the values
    for value in values:
        row = "".join(f"{str(v).rjust(width)} " for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + "\n" + rows
    return final_output


# TODO: support postgres and mysql
def generate_schema_prompt(sql_dialect="SQLite", db_path=None, num_rows=None):
    if sql_dialect == "SQLite":
        return generate_schema_prompt_sqlite(db_path, num_rows)
    # elif sql_dialect == "MySQL":
    #     return generate_schema_prompt_mysql(db_path)
    # elif sql_dialect == "PostgreSQL":
    #     return generate_schema_prompt_postgresql(db_path)
    else:
        raise ValueError("Unsupported SQL dialect: {}".format(sql_dialect))


def generate_schema_prompt_sqlite(db_path, num_rows=None):
    # extract create ddls
    """
    :param root_place:
    :param db_name:
    :return:
    """
    full_schema_prompt_list = []
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}
    for table in tables:
        if table == "sqlite_sequence":
            continue
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(
                table[0]
            )
        )
        create_prompt = cursor.fetchone()[0]
        schemas[table[0]] = create_prompt
        if num_rows:
            cur_table = table[0]
            if cur_table in ["order", "by", "group"]:
                cur_table = "`{}`".format(cur_table)

            cursor.execute("SELECT * FROM {} LIMIT {}".format(cur_table, num_rows))
            column_names = [description[0] for description in cursor.description]
            values = cursor.fetchall()
            rows_prompt = nice_look_table(column_names=column_names, values=values)
            verbose_prompt = "/* \n {} example rows: \n SELECT * FROM {} LIMIT {}; \n {} \n */".format(
                num_rows, cur_table, num_rows, rows_prompt
            )
            schemas[table[0]] = "{} \n {}".format(create_prompt, verbose_prompt)

    for k, v in schemas.items():
        full_schema_prompt_list.append(v)

    schema_prompt = "\n\n".join(full_schema_prompt_list)

    return schema_prompt


def generate_comment_prompt(question, sql_dialect, knowledge=None):
    base_prompt = f"-- Using valid {sql_dialect}"
    knowledge_text = " and understanding External Knowledge" if knowledge else ""
    knowledge_prompt = f"-- External Knowledge: {knowledge}" if knowledge else ""

    combined_prompt = (
        f"{base_prompt}{knowledge_text}, answer the following questions for the tables provided above.\n"
        f"-- {question}\n"
        f"{knowledge_prompt}"
    )
    return combined_prompt


def generate_cot_prompt(sql_dialect):
    return f"\nGenerate the {sql_dialect} for the above question after thinking step by step: "


def generate_instruction_prompt(sql_dialect):
    return f"""
        \nIn your response, you do not need to mention your intermediate steps. 
        Do not include any comments in your response.
        Do not need to start with the symbol ```
        You only need to return the result {sql_dialect} SQL code
        start from SELECT
        """


def generate_combined_prompts_one(db_path, question, sql_dialect, knowledge=None):
    schema_prompt = generate_schema_prompt(sql_dialect, db_path)
    comment_prompt = generate_comment_prompt(question, sql_dialect, knowledge)
    cot_prompt = generate_cot_prompt(sql_dialect)
    instruction_prompt = generate_instruction_prompt(sql_dialect)

    combined_prompts = "\n\n".join(
        [schema_prompt, comment_prompt, cot_prompt, instruction_prompt]
    )
    return combined_prompts


def doc_to_text(doc):
    """Generate prompt."""
    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data/minidev/MINIDEV/dev_databases",
        doc["db_id"],
        f"{doc['db_id']}.sqlite",
    )
    return generate_combined_prompts_one(
        db_path=db_path,
        question=doc["question"],
        sql_dialect="SQLite",
        knowledge=doc.get("evidence", ""),
    )


def doc_to_target(doc):
    """Extract gold SQL."""
    return doc["SQL"]


def process_docs(dataset):
    return dataset


# Metrics
def calculate_ex(predicted_res, ground_truth_res):
    return int(set(predicted_res) == set(ground_truth_res))


def calculate_row_match(predicted_row, ground_truth_row):
    total_columns = len(ground_truth_row)
    matches = 0
    element_in_pred_only = 0
    element_in_truth_only = 0
    for pred_val in predicted_row:
        if pred_val in ground_truth_row:
            matches += 1
        else:
            element_in_pred_only += 1
    for truth_val in ground_truth_row:
        if truth_val not in predicted_row:
            element_in_truth_only += 1
    match_percentage = matches / total_columns
    pred_only_percentage = element_in_pred_only / total_columns
    truth_only_percentage = element_in_truth_only / total_columns
    return match_percentage, pred_only_percentage, truth_only_percentage


def calculate_f1_score(predicted, ground_truth):
    if not predicted and not ground_truth:
        return 1.0

    predicted = list(dict.fromkeys(predicted))
    ground_truth = list(dict.fromkeys(ground_truth))

    match_scores = []
    pred_only_scores = []
    truth_only_scores = []
    for i, gt_row in enumerate(ground_truth):
        if i >= len(predicted):
            match_scores.append(0)
            truth_only_scores.append(1)
            continue
        pred_row = predicted[i]
        match_score, pred_only_score, truth_only_score = calculate_row_match(
            pred_row, gt_row
        )
        match_scores.append(match_score)
        pred_only_scores.append(pred_only_score)
        truth_only_scores.append(truth_only_score)

    for i in range(len(predicted) - len(ground_truth)):
        match_scores.append(0)
        pred_only_scores.append(1)
        truth_only_scores.append(0)

    tp = sum(match_scores)
    fp = sum(pred_only_scores)
    fn = sum(truth_only_scores)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    )
    return f1_score


def clean_abnormal(input_times):
    if not input_times:
        return []
    input_times = np.asarray(input_times)
    processed_list = []
    mean = np.mean(input_times, axis=0)
    std = np.std(input_times, axis=0)
    for x in input_times:
        if x < mean + 3 * std and x > mean - 3 * std:
            processed_list.append(x)
    return processed_list


def calculate_ves(predicted_sql, ground_truth_sql, db_path, iterate_num=10):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth_sql)
        ground_truth_res = cursor.fetchall()

        if set(predicted_res) != set(ground_truth_res):
            conn.close()
            return 0

        diff_list = []
        for _ in range(iterate_num):
            start = time.time()
            cursor.execute(predicted_sql)
            cursor.fetchall()
            predicted_time = time.time() - start

            start = time.time()
            cursor.execute(ground_truth_sql)
            cursor.fetchall()
            ground_truth_time = time.time() - start

            if predicted_time > 0:
                diff_list.append(ground_truth_time / predicted_time)

        conn.close()

        processed_diff_list = clean_abnormal(diff_list)
        if not processed_diff_list:
            return 0

        time_ratio = sum(processed_diff_list) / len(processed_diff_list)

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
    gold_sql = doc.get("SQL", "")
    db_id = doc.get("db_id")

    if not db_id:
        return {"ex": 0, "f1": 0.0, "ves": 0}

    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data/minidev/MINIDEV/dev_databases",
        db_id,
        f"{db_id}.sqlite",
    )

    ex = 0
    f1 = 0.0
    ves = 0

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(pred_sql.strip())
        pred_res = cursor.fetchall()
        cursor.execute(gold_sql.strip())
        gold_res = cursor.fetchall()
        conn.close()

        ex = calculate_ex(pred_res, gold_res)
        f1 = calculate_f1_score(pred_res, gold_res)

    except Exception:
        return {"ex": 0, "f1": 0.0, "ves": 0}

    if ex == 1:
        ves_score = calculate_ves(pred_sql.strip(), gold_sql.strip(), db_path)
        ves = math.sqrt(ves_score) * 100

    return {"ex": ex, "f1": f1, "ves": ves}
