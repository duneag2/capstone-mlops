# data_generator.py
import time
from argparse import ArgumentParser

import pandas as pd
import psycopg2

def get_data():
    file_path = 'monday.json'
    df = pd.read_json(file_path, orient='records', lines=True)
    return df

def create_table(db_connect):
    drop_table_query = "DROP TABLE IF EXISTS cargo;"
    create_table_query = """
    CREATE TABLE cargo (
        id SERIAL PRIMARY KEY,
        timestamp timestamp,
        image_path varchar(255),
        target int
    );
    """
    
    with db_connect.cursor() as cur:
        cur.execute(drop_table_query)
        cur.execute(create_table_query)
        db_connect.commit()

def insert_dataframe(db_connect, df):
    for index, row in df.iterrows():
        insert_row_query = f"""
        INSERT INTO cargo
            (timestamp, image_path, target)
            VALUES (
                NOW(),
                '{row['image_path']}',
                {row['target']}
            );
        """
        with db_connect.cursor() as cur:
            cur.execute(insert_row_query)
            db_connect.commit()

def insert_data(db_connect, data):
    insert_row_query = f"""
    INSERT INTO cargo
        (timestamp, image_path, target)
        VALUES (
            NOW(),
            '{data.image_path}',
            {data.target}
        );
    """
    print("run")
    with db_connect.cursor() as cur:
        cur.execute(insert_row_query)
        print(insert_row_query)
        db_connect.commit()


def generate_data(db_connect, df):
    insert_dataframe(db_connect, df)
    while True:
        insert_data(db_connect, df.sample(1).squeeze())
        time.sleep(1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--db-host", dest="db_host", type=str, default="localhost")
    args = parser.parse_args()

    db_connect = psycopg2.connect(
        user="myuser",
        password="mypassword",
        host=args.db_host,
        port=5432,
        database="mydatabase",
    )
    create_table(db_connect)
    df = get_data()
    generate_data(db_connect, df)