import json
import os.path

from sqlalchemy import create_engine

from db_schema.create_sqlite_db import load_dump

if __name__ == "__main__":
    from src.ui.app import main

    if not os.path.exists('windspectrum.db'):
        dump_file = os.path.join('db_schema', 'dump_schema.sql')
        db_file = 'windspectrum.db'  # Укажите имя выходного файла базы данных
        load_dump(dump_file, db_file)

    with open('config.json', 'r') as file:
        config_db = json.load(file)

    engine = create_engine(config_db['db_url'])

    {
        "db_url": "sqlite:///windspectrum.db",
        "db_url_server": "postgresql://postgres:dSJJNjkn42384*$(#@92.246.143.110:5432/windspectrum_db"
    }

    main(engine)
