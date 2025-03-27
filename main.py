import json
import os.path

from db_schema.create_sqlite_db import load_dump

if __name__ == "__main__":
    from src.ui.app import main

    if os.path.exists('windspectrum.db'):
        os.remove('windspectrum.db')

    if not os.path.exists('windspectrum.db'):
        dump_file = os.path.join('db_schema', 'dump_schema.sql')
        db_file = 'windspectrum.db'  # Укажите имя выходного файла базы данных
        load_dump(dump_file, db_file)

    with open('config.json', 'r') as file:
        config = json.load(file)

    {
        "db_url": "sqlite:///windspectrum.db",
        "db_url_server": "postgresql://postgres:dSJJNjkn42384*$(#@92.246.143.110:5432/windspectrum_db1",
        "DB_URL_SERVER": "postgresql://postgres:1234@localhost/postgres",

    }

    main(config)
