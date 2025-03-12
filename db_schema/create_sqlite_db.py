import sqlite3


def load_dump(dump_file, db_file):
    # Подключаемся к базе данных (если файла нет, он будет создан)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Читаем файл дампа
    with open(dump_file, 'r', encoding='utf-8') as f:
        sql_dump = f.read()

    # Выполняем SQL-запросы из дампа
    cursor.executescript(sql_dump)

    # Сохраняем изменения и закрываем соединение
    conn.commit()
    conn.close()


if __name__ == "__main__":
    dump_file = r'dump_schema.sql'  # Укажите путь к вашему файлу дампа
    db_file = r'../windspectrum.db'  # Укажите имя выходного файла базы данных
    load_dump(dump_file, db_file)
