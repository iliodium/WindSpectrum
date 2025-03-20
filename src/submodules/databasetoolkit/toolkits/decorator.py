from functools import wraps

from sqlalchemy import create_engine


def with_db_connection(db_url):
    """
    Декоратор для создания и закрытия SQLAlchemy Engine.
    :param db_url: Строка подключения к базе данных (например, "sqlite:///windspectrum.db").
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Создаем engine
            engine = create_engine(db_url)
            try:
                # Вызываем функцию, передавая engine как аргумент
                result = func(engine, *args, **kwargs)
            finally:
                # Закрываем engine после выполнения функции
                engine.dispose()
            return result

        return wrapper

    return decorator
