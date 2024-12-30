from sqlalchemy import create_engine

if __name__ == "__main__":
    from src.ui.app import main

    engine = create_engine("postgresql://postgres:1234@localhost/postgres")
    # engine = create_engine("postgresql://postgres:dSJJNjkn42384*$(#@92.246.143.110:5432/windspectrum_db")

    main(engine)
