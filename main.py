import json

from sqlalchemy import create_engine

if __name__ == "__main__":
    from src.ui.app import main

    with open('config.json', 'r') as file:
        config_db = json.load(file)
    engine = create_engine(config_db['db_url_local'])

    # engine = create_engine("postgresql://postgres:dSJJNjkn42384*$(#@92.246.143.110:5432/windspectrum_db")

    main(engine)

# python -m nuitka --standalone --enable-plugin=pyside6 --assume-yes-for-downloads --windows-console-mode=disable --include-data-dir=db=db --include-data-dir=src/ui/resource=src/ui/resource --noinclude-unittest-mode=nofollow --output-dir=windspectrum --windows-icon-from-ico=src/ui/resource/images/mini_logo.ico --output-filename=windspectrum.exe --jobs=16 main.py
