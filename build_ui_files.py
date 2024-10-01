import os
import pathlib

UIC_EXECUTABLE = list(pathlib.Path(__file__).parent.glob("**/uic.*"))[0]

current_path = pathlib.Path(__file__).parent
path_to_ui_files = (
    current_path
    .joinpath("src")
    .joinpath("ui")
    .joinpath("qt")
    .joinpath("views")
)
path_to_py_files = (
    current_path
    .joinpath("src")
    .joinpath("ui")
    .joinpath("qt")
    .joinpath("classes")
)

data = list(path_to_ui_files.glob("**/*.ui"))

for ui_file in data:
    ui_filename = ui_file.name
    abs_path = str(ui_file)
    sub_path = abs_path.replace(str(path_to_ui_files), "")
    target = abs_path.replace(str(path_to_ui_files), str(path_to_py_files))
    target = target.replace(sub_path, sub_path.replace(".ui", ".py"))
    if not pathlib.Path(target).exists() or not pathlib.Path(target).is_dir():
        pathlib.Path(target).parent.mkdir(parents=True, exist_ok=True)
    command = f"{UIC_EXECUTABLE} -g python {ui_file} -o {target}"
    os.system(command)
