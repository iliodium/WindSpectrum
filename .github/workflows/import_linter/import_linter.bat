echo off
set folder=%1
set command-setting-path=%2
set setting-path=%3

echo Script is running in directory: %cd%

isort src/ --settings-path .github/workflows/import_linter/.isort.cfg