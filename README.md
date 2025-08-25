uv run python -m build

twine check dist/*
twine upload dist/*
