# Publishing Checklist

## 1. Set metadata

Update `pyproject.toml`:
- `project.version`
- `project.authors`
- `project.urls`

## 2. Validate package locally

```bash
python -m pip install -U build twine
python -m build
python -m twine check dist/*
```

## 3. Optional: test install in clean env

```bash
python -m venv .venv-publish
source .venv-publish/bin/activate
python -m pip install --upgrade pip
python -m pip install dist/*.whl
python -c "import flash_mhc; print(flash_mhc.__all__)"
```

## 4. Publish to TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
```

## 5. Publish to PyPI

```bash
python -m twine upload dist/*
```

## Notes

- GPU kernels require compatible CUDA + Triton runtime.
- Keep benchmark and tuning docs up to date when changing launch configs.
