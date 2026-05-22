# Project rules — shp2xodr-converter

NGII (Korean precise road map) SHP → OpenDRIVE converter.

## Toolchain

- **Python**: 3.13 for current dev.
- **Env / deps**: `uv` exclusively. Add deps with `uv add <pkg>` (runtime) or
  `uv add --dev <pkg>` (dev). Sync with `uv sync`.
- **Lint / format**: `ruff check` and `ruff format`.
- **Types**: `mypy` with strict mode. New code must type-check clean.
- **Tests**: `pytest`, kept light. Tests lock down tricky invariants — not
  every line.

## Coding rules

- **`pathlib.Path`, never `os.path`.**
- **`@dataclass(slots=True)`** for record-like types. Use `frozen=True` when
  the instance is immutable after construction. Don't reach for `TypedDict`
  or raw dicts when a dataclass fits.
- **Avoid `try/except`.** Let exceptions propagate. Validate at the I/O
  boundary and trust types everywhere else. Allowed exception: catching a
  *specific* exception at a *real* boundary (file not found, network error)
  where there is a meaningful recovery action.
- **Configuration via Hydra.** No magic constants in modules — they live in
  `conf/`. The entry point uses `@hydra.main`.
- **Source layout**: `src/shp2xodr/`. Importable as `shp2xodr.*`.
- **`logging`, never `print`.** Hydra wires up a logger automatically. Use
  `logging.getLogger(__name__)`; let verbosity be controlled from config.
- **Point sequences are `numpy.ndarray` of shape `(N, 2)` or `(N, 3)`**, not
  `list[tuple[float, float]]`. Shapely 2.x and numpy are vectorised; geometry
  math should be written that way too.
- **Unit suffixes on variables where the unit matters.** `width_m`,
  `length_m`, `hdg_rad`, `curvature_per_m`. OpenDRIVE mixes radians/degrees
  and meters/millimeters across fields — ambiguity here is a real bug source.
- **No `assert` in `src/`; tests can use it freely.** Production code uses
  `raise` for every runtime check, including type narrowing (`if x is None:
  raise RuntimeError(...)`) and post-conditions you actually want enforced.
  Asserts get stripped with `python -O`, so any check that should hold at
  runtime must be a `raise`. `assert` is the right tool in `tests/` —
  pytest depends on it.

## Workflow

1. Make the change.
2. `uv run ruff format` and `uv run ruff check --fix`.
3. `uv run mypy src` — clean.
4. `uv run pytest` — green.
5. Commit when the user asks.
