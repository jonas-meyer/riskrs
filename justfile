default:
    @just --list

[group('ruff')]
lint *files:
    uv run --active ruff check {{ files }}

[group('ruff')]
format *files:
    uv run --active ruff format {{ files }} && uv run ruff check {{ files }} --select I --fix

[group('ruff')]
check: lint format

clean:
    rm -rf .venv dist *egg-info target && uv cache clean
    cd libs/riskrs-core && cargo clean
    cd bindings/riskrs-python && cargo clean

# Build the pure Rust library
build-core:
    cd libs/riskrs-core && cargo build --release

# Build Python package (includes bindings)
build:
    cd bindings/riskrs-python && uv run --active maturin build --release

# Develop Python package (install in current environment)
develop:
    cd bindings/riskrs-python && uv run --active maturin develop

# Test the pure Rust library
test-core:
    cd libs/riskrs-core && cargo test

# Test Python package
test:
    cd bindings/riskrs-python && uv run --active pytest tests/

# Check everything before publishing
check-all: check test-core test