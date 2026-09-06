# Contributing to TMC-LM

Thank you for wanting to contribute! Please read the following guidelines before opening a pull request.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/tmc-llm.git
   cd tmc-llm
   ```

2. **Install Python dependencies**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install -e .
   ```

3. **Verify the environment**
   ```bash
   python -m pytest --collect-only  # should discover 7 tests
   ```

4. **Optional: Docker for full pipeline**
   - Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL 2 backend.
   - Enable NVIDIA GPU support if you have an NVIDIA GPU.

## Adding New Source Documents

1. Place your document file in `data/raw/tmc_sources/`.
   Supported formats: `.txt`, `.md`, `.pdf`, `.docx`, `.xlsx`, `.csv`, `.json`.
2. Rebuild the dataset:
   ```powershell
   .\scripts\prepare_dataset.ps1
   ```
3. Re-train the LoRA adapter:
   ```powershell
   .\scripts\train_lora.ps1
   ```

## Code Conventions

- Follow the existing code style (PEP 8, snake_case for functions/variables).
- Add or update tests in `tests/` for any new functionality.
- Run the existing test suite before submitting:
  ```bash
  python -m pytest tests/
  ```
- Ensure `train_lora.py`, `dataset_builder.py`, and `document_loader.py` have appropriate docstrings.
- Use type hints where practical (the codebase uses `from __future__ import annotations`).

## Pull Request Process

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes with a clear description:
   ```bash
   git commit -m "Add: new feature description"
   ```
3. Push to your fork and open a Pull Request against the `main` branch.
4. The PR should include:
   - A summary of the change.
   - Motivation/rationale.
   - Any relevant screenshots or output snippets.
   - Tests covering the new functionality.

## Reporting Issues

- Use the [GitHub Issues](https://github.com/your-username/tmc-llm/issues) tracker.
- Include your environment (OS, Python version, hardware).
- Attach relevant logs or error messages.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 license.