# Contributing to Neuromorphic Sleep Stage Scoring

Thank you for your interest in contributing! This guide will help you get started.

## Code of Conduct

Be respectful, inclusive, and supportive of all contributors. We are committed to providing a welcoming environment for everyone.

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/neuromorphic-sleep-staging-pipeline-project.git
cd neuromorphic-sleep-staging-pipeline-project
```

### 2. Set Up the Development Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"  # Once we add dev dependencies
```

### 3. Pick an Issue

Look for issues labeled `good first issue` or `help wanted`. Comment on the issue to let us know you're working on it, so we don't have duplicates.

## Contribution Types

### Documentation
- Typo fixes in README or docstrings
- Adding examples or tutorials
- Improving clarity in ARCHITECTURE.md

**Process:**
1. Fork, make edits, commit with a clear message.
2. Submit a PR with "Docs:" prefix (e.g., "Docs: fix typo in README").

### Bug Fixes
- Issues labeled `bug` with clear reproduction steps.
- Code that handles edge cases (e.g., missing EDF files, Windows encoding).

**Process:**
1. Write a test that reproduces the bug first (if applicable).
2. Fix the bug.
3. Ensure all tests pass: `pytest tests/ -v`
4. Submit a PR with "Fix:" prefix (e.g., "Fix: Windows UTF-8 logging").

### Features
- New dataset support (ISRUC-Sleep, etc.)
- New loss functions (Focal Loss, SWDF).
- Visualization tools (confusion matrix, attention maps).
- Performance improvements.

**Process:**
1. Comment on the issue to discuss the approach (or create an issue first).
2. Work on a feature branch: `git checkout -b feature/my-feature-name`
3. Write tests for your code.
4. Submit a PR with "Feature:" prefix (e.g., "Feature: add ISRUC-Sleep loader").

## Pull Request Process

### Before Submitting

1. **Run tests locally:**
   ```bash
   pytest tests/ -v
   ```

2. **Check code style** (optional, no strict enforcement yet):
   ```bash
   python -m black src/sleep_staging/  # Format code
   python -m pylint src/sleep_staging/ # Lint
   ```

3. **Update CHANGELOG.md** (if adding/changing features):
   ```
   ## [Unreleased]
   ### Added
   - New confusion matrix visualization in evaluate.py
   
   ### Fixed
   - Windows UTF-8 encoding bug in train.py
   ```

### Submitting a PR

1. Push your branch to your fork.
2. Open a PR on the main repo.
3. Fill out the PR template with:
   - **What does this PR do?** (brief description)
   - **Why?** (context or which issue it closes: `Closes #3`)
   - **Testing** (how to verify your changes work)
   - **Screenshots/Output** (if applicable)

### Example PR Title

- `Fix: Windows UTF-8 encoding in train.py`
- `Feature: Add ISRUC-Sleep dataset loader`
- `Docs: Clarify data preprocessing steps`

## Code Style Guidelines

We don't have strict formatting yet, but please follow these principles:

- **Docstrings:** Include a one-line summary + description for functions.
  ```python
  def bandpass_filter(x, fs, lo, hi):
      """Apply bandpass filter to remove DC and high-frequency noise.
      
      Args:
          x: Input signal (ndarray, shape (N,)).
          fs: Sampling rate in Hz.
          lo: Low-frequency cutoff (Hz).
          hi: High-frequency cutoff (Hz).
      
      Returns:
          Filtered signal (ndarray, shape (N,)).
      """
  ```

- **Comments:** Explain the "why", not the "what".
  ```python
  # Use robust normalization to handle outlier artifacts
  x = (x - np.median(x)) / np.std(x)
  ```

- **Type hints** (optional but encouraged for new code):
  ```python
  def load_manifest(csv_path: str) -> list[dict]:
      """Load manifest CSV."""
  ```

## Testing

All new features should include tests. We use `pytest`.

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_preprocess.py -v

# Run with coverage
pytest tests/ --cov=src/sleep_staging
```

Example test:

```python
import pytest
from sleep_staging.preprocess import bandpass_filter
import numpy as np

def test_bandpass_filter_removes_dc():
    """Verify bandpass filter removes DC component."""
    x = np.ones(1000) + 100  # Signal with DC offset
    y = bandpass_filter(x, fs=100, lo=0.5, hi=30)
    assert abs(np.mean(y)) < 1.0  # DC should be near zero
```

## Reporting Issues

Found a bug? Please open an issue with:

1. **Title:** Clear, one-line description.
2. **Description:** What did you expect? What happened instead?
3. **Steps to reproduce:**
   ```
   1. Run `python -m sleep_staging.cli build-all --mode synthetic`
   2. Observe error on line X
   ```
4. **Environment:**
   ```
   OS: Windows 11
   Python: 3.10
   PyTorch: 2.3
   ```

## Questions?

- Check the README and ARCHITECTURE.md first.
- Comment on the issue you're working on.
- Open a discussion if it's not a specific bug or feature.

## Recognition

Contributors will be recognized in:
- A `CONTRIBUTORS.md` file in the main repo.
- GitHub's contributor graph.
- Project publications or presentations (with permission).

---

**We're excited to have you here!** Feel free to reach out if you get stuck. Happy coding! 🚀
