# Contributing to READ

Our project welcomes external contributions. If you have an itch, please feel
free to scratch it.

To contribute code or documentation, please submit a [pull request](https://github.com/IBM/READ/pulls).

A good way to familiarize yourself with the codebase and contribution process is
to look for and tackle low-hanging fruit in the [issue tracker](https://github.com/IBM/READ/issues).
Before embarking on a more ambitious contribution, please quickly [get in touch](#communication) with us.

**Note: We appreciate your effort, and want to avoid a situation where a contribution
requires extensive rework (by you or by us), sits in backlog for a long time, or
cannot be accepted at all!**

### Proposing new features

If you would like to implement a new feature, please [raise an issue](https://github.com/IBM/READ/issues)
before sending a pull request so the feature can be discussed. This is to avoid
you wasting your valuable time working on a feature that the project developers
are not interested in accepting into the code base.

### Fixing bugs

If you would like to fix a bug, please [raise an issue](https://github.com/IBM/READ/issues) before sending a
pull request so it can be tracked.

### Merge approval

The project maintainers use LGTM (Looks Good To Me) in comments on the code
review to indicate acceptance. For a list of the maintainers, see the
[MAINTAINERS.md](MAINTAINERS.md) page.

## Legal

Each source file must include a license header for the Apache
Software License 2.0. Using the SPDX format is the simplest approach,
e.g.

```
# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
```

We have tried to make it as easy as possible to make contributions. This
applies to how we handle the legal aspects of contribution. We use the
same approach, the [Developer's Certificate of Origin 1.1 (DCO)](https://elinux.org/Developer_Certificate_Of_Origin), that the
Linux® Kernel community uses to manage code contributions.

We simply ask that when submitting a patch for review, the developer
must include a sign-off statement in the commit message:

```
Signed-off-by: John Doe <john.doe@example.com>
```

You can include this automatically when you commit a change to your
local git repository using the following command:

```
git commit -s
```

## Communication

Please use [GitHub issues](https://github.com/IBM/READ/issues) for questions,
bug reports and feature discussions.

## Setup

```bash
git clone https://github.com/IBM/READ.git
cd READ
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

## Testing

The test suite is fully offline (no API keys needed):

```bash
pytest tests/ -v
```

Please add tests for new metrics in `tests/test_metrics.py` and for new
perturbations in `tests/test_parser.py`, and make sure the whole suite passes
before opening a pull request.

## Coding style guidelines

- Python ≥ 3.10, standard library first; keep dependencies minimal.
- Type hints on public functions.
- Module-level docstrings explaining what each module does.
