Development
===============

This page describes how to set up a development environment and which
checks must pass before changes are merged.

Setup
-----

The project uses `uv <https://docs.astral.sh/uv/>`_ for environment and
dependency management. Clone and install all development tools with:

.. code-block:: console

   git clone https://github.com/dtudk/nest-soc.git
   cd nest-soc
   uv sync --group dev

This creates a virtual environment with every tool version pinned in
``uv.lock``. Run commands inside it with ``uv run``, e.g. ``uv run python``.

Checks
------

All of the following must pass before changes are merged. The same
commands run in CI on every pull request.

Linting and formatting
^^^^^^^^^^^^^^^^^^^^^^

Ruff is used for linting and formatting:

.. code-block:: console

   uv run ruff check .
   uv run ruff format --check .

Type checking
^^^^^^^^^^^^^^

.. code-block:: console

   uv run ty check

Dependencies
^^^^^^^^^^^^

`deptry <https://deptry.com/>`_ checks that all declared dependencies
are used and all imports are declared:

.. code-block:: console

   uv run deptry .

Tests
^^^^^

.. code-block:: console

   uv run pytest

Documentation
-------------

The documentation is built with Sphinx and published to Read the Docs.
Build it locally with the same strictness Read the Docs uses:

.. code-block:: console

   uv run sphinx-build -W -b html docs/source docs/build/html

New public functions and classes are documented automatically from their
docstrings via sphinx-autoapi — no manual pages needed. Use Google- or
NumPy-style docstrings; napoleon renders both.

Pull requests
--------------

- CI checks (ruff, ty, deptry, pytest) must pass.
- The documentation build must be warning-free.
- Docs build as a PR preview on Read the Docs.
