"""Conformance suites run against the fakes in ``conformance.plugin``.

The plugin is registered once, from the root ``tests/conftest.py``.  A real
adapter's test module declares ``pytest_plugins = ["conformance.plugin"]``,
imports a suite's test functions, and overrides the suite's fixtures with
its own adapter (F9).
"""
