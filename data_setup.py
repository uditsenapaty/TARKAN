"""Compatibility entry point for explicit local data preparation.

Run 'python data_setup.py --help' for the same stages as prepare.py.
Dataset files and licensed KG resources must be supplied locally.
"""
from prepare import main

if __name__ == "__main__":
    main()

