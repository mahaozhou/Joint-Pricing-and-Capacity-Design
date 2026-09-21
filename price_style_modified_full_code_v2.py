"""Compatibility entry point for the fully revised implementation.
Historical code is preserved under archive/pre_revision_20260917.
See revision/README.md for the full workflow and final 10,000-replication data.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'revision'))
from experiments import run

if __name__=='__main__':
    run(4)
