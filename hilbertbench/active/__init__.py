#!/usr/bin/env python
#
# file: hilbertbench/active/__init__.py
#
# revision history:
#  20260604 (am): cleaned up to project coding standards
#
# hilbertbench.active — Active Mode: controlled, explicitly-authorized
# sampling for diagnostics (e.g. expressibility) that passive observation
# cannot provide.
#
#   from hilbertbench.active import active_probe_qiskit
#------------------------------------------------------------------------------

# import active mode components
#
from hilbertbench.active.probe import (
    active_probe_pennylane,
    active_probe_qiskit,
    probe_expressibility,
)

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# define the public API
#
__all__ = [
    "probe_expressibility",
    "active_probe_qiskit",
    "active_probe_pennylane",
]

#
# end of file
