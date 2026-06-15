#!/usr/bin/env python
#
# file: hilbertbench/__init__.py
#
# revision history:
#  20260604 (am): cleaned up to project coding standards
#
# HilbertBench — non-intrusive diagnostic framework for quantum machine
# learning. The recorder, reader, and models import only the standard
# library. The analysis layer is exposed lazily so that recorder-only
# users never pay for numpy/pandas imports.
#
#   from hilbertbench import HilbertTrace
#   trace = HilbertTrace("runs/20260605_xxx")
#
# Source comments reference architectural invariants as INV-NNN (e.g.
# INV-001). These are the framework's non-negotiable guarantees; the
# canonical list lives in docs/reference/invariants.md.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# define the public API
#
__all__ = ["HilbertTrace", "SpanView"]

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def __getattr__(name: str):
    """
    function: __getattr__

    arguments:
     name: the attribute name being accessed

    return:
     the requested attribute object

    description:
     PEP 562 lazy attribute access — keeps `import hilbertbench`
     lightweight. The analysis layer (numpy/pandas) is only imported
     when explicitly requested by the caller.
    """

    # resolve HilbertTrace lazily
    #
    if name == "HilbertTrace":
        from hilbertbench.trace import HilbertTrace
        return HilbertTrace

    # resolve SpanView lazily
    #
    if name == "SpanView":
        from hilbertbench.trace import SpanView
        return SpanView

    # exit ungracefully — unknown attribute
    #
    raise AttributeError(
        f"module 'hilbertbench' has no attribute {name!r}"
    )
#
# end of function

#
# end of file
