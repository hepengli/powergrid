"""
Optimization module for power grid control.

This module contains Mixed-Integer Second-Order Cone Programming (MISOCP) solvers
for optimal power flow problems on IEEE test systems.

Note: These modules require pyscipopt to be installed separately:
    pip install pyscipopt
"""

# Import optimization solvers when needed
# from .misocp import *
# from .misocp_ieee123 import *

__all__ = [
    "misocp",
    "misocp_ieee123",
]
