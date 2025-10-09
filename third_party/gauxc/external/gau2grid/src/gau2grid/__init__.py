"""
Gau2grid base init
"""

from . import RSH, codegen, order
from . import c_generator as c_gen
from . import python_reference as ref

# Handle versioneer
from ._version import get_versions

# Pull in code from the c wrapper
from .c_wrapper import (
    c_compiled,
    cgg_path,
    collocation,
    collocation_basis,
    get_cgg_shared_object,
    ncomponents,
    orbital,
    orbital_basis,
)

# Pull in tests
from .extras import test

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
