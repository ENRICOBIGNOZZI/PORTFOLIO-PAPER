"""Compatibility alias for the final Monte Carlo extension.

The branch's validated DGP and estimator are implemented in ``finance_mc_core``.
The final comparative-statistics scripts import this module name so that the same
source release runs both standalone and inside the repository.
"""
from finance_mc_core import *  # noqa: F401,F403
