import quacc.dataset as dataset  # noqa: F401
import quacc.error as error  # noqa: F401
import quacc.plot as plot  # noqa: F401
import quacc.utils.commons as commons  # noqa: F401
from quacc.environment import env


def _get_njobs(n_jobs):
    return env["N_JOBS"] if n_jobs is None else n_jobs
