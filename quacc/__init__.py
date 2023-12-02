import quacc.dataset as dataset
import quacc.error as error
import quacc.logger as logger
import quacc.plot as plot
import quacc.utils as utils
from quacc.environment import env


def _get_njobs(n_jobs):
    return env.N_JOBS if n_jobs is None else n_jobs
