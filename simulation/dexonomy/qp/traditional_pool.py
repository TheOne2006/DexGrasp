"""Opt-in CPU-only workers for the unchanged boundary ContactQP solve."""
import os


def initialize():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import torch
    torch.set_num_threads(1)


def solve_job(job):
    from dexonomy.qp.qp_single import ContactQP
    pos, normal, wrench, center, friction = job
    return ContactQP(miu_coef=friction).solve(pos, normal, wrench, center)


class TraditionalQPPool:
    """Bank-owned independent QPs, ordered results and explicit shutdown.

    Threads share no solver or host-model state. Spawn is available for measured
    comparison; neither path sends CUDA arrays or mutable simulation hosts.
    """
    def __init__(self, workers, backend='thread'):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        from dexonomy.util.cpu_slots import detect_available_cpu_slots
        if not 1 <= workers <= detect_available_cpu_slots():
            raise ValueError('Traditional QP workers exceed affinity/cgroup budget')
        self.workers = workers
        self.backend = backend
        if backend == 'thread':
            self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='traditional-qp')
        elif backend == 'process':
            self.executor = ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn'), initializer=initialize)
        else:
            raise ValueError('Traditional QP pool backend must be thread or process')

    def solve(self, jobs):
        return list(self.executor.map(solve_job, jobs, chunksize=1))

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
