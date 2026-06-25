import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}      # job_id → data
        self.results = {}    # job_id → processed result
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        """
        Add a new job to the queue and return its id.
        The assignment is guarded by the lock so that concurrent
        readers/writers cannot see inconsistent state.
        """
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """
        Pick up a job by id, process it and store the result.
        The whole body is locked so that reading a job, computing the
        result and writing the result/erase the job are atomic.
        """
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]          # retrieve the data
            result = processor(data)           # compute the result
            self.results[job_id] = result       # store result
            del self.jobs[job_id]              # remove completed job
            return True

    def get_result(self, job_id):
        """
        Return the result for a completed job.
        Reading is safe without a lock, but wrapping it in the lock
        makes the intent explicit and prevents any possible race with
        future updates.
        """
        with self.lock:
            return self.results.get(job_id)