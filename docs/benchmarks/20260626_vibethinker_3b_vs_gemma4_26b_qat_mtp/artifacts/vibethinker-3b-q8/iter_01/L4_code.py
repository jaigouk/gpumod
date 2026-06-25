import threading

class JobQueue:
    """
    A simple thread‑safe job queue.

    - add_job(job_id, data)   → int
      Enqueues a job. Returns its id so the caller can locate the data.

    - process_job(job_id, processor) → bool
      Runs `processor` on the stored data of `job_id`. The job is removed from the queue
      and the result is stored in `results`. Returns True if the job existed, False otherwise.

    - get_result(job_id) → int or None
      Returns the result for `job_id` (or None if it has not been computed yet).
    """

    def __init__(self):
        self.jobs = {}      # job_id → data (still to be processed)
        self.results = {}     # job_id → computed result
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        """Enqueue a new job.

        The method is protected by a lock so other threads cannot insert the same
        job_id simultaneously, guaranteeing uniqueness.
        """
        with self.lock:
            if job_id in self.jobs:
                raise ValueError(f"Job {job_id} already in the queue")
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """Process a pending job.

        The job is removed from the queue, its result is computed, and stored.
        All dictionary operations are locked to avoid race conditions.
        """
        with self.lock:
            if job_id not in self.jobs:
                return False

            # pop removes the entry while we fetch the data – this eliminates a
            # subtle race where the entry could be mutated after the check.
            data = self.jobs.pop(job_id)

            result = processor(data)
            self.results[job_id] = result
        return True

    def get_result(self, job_id):
        """Retrieve the result for a processed job.

        The read is also locked so that a later thread cannot overwrite the entry
        before it has a chance to read it.
        """
        with self.lock:
            return self.results.get(job_id)