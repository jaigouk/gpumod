import threading

class JobQueue:
    """
    A simple thread‑safe job queue.

    - add_job(job_id, data)      : enqueues a job
    - process_job(job_id, processor) : removes and runs the job and returns True
    - get_result(job_id)        : returns the result of a finished job (or None)
    """

    def __init__(self):
        self.jobs = {}      # job_id → data to be processed
        self.results = {}     # job_id → result of the processed job
        self.lock = threading.Lock()   # protects all mutual‑exclusion

    def add_job(self, job_id, data):
        """Enqueue a new job.

        Returns the job identifier so the caller can refer to the job later.
        """
        with self.lock:                 # lock while updating the dict
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """
        Process a pending job.

        Returns True if a job was processed, False if the job id was not found.
        """
        with self.lock:                 # lock while reading, computing, and writing
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]   # retrieve the payload
            result = processor(data)    # run the processor (may block other threads)

            # Write the result atomically
            self.results[job_id] = result

            # Remove the job from the queue
            del self.jobs[job_id]
        return True

    def get_result(self, job_id):
        """Return the result for a completed job, or None if it doesn’t exist.)

        The read operation is protected to keep the interface thread‑safe.
        """
        with self.lock:
            return self.results.get(job_id, None)