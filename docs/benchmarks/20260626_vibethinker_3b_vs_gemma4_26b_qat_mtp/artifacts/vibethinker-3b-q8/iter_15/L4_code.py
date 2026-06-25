import threading

class JobQueue:
    """
    A simple job queue that avoids the race condition by protecting all
    modifications of the internal dictionaries with a lock.
    ``process_job`` runs concurrently, but any thread that writes to
    ``self.results`` or reads a value must obtain the lock first.
    ``get_result`` now blocks until the result for the requested job_id
    has been computed and stored.
    """

    def __init__(self):
        self.jobs = {}      # job_id -> data (still to be processed)
        self.results = {}    # job_id -> result (already processed)
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        """Enqueue a new job.

        The job is stored atomically in ``self.jobs``.
        Returns the job_id so the caller can pass it to another thread.
        """
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """Process a pending job.

        The method acquires the lock so that the lookup, computation,
        and deletion are atomic.  Because the lock is held while the
        result is written to ``self.results`` , any thread calling
        ``get_result`` will block until the write finishes.
        """
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            result = processor(data)

            # Write the result atomically
            self.results[job_id] = result

            # The job is considered finished
            del self.jobs[job_id]

        return True

    def get_result(self, job_id):
        """Return the result for ``job_id``.

        This method blocks until the result has been computed and stored
        in ``self.results``.  The lock ensures that the waiting thread
        blocks on the writer thread while it is writing the result.
        """
        while True:
            with self.lock:
                if job_id in self.results:
                    return self.results[job_id]
                # If the result is not yet present, wait for the lock to be released.
                # The loop repeats until the result is found.