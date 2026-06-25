import threading

class JobQueue:
    """
    A simple thread‑safe job queue.

    Methods
    -------
    add_job(job_id, data) -> int
        Insert a new job into the queue and return its id.

    process_job(job_id, processor) -> bool
        Perform the job associated with `job_id` using `processor`.
        Returns True if the job was processed, False if the job id was not found.

    get_result(job_id) -> Optional[Any]
        Return the result of the job with `job_id`, or None if it hasn't been
        computed yet or does not exist.
    """

    def __init__(self):
        # Storage for pending jobs and completed results.
        self.jobs = {}      # job_id -> data
        self.results = {}    # job_id -> result
        self.lock = threading.Lock()   # Shared lock for all dictionary ops

    def add_job(self, job_id, data):
        """
        Add a new job to the queue.

        Parameters
        ----------
        job_id : hashable
            Unique identifier for the job.
        data : any
            Data associated with the job.

        Returns
        -------
        int
            The returned job_id (for convenience; it is guaranteed to be present
            in `jobs` after the call).
        """
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """
        Process a pending job.

        Parameters
        ----------
        job_id : hashable
            The id of the job to process.
        processor : callable
            Function (or lambda) that takes the job data and returns the result.

        Returns
        -------
        bool
            True if the job was processed, False if the job id was not found.
        """
        with self.lock:
            # Check existence of the job.
            if job_id not in self.jobs:
                return False

            # Retrieve the data (no race possible inside the lock).
            data = self.jobs[job_id]

            # Compute the result.
            result = processor(data)

            # Remove the job from the pending queue.
            del self.jobs[job_id]

            # Store the result.
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        """
        Retrieve the result for a completed job.

        Parameters
        ----------
        job_id : hashable
            The id of the job whose result is requested.

        Returns
        -------
        Any or None
            The result stored under `job_id` in `results`, or None if the job
            has not been processed yet or does not exist.
        """
        with self.lock:
            return self.results.get(job_id)