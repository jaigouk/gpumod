class JobQueue:
    def __init__(self):
        self._jobs = []               # FIFO list of jobs
        self._next_to_process = 0     # Index of the next job to be processed

    def add_job(self, job_id: str, data: dict) -> str:
        """Add a job to the queue and return its unique job_id."""
        self._jobs.append({'job_id': job_id, 'data': data, 'result': None})
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """Return the result of the specified job, processing in FIFO order."""
        try:
            target_index = self._jobs.index({'job_id': job_id})
        except ValueError:
            return None

        # Process any pending jobs up to the target job to maintain FIFO behavior
        for i in range(self._next_to_process, target_index + 1):
            self._process_job(self._jobs[i])
            self._next_to_process = i + 1

        # Return the computed result (or None if not yet computed)
        result = self._jobs[target_index]['result']
        return result if result is not None else None

    def _process_job(self, job_entry):
        """Simulate job execution and set its result."""
        # If result is already computed, skip processing
        if job_entry['result'] is not None:
            return
        # Simulate execution: store a copy of the data as the result
        job_entry['result'] = job_entry['data'].copy()