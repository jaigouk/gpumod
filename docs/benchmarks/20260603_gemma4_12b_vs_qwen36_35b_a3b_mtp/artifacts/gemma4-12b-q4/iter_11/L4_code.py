import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to synchronize access to the shared dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, Processor):
        # 1. Claim the job
        # We use the lock only to check and pop the job.
        # Popping ensures that only ONE thread can successfully 
        # claim this specific job_id.
        with self._lock:
            if job_id not in self.jobs:
                return False

            # Removing it here is the key fix:
            # It ensures other threads won't see this job_id 
            # while this thread is still processing it.
            data = self.jobs.pop(job_id)

        # 2. Process the job
        # We release the lock here so other threads can 
        # add/process other jobs while this one is running.
        try:
            result = Processor(data)

            # 3. Store the result
            with self._lock:
                self.results[job_id] = result
            return True
        except Exception as e:
            # Handle or log the exception as needed
            print(f"Error processing job {job_id}: {e}")
            return False

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)