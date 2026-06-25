import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock ensures that adding a job is atomic
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock ensures that checking, reading, processing, writing, and deleting
        # happen without interference from other threads.
        with self._lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]

            # Note: In a high-performance scenario, you might want to release 
            # the lock before calling processor(data) to allow other threads 
            # to add jobs or check results. However, to strictly fix the 
            # race condition in this specific logic flow (where the job must 
            # remain in 'jobs' until processed), holding the lock is the safest 
            # approach without refactoring the logic significantly.
            result = processor(data)

            # FIX: This write is now protected by the lock
            self.results[job_id] = result

            del self.jobs[job_id]
            return True

    def get_result(self, job_id):
        # Lock ensures consistent read of the results
        with self._lock:
            return self.results.get(job_id)