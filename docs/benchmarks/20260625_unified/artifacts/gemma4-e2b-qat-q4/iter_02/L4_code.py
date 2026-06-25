import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared resources
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
            return job_id

    def process_job(self, job_id, processor):
        # Use the lock to ensure we don't read/write shared state concurrently
        with self._lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]

            # Note: While the processing happens inside the lock, this ensures
            # that the job state (in self.jobs) cannot be modified concurrently 
            # by another thread attempting to add or delete it.

            # Execute the processing (Ideally, heavy processing should happen 
            # outside the lock, but for basic concurrency safety in this structure, 
            # keeping it inside the critical section protects the state transitions).
            result = processor(data)

            # Write to results
            self.results[job_id] = result

            # Delete from jobs
            del self.jobs[job_id]

            return True

    def get_result(self, job_id):
        # Use the lock to safely read from the results dictionary
        with self._lock:
            return self.results.get(job_id)

# Example usage (demonstration):
# q = JobQueue()
# def worker(job_id):
#     def heavy_processor(data):
#         threading.Thread(target=q.process_job, args=(job_id, lambda d: "Processed " + d)).start()
#         return True

# job_id_1 = q.add_job(1, "Item A")
# threading.Thread(target=worker, args=(job_id_1,)).start()
# threading.Thread(target=worker, args=(2,)).start()