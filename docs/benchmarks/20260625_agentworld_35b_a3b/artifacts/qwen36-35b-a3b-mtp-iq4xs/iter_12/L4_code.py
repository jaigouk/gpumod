import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock access to self.jobs to prevent race conditions during addition
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock the entire critical section to ensure atomicity of check, process, write, and delete
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]

            # Note: Holding the lock during processor(data) execution ensures that
            # no other thread can interfere with the job state. 
            # While this blocks other threads, it is the safest fix for this synchronous structure.
            result = processor(data)

            # Write result safely
            self.results[job_id] = result

            # Remove job safely
            del self.jobs[job_id]

        return True

    def get_result(self, job_id):
        # Lock access to self.results to ensure consistent reads
        with self.lock:
            return self.results.get(job_id)