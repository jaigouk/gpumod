import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            # Remove the job from the queue to prevent multiple threads from processing it
            del self.jobs[job_id]

        # Process the job outside the lock to avoid blocking other jobs
        result = processor(data)

        with self.lock:
            # Safe to write the result as no other thread can be writing to this job_id
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)