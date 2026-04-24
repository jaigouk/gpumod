import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to manage thread-safe access to shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Protect the 'jobs' dictionary while adding a new job
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Acquire lock to safely check existence and retrieve the job
        with self.lock:
            if job_id not in self.jobs:
                return False
            # Using pop() is atomic: it removes the job from the queue
            # and returns the data in one step. This prevents multiple
            # threads from processing the same job_id.
            data = self.jobs.pop(job_id)
        
        # Perform the actual processing outside the lock.
        # This allows other threads to add or process other jobs
        # while this processor is busy.
        result = processor(data)

        # Acquire lock to safely write the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Protect the 'results' dictionary while reading
        with self.lock:
            return self.results.get(job_id)