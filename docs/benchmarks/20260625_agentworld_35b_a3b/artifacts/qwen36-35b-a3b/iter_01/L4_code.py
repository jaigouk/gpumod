import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock ensures that adding a job is atomic
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock ensures that the check, retrieval, processing, and storage are handled
        # safely. Note: Holding the lock during 'processor(data)' prevents other
        # threads from adding jobs or getting results until this job is finished.
        # This is the safest approach to prevent race conditions without complex state management.
        with self.lock:
            if job_id not in self.jobs:
                return False

            # Retrieve data
            data = self.jobs[job_id]

            # Process the job
            result = processor(data)

            # Store result
            self.results[job_id] = result

            # Remove job from queue
            del self.jobs[job_id]

        return True

    def get_result(self, job_id):
        # Lock ensures we get a consistent view of the results dictionary
        with self.lock:
            return self.results.get(job_id)