import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Create a lock to synchronize access to shared data
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to safely add jobs to the queue
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock to ensure only one thread can check and retrieve a job at a time
        with self.lock:
            if job_id not in self.jobs:
                return False

            # Atomically retrieve and remove the job from the queue.
            # Using pop() prevents other threads from picking up this job 
            # while this thread is processing it.
            data = self.jobs.pop(job_id)

        # Process the job outside the lock. 
        # This allows other threads to pick up different jobs concurrently.
        result = processor(data)

        # Lock to safely store the result in the results dictionary
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock to safely read the result
        with self.lock:
            return self.results.get(job_id)