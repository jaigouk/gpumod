import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared data
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to ensure thread-safe modification of the jobs dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock to atomically check existence and retrieve the job
        with self.lock:
            if job_id not in self.jobs:
                return False
            # Using pop() removes the job from the queue immediately.
            # This prevents another thread from picking up the same job 
            # while this thread is processing it.
            data = self.jobs.pop(job_id)
        
        # Process the job OUTSIDE the lock.
        # This is best practice to allow other threads to add or check 
        # other jobs while this one is running.
        result = processor(data)

        # Lock to safely store the result in the results dictionary
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock to ensure thread-safe reading of the results dictionary
        with self.lock:
            return self.results.get(job_id)