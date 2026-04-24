import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock is needed here to safely write to the shared dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # We must hold the lock for the entire critical section:
        # 1. Checking if the job exists
        # 2. Reading the data
        # 3. Writing the result
        # 4. Deleting the job
        # 
        # Note: Holding the lock during 'processor(data)' serializes the execution
        # of jobs. This is necessary in this specific implementation because the
        # job is deleted from self.jobs only *after* processing. If we released
        # the lock before processing, another thread could grab the same job_id,
        # leading to duplicate processing or a KeyError when trying to delete it.
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            
            # Perform the work
            result = processor(data)
            
            # Update shared state atomically
            self.results[job_id] = result
            del self.jobs[job_id]
            
            return True

    def get_result(self, job_id):
        # Lock is needed here to safely read from the shared dictionary
        with self.lock:
            return self.results.get(job_id)