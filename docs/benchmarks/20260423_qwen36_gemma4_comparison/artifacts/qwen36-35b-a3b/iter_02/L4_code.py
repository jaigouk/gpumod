import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # 1. Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # 2. Lock the dictionary while adding a job
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 3. Lock to safely check existence and retrieve the job
        # We must hold the lock to prevent another thread from 
        # deleting or modifying the job between the check and the access.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs[job_id]
        
        # 4. Process the job outside the lock.
        # Holding the lock during processing would block other threads 
        # from adding jobs or retrieving results, reducing performance.
        result = processor(data)
        
        # 5. Lock to safely store the result and remove the job
        with self.lock:
            self.results[job_id] = result
            del self.jobs[job_id]
        
        return True

    def get_result(self, job_id):
        # 6. Lock to safely read the result
        with self.lock:
            return self.results.get(job_id)