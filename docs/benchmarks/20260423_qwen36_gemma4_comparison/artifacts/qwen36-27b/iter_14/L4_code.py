import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Protecting add_job ensures thread-safe insertion into the dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Acquire the lock to ensure atomicity of the check, read, write, and delete operations.
        # This prevents multiple threads from processing the same job simultaneously.
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            
            # Note: We hold the lock during processing. While this serializes execution 
            # (only one job processed at a time), it is the safest direct fix for the 
            # provided logic structure. 
            # For higher performance, one could 'pop' the job here, release the lock, 
            # process, and then re-acquire the lock to store the result.
            result = processor(data)

            self.results[job_id] = result
            del self.jobs[job_id]
        
        return True

    def get_result(self, job_id):
        # Protecting get_result ensures we read a consistent state
        with self.lock:
            return self.results.get(job_id)