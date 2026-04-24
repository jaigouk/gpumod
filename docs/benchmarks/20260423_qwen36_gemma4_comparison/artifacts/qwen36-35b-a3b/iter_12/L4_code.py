import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to manage thread safety
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Acquire lock before modifying shared state
        with self.lock:
            self.jobs[job_id] = data

    def process_job(self, job_id, processor):
        # Acquire lock to safely check and remove the job from the queue.
        # This atomic check-and-remove prevents multiple threads from processing 
        # the same job_id.
        with self.lock:
            if job_id not in self.jobs:
                return False
            
            data = self.jobs[job_id]
            del self.jobs[job_id]  # Remove job immediately to prevent re-processing
        
        # Process the job outside the lock.
        # This allows other threads to add jobs or query results while this 
        # CPU-bound task is executing.
        try:
            result = processor(data)
        except Exception as e:
            # Optional: Handle errors if necessary, but job is already removed from queue
            result = None 

        # FIX: Acquire lock before writing to shared results dictionary.
        # This prevents the race condition where multiple threads write simultaneously.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Acquire lock before reading shared state
        with self.lock:
            return self.results.get(job_id)