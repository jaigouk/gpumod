import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        self.lock = threading.Lock()  # Initialize the lock

    def add_job(self, job_id, data):
        # Protect the shared 'jobs' dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Claim the job atomically
        with self.lock:
            if job_id not in self.jobs:
                return False
            
            # Retrieve data and remove job from queue to claim it
            data = self.jobs[job_id]
            del self.jobs[job_id]
        
        # 2. Process the job outside the lock
        # (This allows other threads to access the queue while this job is being processed)
        try:
            result = processor(data)
            
            # 3. Store the result atomically
            with self.lock:
                self.results[job_id] = result
        except Exception:
            # Optional: Handle processor failures. 
            # Note: The job was already removed from self.jobs during 'claim',
            # so it will not be re-queued unless you implement retry logic here.
            pass
            
        return True

    def get_result(self, job_id):
        # Protect the shared 'results' dictionary
        with self.lock:
            return self.results.get(job_id)