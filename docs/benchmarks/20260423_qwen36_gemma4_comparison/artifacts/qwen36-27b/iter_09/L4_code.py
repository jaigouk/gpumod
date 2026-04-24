import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared data structures
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Protect the shared 'jobs' dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Phase 1: Claim the job
        # We must acquire the lock to safely check for the job's existence
        # and retrieve it.
        with self.lock:
            if job_id not in self.jobs:
                return False
            
            # Atomically retrieve the data and remove the job from the queue.
            # Removing it here (claiming it) prevents other threads from 
            # picking up the same job while this thread processes it.
            data = self.jobs.pop(job_id)

        # Phase 2: Process the job
        # We release the lock here to allow other threads to add jobs or 
        # process other jobs concurrently.
        try:
            result = processor(data)
        except Exception:
            # If processing fails, we return False. 
            # Note: The job has already been removed from the queue.
            return False

        # Phase 3: Store the result
        # Re-acquire the lock to safely write to the results dictionary.
        with self.lock:
            self.results[job_id] = result
        
        return True

    def get_result(self, job_id):
        # Protect the shared 'results' dictionary
        with self.lock:
            return self.results.get(job_id)