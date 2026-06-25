import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize the lock to manage thread access
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Protect the addition of jobs to the queue
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        with self.lock:
            # Check if the job exists
            if job_id not in self.jobs:
                return False

            # Retrieve data and remove the job from the queue immediately.
            # This "claims" the job, preventing other threads from processing it.
            data = self.jobs[job_id]
            del self.jobs[job_id]

        # Process the job outside the lock to allow concurrency.
        # Other threads can add jobs or check results while this job is being processed.
        # Note: If the processor raises an exception here, the job is considered 
        # failed and removed from the queue.
        result = processor(data)

        with self.lock:
            # Save the result safely
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Protect reading from the results dictionary
        with self.lock:
            return self.results.get(job_id)