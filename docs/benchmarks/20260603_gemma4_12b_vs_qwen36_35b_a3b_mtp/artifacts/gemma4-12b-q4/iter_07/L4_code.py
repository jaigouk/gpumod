import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to the dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, Processor):
        # Step 1: Atomic "Claim" of the job
        # We pop the job out of the dictionary immediately. 
        # If two threads call this at once, only one will successfully pop it.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # Step 2: Process the data outside of the lock.
        # This allows other threads to continue adding/processing other jobs.
        try:
            result = Processor(data)
        except Exception as e:
            # Handle or log exception if necessary
            raise e

        # Step 3: Save the result safely
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# Example usage:
if __name__ == "__main__":
    def sample_processor(data):
        # Simulate work
        return f"Processed: {data}"

    queue = JobQueue()
    queue.add_job("job1", "Some Data")

    # This is now thread-safe
    success = queue.process_job("job1", sample_processor)
    print(f"Success: {success}, Result: {queue.get_result('job1')}")