import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect the shared state (self.jobs and self.results)
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        """Adds a job safely."""
        with self._lock:
            self.jobs[job_id] = data
            return job_id

    def process_job(self, job_id, processor):
        """Processes a job safely."""
        # We need the lock when checking and accessing self.jobs
        with self._lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            # NOTE: We perform the actual processing *outside* the critical section
            # to allow other threads to access the queue while this job runs, 
            # but the state transition (deletion and result storage) must be locked.

        # Perform the expensive processing outside the lock (optional, but good practice)
        result = processor(data)

        # Re-acquire the lock to safely update the results and remove the job
        with self._lock:
            # This critical section now protects both writing to results and deleting from jobs
            self.results[job_id] = result
            del self.jobs[job_id]
            return True

    def get_result(self, job_id):
        """Retrieves a result safely."""
        with self._lock:
            return self.results.get(job_id)

# Example usage (optional, for testing the fix):
if __name__ == '__main__':
    queue = JobQueue()

    # Test adding jobs
    for i in range(10):
        queue.add_job(f"job_{i}", {"input": i})

    # Test processing jobs in separate threads
    threads = []

    # Example processor function
    def my_processor(data):
        import time
        import threading
        time.sleep(0.01) # Simulate work
        return f"Processed {data['input']} successfully"

    for i in range(10):
        t = threading.Thread(target=queue.process_job, args=(f"job_{i}", my_processor))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Test retrieving results
    print("\nRetrieving results:")
    print(f"Result for job_5: {queue.get_result('job_5')}")
    print(f"Result for job_9: {queue.get_result('job_9')}")
    print(f"Result for job_10 (not processed): {queue.get_result('job_10')}")