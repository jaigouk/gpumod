import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared resources
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
            return job_id

    def process_job(self, job_id, processor):
        # Acquire lock before accessing or modifying shared state
        with self._lock:
            if job_id not in self.jobs:
                return False

            # 1. Retrieve data (Read from self.jobs)
            data = self.jobs[job_id]

            # NOTE: If the processor is computationally expensive, it might be better
            # to release the lock before calling processor, but since we need 
            # atomic update of self.jobs and self.results, we keep the lock here.
            # For this fix, we assume the processing is quick or that the lock protects 
            # the dictionary operations entirely.
            result = processor(data)

            # 2. Write result (Write to self.results)
            self.results[job_id] = result

            # 3. Delete from jobs (Modify self.jobs)
            del self.jobs[job_id]
            return True

    def get_result(self, job_id):
        # We can use a lock here if we want to guarantee a consistent read 
        # immediately after a write has finished, but for simple lookups 
        # in Python dictionaries, we can often omit the lock if the 
        # underlying operations (like read) aren't structurally changing the map.
        # However, to be safe and consistent:
        with self._lock:
            return self.results.get(job_id)

# Example usage (optional verification):
if __name__ == '__main__':
    queue = JobQueue()

    # Thread 1: Add a job
    t1 = threading.Thread(target=queue.add_job, args=("job1", "data_for_job1"))
    t1.start()

    # Thread 2: Add another job
    t2 = threading.Thread(target=queue.add_job, args=("job2", "data_for_job2"))
    t2.start()

    t1.join()
    t2.join()

    print(f"Job 1 result: {queue.get_result('job1')}")
    print(f"Job 2 result: {queue.get_result('job2')}")
    print(f"Non-existent job result: {queue.get_result('nonexistent')}")