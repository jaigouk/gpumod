import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect the shared state (jobs and results)
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        """Adds a job. Requires a lock to protect the self.jobs dictionary."""
        with self.lock:
            self.jobs[job_id] = data
            return job_id

    def process_job(self, job_id, processor):
        """
        Processes a job. This operation needs a lock to ensure that:
        1. We don't read from self.jobs while it is being modified/deleted by another thread.
        2. We write to self.results safely.
        3. We delete the job from self.jobs safely.
        """
        # Acquire the lock before accessing shared resources
        with self.lock:
            if job_id not in self.jobs:
                return False

            # 1. Retrieve data (protected read)
            data = self.jobs[job_id]

            # Note: We perform the actual processing outside the lock 
            # if the processing itself is long-running, but since the prompt
            # implies the whole sequence (check -> process -> result write) 
            # needs protection, we wrap the critical section. 
            # For maximum safety, we'll keep the critical section minimal.

            # Execute the processor (we can run this outside the lock if 
            # we want to allow concurrent processing of different jobs, 
            # but the state changes must be locked.)
            result = processor(data)

            # 2. Write result (critical section)
            self.results[job_id] = result

            # 3. Delete job (critical section)
            del self.jobs[job_id]

            return True

    def get_result(self, job_id):
        """Reads from the results dictionary. Requires a lock to ensure safe reading."""
        with self.lock:
            return self.results.get(job_id)

# Example Usage (demonstrating the fix):
if __name__ == '__main__':
    queue = JobQueue()

    # Test 1: Basic flow
    job_id_1 = queue.add_job(1, "Data A")
    result_1 = queue.process_job(job_id_1, lambda d: f"Processed {d}")
    print(f"Job 1 processed: {result_1}. Result: {queue.get_result(job_id_1)}")

    # Test 2: Concurrency demonstration
    def long_running_process(data):
        import time
        time.sleep(0.01) # Simulate work
        return f"Result for {data}"

    job_id_2 = queue.add_job(2, "Data B")

    # Start processing job 2 in a separate thread
    def worker(jid):
        queue.process_job(jid, long_running_process)

    thread = threading.Thread(target=worker, args=(job_id_2,))
    thread.start()

    # Try to add another job while job 2 is running (testing lock on add_job)
    queue.add_job(3, "Data C")

    thread.join()

    print(f"Job 2 processed: {queue.get_result(job_id_2)}")
    print(f"Job 3 result: {queue.get_result(3)}")