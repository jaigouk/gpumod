import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect the shared resources
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        # Acquire lock before writing to self.jobs
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # --- Critical Section Start ---
        # We need the lock to ensure that reading self.jobs and writing to self.results
        # and deleting from self.jobs happen atomically.
        with self._lock:
            if job_id not in self.jobs:
                return False

            # 1. Get data (Read)
            data = self.jobs[job_id]

            # Note: If the processor function is long-running, 
            # holding the lock here will block other threads. 
            # In a production system, processing should happen outside the lock.
            # However, for this specific implementation fix, we include it inside the lock 
            # to ensure data consistency of the queue state transitions.
            result = processor(data)

            # 2. Record result (Write)
            self.results[job_id] = result

            # 3. Clean up job entry (Delete)
            del self.jobs[job_id]

        # --- Critical Section End ---
        return True

    def get_result(self, job_id):
        # Acquire lock to safely read from self.results
        with self._lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == '__main__':
    queue = JobQueue()

    # 1. Add a job
    job1_id = queue.add_job("job1", {"input": 10})

    # 2. Define a processor function
    def mock_processor(data):
        print(f"Processing job {data['input']}...")
        # Simulate work time
        import time
        time.sleep(0.1) 
        return f"Processed: {data['input'] * 2}"

    # 3. Start a thread to process the job
    t1 = threading.Thread(target=queue.process_job, args=(job1_id, mock_processor))
    t1.start()

    # 4. Check result (will be None initially)
    print(f"Result before completion: {queue.get_result(job1_id)}")

    # 5. Wait for the thread to finish
    t1.join()

    # 6. Check final result
    final_result = queue.get_result(job1_id)
    print(f"Final result: {final_result}")