import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Introduce a lock to synchronize access to shared resources
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Use the lock to protect modification of self.jobs
        with self.lock:
            self.jobs[job_id] = data
            return job_id

    def process_job(self, job_id, processor):
        # Lock needed to check if job exists and retrieve data (reading self.jobs)
        with self.lock:
            if job_id not in self.jobs:
                return False

            # Retrieve data safely while holding the lock
            data = self.jobs[job_id]

            # Release the lock temporarily if the processor function is CPU-intensive,
            # but for simplicity and thread safety regarding self.results, we keep it.
            # However, for better concurrency, the lock is held only during shared state mutation.

        # Perform the potentially long operation (processor call) outside the lock
        # so that other threads can still add jobs or get results.
        result = processor(data)

        # Lock needed to safely write to self.results and delete from self.jobs
        with self.lock:
            self.results[job_id] = result
            del self.jobs[job_id]

        return True

    def get_result(self, job_id):
        # Use the lock to safely read from self.results
        with self.lock:
            return self.results.get(job_id)

# Example usage (for verification - optional)
if __name__ == '__main__':
    queue = JobQueue()

    def worker(job_id, job_data):
        print(f"Processing Job {job_id}...")
        # Simulate work
        import time
        time.sleep(0.05)
        return f"Result for {job_id}: Processed data {job_data}"

    job_ids = {}

    # Add jobs
    for i in range(5):
        job_id = f"job_{i}"
        data = f"Input {i}"
        job_ids[job_id] = queue.add_job(job_id, data)

    # Start processing jobs in threads
    threads = []
    for jid, data in job_ids.items():
        t = threading.Thread(target=queue.process_job, args=(jid, data))
        threads.append(t)
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    # Get and display results
    print("\n--- Final Results ---")
    for jid in job_ids.keys():
        print(f"Job {jid} result: {queue.get_result(jid)}")