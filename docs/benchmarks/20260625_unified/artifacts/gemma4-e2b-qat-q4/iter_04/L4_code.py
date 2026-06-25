import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect access to shared resources
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        """Adds a job to the queue. Must be thread-safe."""
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """Processes a job, ensuring atomic access to jobs and results."""
        # Acquire the lock before checking or modifying self.jobs
        with self.lock:
            if job_id not in self.jobs:
                return False

            # 1. Safely retrieve data
            data = self.jobs[job_id]

            # Note: Depending on where the processor runs, you might consider
            # releasing the lock during the actual processing, but since
            # the deletion and results storage happen immediately after, 
            # keeping the lock held during these steps guarantees atomicity.
            result = processor(data)

            # 2. Safely store result
            self.results[job_id] = result

            # 3. Safely remove job
            del self.jobs[job_id]

            return True

    def get_result(self, job_id):
        """Retrieves a result, ensuring thread-safe reading."""
        with self.lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == '__main__':
    queue = JobQueue()

    job_id_1 = queue.add_job("job_1", "Data A")
    job_id_2 = queue.add_job("job_2", "Data B")

    def worker(jid):
        print(f"Processing job: {jid}")
        if queue.process_job(jid, lambda data: f"Processed: {data}"):
            print(f"Job {jid} completed successfully.")
        else:
            print(f"Job {jid} not found.")

    thread1 = threading.Thread(target=worker, args=(job_id_1,))
    thread2 = threading.Thread(target=worker, args=(job_id_2,))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    print("\n--- Final Results ---")
    print(f"Result for job_1: {queue.get_result(job_id_1)}")
    print(f"Result for job_2: {queue.get_result(job_id_2)}")