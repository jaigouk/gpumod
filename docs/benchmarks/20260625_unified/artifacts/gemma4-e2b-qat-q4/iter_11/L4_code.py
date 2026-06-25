import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect the shared resources (self.jobs and self.results)
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Acquire the lock before modifying the jobs dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Acquire the lock to ensure thread safety during the entire job processing lifecycle
        with self.lock:
            if job_id not in self.jobs:
                return False

            # 1. Read data (protected)
            data = self.jobs[job_id]

            # NOTE: We run the processor *inside* the lock. 
            # In a production system, heavy processing should ideally happen outside 
            # the lock, but for this class structure, locking the state transition is essential.
            result = processor(data) 

            # 2. Write result (protected)
            self.results[job_id] = result

            # 3. Delete job from pending queue (protected)
            del self.jobs[job_id]
            return True

    def get_result(self, job_id):
        # Reading results should also be protected, though less critical than write operations
        with self.lock:
            return self.results.get(job_id)

# Example usage (for testing purposes - optional)
if __name__ == '__main__':
    queue = JobQueue()

    def worker(q: JobQueue, job_id):
        if q.process_job(job_id, lambda d: "Result for " + str(d)):
            print(f"Job {job_id} processed successfully.")

    # Start several threads to process jobs
    threads = []
    for i in range(5):
        job_id = f"Job-{i}"
        queue.add_job(job_id, f"Data for Job {i}")
        t = threading.Thread(target=worker, args=(queue, job_id))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("\nFinal results:")
    for i in range(5):
        print(f"Job-{i} result: {queue.get_result(f'Job-{i}')}")