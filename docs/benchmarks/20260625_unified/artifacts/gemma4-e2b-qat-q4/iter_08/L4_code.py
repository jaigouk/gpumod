import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect all shared resources
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
            return job_id

    def process_job(self, job_id, processor):
        # Acquire the lock before accessing shared data (self.jobs)
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]

            # Perform the computation *outside* the lock if possible 
            # to avoid blocking other queue operations during long processing.
            # However, since the requirement is fixing the data access race condition,
            # we keep the core logic under the lock for simplicity and safety 
            # in this specific implementation.

            result = processor(data)

            # Writing to self.results and deleting from self.jobs 
            # happens while the lock is held, ensuring atomicity.
            self.results[job_id] = result
            del self.jobs[job_id]

            return True

    def get_result(self, job_id):
        # Acquire the lock to safely read from self.results
        with self.lock:
            return self.results.get(job_id)

# Example usage (optional, for testing the fix):
# if __name__ == '__main__':
#     queue = JobQueue()
#     def my_processor(data):
#         # Simulate work
#         import time
#         time.sleep(0.1)
#         return data * 2

#     job_id_1 = queue.add_job(1, 10)
#     job_id_2 = queue.add_job(2, 20)

#     # Start processing concurrently
#     t1 = threading.Thread(target=queue.process_job, args=(job_id_1, my_processor))
#     t2 = threading.Thread(target=queue.process_job, args=(job_id_2, my_processor))

#     t1.start()
#     t2.start()

#     t1.join()
#     t2.join()

#     print(f"Result for job 1: {queue.get_result(job_id_1)}") # Output: 20
#     print(f"Result for job 2: {queue.get_result(job_id_2)}") # Output: 40