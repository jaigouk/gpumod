import threading
import time

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect the shared resources
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock is not strictly necessary here if job_id is guaranteed unique upon insertion,
        # but it's good practice to lock modifications to self.jobs.
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """
        Processes a job. It must acquire the lock to ensure atomic reading,
        writing results, and deleting the job from self.jobs.
        """

        # 1. Acquire lock before accessing shared state
        with self.lock:
            # 2. Check if job exists (protected by the lock)
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]

            # NOTE: We perform the CPU-intensive 'processor' operation OUTSIDE 
            # the lock if possible, as holding the lock during a long computation 
            # would block all other queue operations. 
            # However, since the critical update steps follow, we proceed.

        # Run the potentially long operation outside the lock
        result = processor(data)


        # 3. Re-acquire lock to safely update shared state
        with self.lock:
            # We must re-check if the job still exists, as another thread 
            # might have processed it between the initial check and re-acquiring the lock.
            if job_id in self.jobs:
                # Store the result
                self.results[job_id] = result

                # Delete the job
                del self.jobs[job_id]
                return True
            else:
                # The job was processed by someone else
                return False

    def get_result(self, job_id):
        # Locking ensures we read a consistent state of the results dictionary
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage ---
if __name__ == '__main__':
    queue = JobQueue()

    def sample_processor(data):
        print(f"Processing job {data['id']} with data: {data['data']}")
        # Simulate work
        time.sleep(0.1)
        return f"Processed successfully: {data['data']}"

    # Add jobs
    job1_id = queue.add_job("J1", {"id": 1, "data": "Task A"})
    job2_id = queue.add_job("J2", {"id": 2, "data": "Task B"})

    # Simulate concurrent processing
    threads = []
    t1 = threading.Thread(target=queue.process_job, args=(job1_id, sample_processor))
    t2 = threading.Thread(target=queue.process_job, args=(job2_id, sample_processor))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    # Check results
    print("\nResults:")
    print(f"Result for J1: {queue.get_result('J1')}")
    print(f"Result for J2: {queue.get_result('J2')}")
    print(f"Result for J3 (non-existent): {queue.get_result('J3')}")