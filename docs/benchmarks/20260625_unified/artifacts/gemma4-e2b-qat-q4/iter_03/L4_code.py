import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect access to shared resources
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Use the lock to ensure that reading, processing, 
        # writing to results, and deleting from jobs is an atomic operation
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]

            # Perform the job processing (Note: This CPU-intensive part happens 
            # while the lock is held, which might block other queue operations. 
            # In a high-performance system, you might release the lock before 
            # running the heavy 'processor' function, but for simple thread 
            # safety, protecting the write operations is key.)
            result = processor(data)

            self.results[job_id] = result

            del self.jobs[job_id]
            return True

    def get_result(self, job_id):
        # Use the lock to ensure consistent reading of results
        with self.lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == '__main__':
    queue = JobQueue()

    def sample_processor(data):
        # Simulate work
        return f"Processed: {data}"

    # Add a job
    job_id = queue.add_job(1, "Task A")
    print(f"Job added: {job_id}")

    # Process the job
    success = queue.process_job(job_id, sample_processor)
    print(f"Job processed successfully: {success}")

    # Get the result
    result = queue.get_result(job_id)
    print(f"Result retrieved: {result}")