import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Create a lock to synchronize access to shared data structures
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
            return job_id

    def process_job(self, job_id, processor):
        data = None

        # --- CRITICAL SECTION 1: Check and retrieve the job ---
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs[job_id]
            # Remove the job from the queue immediately (optional, but good practice)
            del self.jobs[job_id]

        # --- Computation (Safe to do outside the lock as it doesn't touch shared state) ---
        result = processor(data)

        # --- CRITICAL SECTION 2: Store the result ---
        with self._lock:
            # Now safe to write to results, as no other thread can modify the dict
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Reading the results dictionary should also be protected for consistency
        with self._lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == '__main__':
    queue = JobQueue()

    def simple_processor(data):
        thread_id = threading.get_ident()
        print(f"Job {data} processing by thread {thread_id}")
        return f"Finished job {data}"

    # Add initial jobs
    queue.add_job(1, "Task A")
    queue.add_job(2, "Task B")

    # Start processing jobs in different threads
    thread1 = threading.Thread(target=queue.process_job, args=(1, simple_processor))
    thread2 = threading.Thread(target=queue.process_job, args=(2, simple_processor))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    print("\n--- Final Results ---")
    print(f"Result for Job 1: {queue.get_result(1)}")