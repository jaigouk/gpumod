import threading
import time # Added for demonstration purposes

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # 1. Initialize a Lock to protect shared resources
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        # Acquire the lock before modifying self.jobs
        with self._lock:
            self.jobs[job_id] = data
            return job_id
        # Lock is automatically released when exiting the 'with' block

    def process_job(self, job_id, processor):
        # Acquire the lock to ensure atomicity of the entire operation
        with self._lock:
            if job_id not in self.jobs:
                return False

            # Retrieve data safely
            data = self.jobs[job_id]
            
            # IMPORTANT: The processing itself (processor(data)) should ideally happen 
            # outside the lock if it takes a long time, but since the original 
            # code implies the processing happens while holding the state, 
            # we keep it inside for correctness relative to the original structure.
            # If 'processor' is slow, the lock should be released before calling it.
            
            # For this fix, we assume processor is fast or that the queue state 
            # modification must be fully atomic.
            result = processor(data)

            # Safely write the result
            self.results[job_id] = result

            # Safely remove the job
            del self.jobs[job_id]
            return True
        # Lock is automatically released

    def get_result(self, job_id):
        # Acquire the lock even for reading to ensure the dictionary state 
        # is consistent (i.e., we don't read while another thread is deleting/writing)
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage (Demonstration) ---
if __name__ == '__main__':
    queue = JobQueue()
    
    def simple_processor(data):
        # Simulate work
        time.sleep(0.01) 
        return f"Processed: {data}"

    # Add jobs
    job_id_1 = queue.add_job(1, "Task A")
    job_id_2 = queue.add_job(2, "Task B")
    job_id_3 = queue.add_job(3, "Task C")

    def worker(queue, job_id, processor):
        # Simulate a worker trying to process a job
        success = queue.process_job(job_id, processor)
        if success:
            print(f"Worker processed Job {job_id}")
        else:
            print(f"Worker failed to find Job {job_id}")

    # Use threads to simulate concurrent access
    t1 = threading.Thread(target=worker, args=(queue, job_id_1, simple_processor))
    t2 = threading.Thread(target=worker, args=(queue, job_id_3, simple_processor))
    t3 = threading.Thread(target=worker, args=(queue, job_id_2, simple_processor))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print("\n--- Final Results ---")
    print(f"Result 1: {queue.get_result(job_id_1)}")
    print(f"Result 2: {queue.get_result(job_id_2)}")
    print(f"Result 3: {queue.get_result(job_id_3)}")
    print(f"Jobs remaining in queue: {queue.jobs}")