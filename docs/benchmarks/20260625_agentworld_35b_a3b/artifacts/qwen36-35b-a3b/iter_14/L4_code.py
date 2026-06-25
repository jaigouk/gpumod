import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Protect the addition of jobs to the queue
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        with self.lock:
            if job_id not in self.jobs:
                return False

            # Atomically retrieve and remove the job from the queue.
            # Using pop() ensures that if multiple threads try to process the same job,
            # only one thread succeeds in retrieving it. The job is removed immediately
            # to prevent re-processing.
            data = self.jobs.pop(job_id)

        # Perform the actual processing outside the lock.
        # This allows other threads to add jobs or process other jobs concurrently
        # while this thread is busy executing the processor function.
        result = processor(data)

        with self.lock:
            # Protect the writing of results
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Protect the reading of results for consistency
        with self.lock:
            return self.results.get(job_id)