import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            self.lock = threading.Lock() # Create lock

        def add_job(self, job_id, data):
            with self.lock: # Protect write
                self.jobs[job_id] = data
            return job_id

        def process_job(self, job_id, processor):
            with self.lock: # Protect check, read, write, delete
                if job_id not in self.jobs:
                    return False

                data = self.jobs[job_id]
                # Note: The actual processing (processor(data)) is usually the expensive
                # part. Ideally, we shouldn't hold the lock while processing to allow
                # concurrency. However, the prompt implies a simple queue where the
                # *state update* is the critical section.
                # To be strictly safe regarding the specific bug mentioned (writing results),
                # we can unlock before processing, but we must ensure the job is "removed"
                # from the queue before processing starts so it isn't processed twice.

                del self.jobs[job_id] # Remove immediately to prevent double processing

            # Process outside lock (Best Practice)
            result = processor(data)

            with self.lock: # Protect write to results
                self.results[job_id] = result

            return True

        def get_result(self, job_id):
            with self.lock: # Protect read
                return self.results.get(job_id)