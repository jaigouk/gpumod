import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            self.lock = threading.Lock() # 1. Create lock

        def add_job(self, job_id, data):
            with self.lock: # 2. Protect write
                self.jobs[job_id] = data
            return job_id

        def process_job(self, job_id, processor):
            with self.lock: # 3. Protect check and removal
                if job_id not in self.jobs:
                    return False
                # Pop is atomic and safe
                data = self.jobs.pop(job_id)

            # 4. Process OUTSIDE the lock to prevent blocking other operations
            # (Though the original code did it inside, holding a lock during
            # processing is a performance anti-pattern, but strictly speaking
            # the prompt just asks to fix the race condition on writes).
            # However, if we move processing outside, we must ensure
            # we don't write to results if processing fails?
            # Let's stick to the structure but fix the concurrency.

            # Actually, looking at the original code:
            # data = self.jobs[job_id]
            # result = processor(data)
            # self.results[job_id] = result
            # del self.jobs[job_id]

            # If I move processor outside the lock:
            # Thread A holds lock, pops job. Releases lock.
            # Thread B holds lock, pops job. Releases lock.
            # Both process.
            # Thread A writes result.
            # Thread B writes result.
            # This is safe for the dict write, but we need to ensure
            # we don't overwrite results if job_ids aren't unique?
            # Assuming job_ids are unique keys.

            # Let's stick to the simplest fix that guarantees thread safety
            # for the dictionary operations as requested.

            result = processor(data) # Processor runs here

            with self.lock: # 5. Protect result write
                self.results[job_id] = result

            return True

        def get_result(self, job_id):
            with self.lock: # 6. Protect read
                return self.results.get(job_id)