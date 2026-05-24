import threading

        class JobQueue:
            def __init__(self):
                self.jobs = {}
                self.results = {}
                self.lock = threading.Lock() # 1. Create lock

            def add_job(self, job_id, data):
                with self.lock: # 2. Lock access
                    self.jobs[job_id] = data
                return job_id

            def process_job(self, job_id, processor):
                with self.lock: # 3. Lock access
                    if job_id not in self.jobs:
                        return False

                    data = self.jobs[job_id]
                    # 4. Remove job from queue immediately to prevent double processing
                    del self.jobs[job_id]

                # 5. Process outside lock to allow concurrency (optional but good practice)
                # However, if we write result inside lock, we must lock again.
                # Let's stick to the simplest correct fix first.
                
                result = processor(data)

                with self.lock: # 6. Lock to write result
                    self.results[job_id] = result

                return True

            def get_result(self, job_id):
                with self.lock: # 7. Lock to read
                    return self.results.get(job_id)