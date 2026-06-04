class JobQueue:
            def __init__(self):
                self.jobs = {}
                self.retries = {}

            def add_job(self, job_id, data):
                self.jobs[job_id] = data
                self.retries[job_id] = 0

            def process_job(self, job_id, processor):
                data = self.jobs[job_id]
                for attempt in range(4): # 0, 1, 2, 3
                    try:
                        processor(data)
                        return True
                    except Exception:
                        self.retries[job_id] += 1
                        if self.retries[job_id] >= 3:
                            return False
                        # Backoff logic here
                return False