import time

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.retries = {}

        def add_job(self, job_id: str, data: dict):
            self.jobs[job_id] = data
            self.retries[job_id] = 0

        def process_job(self, job_id: str, processor: callable) -> bool:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            max_retries = 3
            current_retry = 0

            while current_retry <= max_retries:
                try:
                    processor(data)
                    return True
                except Exception as e:
                    current_retry += 1
                    self.retries[job_id] = current_retry
                    
                    if current_retry <= max_retries:
                        # Exponential backoff: 1s, 2s, 4s
                        delay = 2 ** (current_retry - 1)
                        self.retries[job_id] = {
                            "count": current_retry,
                            "delay": delay
                        }
                        # Simulate sleep
                        # time.sleep(delay) 
                    else:
                        return False
            return False