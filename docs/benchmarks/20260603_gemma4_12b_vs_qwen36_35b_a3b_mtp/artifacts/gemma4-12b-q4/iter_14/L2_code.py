from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Dict[str, Any]] = {}

        def add_job(self, job_id: str, data: Dict[str, Any]):
            self.jobs[job_id] = {
                "data": data,
                "retries": 0
            }

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            job_info = self.jobs[job_id]
            max_retries = 3

            for attempt in range(max_retries + 1):
                try:
                    processor(job_info["data"])
                    # Success
                    return True
                except Exception as e:
                    # Log/handle exception if needed, but requirement is just retry
                    job_info["retries"] += 1
                    if job_info["retries"] > max_retries:
                        break
                    # Backoff logic: 1, 2, 4 (2^0, 2^1, 2^2)
                    # Since we start at attempt 0 (the first try),
                    # if it fails, we move to attempt 1.
                    # Delay for 1st failure (to try 2nd): 2^0 = 1
                    # Delay for 2nd failure (to try 3rd): 2^1 = 2
                    # Delay for 3rd failure (to try 4th): 2^2 = 4
                    delay = 2 ** (attempt)
                    # "Backoff delays can be stored/tracked rather than actually sleeping"
                    # I'll just note the delay or simulate it.
                    pass

            return False