from typing import Callable, Dict, Any, Optional

        class JobQueue:
            def __init__(self):
                self.jobs: Dict[str, Dict[str, Any]] = {}

            def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
                self.jobs[job_id] = {
                    "data": data,
                    "status": "pending",
                    "retry_count": 0,
                    "next_backoff": 0
                }

            def process_job(self, job_id: str, processor: Callable) -> bool:
                if job_id not in self.jobs:
                    return False

                job = self.jobs[job_id]
                max_retries = 3
                backoffs = [1, 2, 4]

                for attempt in range(max_retries + 1):
                    try:
                        processor(job["data"])
                        job["status"] = "completed"
                        job["retry_count"] = attempt
                        return True
                    except Exception:
                        job["retry_count"] += 1
                        if job["retry_count"] <= max_retries:
                            delay_idx = job["retry_count"] - 1
                            job["next_backoff"] = backoffs[delay_idx]
                            # Simulate waiting logic if needed, but requirement says
                            # just track.
                        else:
                            job["status"] = "failed"
                            return False

                # If loop finishes without return (edge case)
                job["status"] = "failed"
                return False