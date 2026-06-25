from typing import Callable

class JobQueue:
    """JobQueue with a retry capability for processed jobs."""
    # Assume that job data is stored in self._jobs {job_id: data}

    def process_job(self, job_id: str, processor: Callable) -> bool:
        """Process a job with up to 4 attempts, retrying on exception.

        The method calls `processor(data)` where `data` is the dictionary
        stored for the given `job_id`. On any exception it performs
        exponential back‑off (1 s, 2 s, 4 s) but does not actually sleep.
        The computed delays are recorded as keys in the original `data`
        dictionary. Returns True on the first successful execution,
        False if all four attempts fail.
        """
        # Retrieve the data for the job; default to empty dict if not present
        data = self._jobs.get(job_id, {})

        attempt = 1  # number of attempts performed (including current)
        max_attempts = 4

        while attempt <= max_attempts:
            try:
                processor(data)
                return True
            except Exception:
                # No more attempts left?
                if attempt == max_attempts:
                    return False

                # Compute exponential backoff delay (1, 2, 4 seconds)
                # For the second attempt (attempt == 2) the delay is 1 sec,
                # for the third attempt 2 secs, for the fourth attempt 4 secs.
                delay = 2 ** (attempt - 2) if attempt > 1 else 1

                # Record the delay in the data dict as a placeholder
                data.setdefault('#delay', {})[('value',)] = delay
                # Record the current retry count (failed attempts so far)
                data.setdefault('#retry', {})[('count',)] = attempt - 1

                attempt += 1
        # All attempts exhausted without success
        return False