import collections

class JobQueue:
    def __init__(self):
        # Index 0: normal, 1: high, 2: critical
        self._deques = [collections.deque() for _ in range(3)]

    def add_job(self, job_id, job_data, priority=0):
        """Add a job to the queue.

        Args:
            job_id (str): Identifier for the job.
            job_data (dict): Data associated with the job.
            priority (int, optional): Priority level (0=normal, 1=high, 2=critical).
        """
        if priority not in (0, 1, 2):
            raise ValueError("Priority must be 0, 1, or 2")
        self._deques[priority].append((job_id, job_data))

    def get_next_job(self):
        """Return the highest‑priority job available.

        Returns:
            tuple[str, dict] | None: (job_id, job_data) of the next job, or None if empty.
        """
        for p in range(2, -1, -1):
            if self._deques[p]:
                job_id, job_data = self._deques[p].popleft()
                return (job_id, job_data)
        return None