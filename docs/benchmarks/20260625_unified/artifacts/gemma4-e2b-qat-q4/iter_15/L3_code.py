class JobQueue:
    def __init__(self):
        # Stores jobs as (priority, insertion_index, name, metadata)
        self._jobs = []
        self._counter = 0

    def add_job(self, name: str, metadata: dict, priority: int = 0):
        """
        Adds a job to the queue with an associated priority level.
        Higher priority numbers indicate higher importance.
        """
        self._jobs.append((priority, self._counter, name, metadata))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Retrieves the highest priority job.

        Jobs are sorted primarily by priority (descending), 
        and secondarily by insertion order (FIFO).

        Returns: (job_name: str, job_metadata: dict) or None if queue is empty.
        """
        if not self._jobs:
            return None

        # Sort criteria:
        # 1. Highest priority first (using negative priority for ascending sort)
        # 2. FIFO for ties (using insertion_index)
        self._jobs.sort(key=lambda x: (-x[0], x[1]))

        # Pop the highest priority job
        priority, index, name, metadata = self._jobs.pop(0)

        return name, metadata