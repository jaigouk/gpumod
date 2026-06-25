class JobQueue:
    def __init__(self):
        # Structure: {priority: [job_name, job_data]}
        # We initialize the structure to handle priorities 0, 1, and 2.
        self.queues = {0: [], 1: [], 2: []}

    def add_job(self, job_name: str, job_data: dict, priority: int = 0):
        """
        Adds a job to the queue based on its priority level.
        Priority 2 is highest, Priority 0 is lowest.
        """
        if priority in self.queues:
            self.queues[priority].append((job_name, job_data))

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Retrieves the highest priority job.
        Jobs within the same priority level maintain FIFO order.
        """
        # Check priorities in descending order (2, 1, 0)
        for p in sorted(self.queues.keys(), reverse=True):
            if self.queues[p]:
                job_name, job_data = self.queues[p].pop(0)
                return (job_name, job_data)
        return None