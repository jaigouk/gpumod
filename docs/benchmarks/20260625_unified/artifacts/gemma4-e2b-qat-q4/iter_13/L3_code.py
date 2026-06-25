class JobQueue:
    def __init__(self):
        # Dictionary mapping priority level (int) to a list (queue) of jobs.
        # Jobs stored as tuples: (job_name, metadata)
        self.jobs = {0: [], 1: [], 2: []}

    def add_job(self, job_name: str, metadata: dict = None, priority: int = 0):
        """
        Adds a job to the queue with the specified priority.
        Priority levels: 2 (Critical) > 1 (High) > 0 (Normal)
        """
        priority = int(priority)
        if priority not in self.jobs:
            # Ensure all three levels exist, although 0, 1, 2 are expected inputs
            self.jobs[priority] = []

        job_data = (job_name, metadata if metadata is not None else {})
        self.jobs[priority].append(job_data)

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Retrieves the highest priority job. 
        Priorities are checked in descending order (2, 1, 0).
        """
        # Iterate through priorities from highest (2) to lowest (0)
        for p in sorted(self.jobs.keys(), reverse=True):
            if self.jobs[p]:
                # Retrieve the job from the front of the queue (FIFO)
                job_name, metadata = self.jobs[p].pop(0)
                return job_name, metadata

        # If no jobs are found in any queue
        return None

if __name__ == "__main__":
    queue = JobQueue()

    # Normal job
    queue.add_job("normal_task_a", {"type": "normal"}, priority=0)
    # Critical job
    queue.add_job("critical_task_x", {"type": "critical"}, priority=2)
    # High job
    queue.add_job("high_task_b", {"type": "high"}, priority=1)
    # Another normal job to test FIFO
    queue.add_job("normal_task_c", {"type": "normal"}, priority=0)
    # Another high job
    queue.add_job("high_task_d", {"type": "high"}, priority=1)

    print("--- Testing Priority Ordering ---")
    job = queue.get_next_job()  # Expected: critical_task_x (2)
    print(f"Next job: {job}")

    job = queue.get_next_job()  # Expected: high_task_b (1) (First high job added)
    print(f"Next job: {job}")

    job = queue.get_next_job()  # Expected: high_task_d (1) (FIFO for priority 1)
    print(f"Next job: {job}")

    job = queue.get_next_job()  # Expected: normal_task_a (0) (First normal job added)
    print(f"Next job: {job}")

    job = queue.get_next_job()  # Expected: normal_task_c (0) (FIFO for priority 0)
    print(f"Next job: {job}")

    print("Next job (should be None):", queue.get_next_job())