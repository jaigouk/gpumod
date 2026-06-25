class JobQueue:
    def __init__(self):
        # Structure: {priority: [ (job_name: str, job_data: dict), ... ]}
        # We use a dictionary where keys are priority levels (0, 1, 2)
        # and values are lists acting as FIFO queues.
        self.jobs = {0: [], 1: [], 2: []}

    def add_job(self, job_name: str, job_data: dict, priority: int = 0):
        """
        Adds a job to the queue based on its priority level.
        Higher priority integers mean higher priority jobs.
        """
        # Ensure priority exists in the dictionary (though initialized above)
        if priority not in self.jobs:
            self.jobs[priority] = []

        self.jobs[priority].append((job_name, job_data))

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Returns the highest priority job. Jobs are processed in order: 2 -> 1 -> 0.
        """
        # Check priorities in descending order of level
        for priority_level in [2, 1, 0]:
            if self.jobs[priority_level]:
                # Pop the job from the front of the queue (FIFO)
                job = self.jobs[priority_level].pop(0)
                job_name, job_data = job
                return (job_name, job_data)

        return None

if __name__ == "__main__":
    queue = JobQueue()

    # Add jobs with varying priorities
    queue.add_job("normal", {"type": "normal"}, priority=0)
    queue.add_job("critical", {"type": "critical"}, priority=2)
    queue.add_job("high", {"type": "high"}, priority=1)
    queue.add_job("normal_2", {"type": "normal_2"}, priority=0)
    queue.add_job("critical_2", {"type": "critical_2"}, priority=2)

    print("--- Starting Scheduling ---")

    # Should return critical (priority 2, FIFO within 2s)
    job1 = queue.get_next_job()
    print(f"Job 1 retrieved: {job1[0]} (P={2})")

    # Should return critical_2 (next critical job)
    job2 = queue.get_next_job()
    print(f"Job 2 retrieved: {job2[0]} (P={2})")

    # Should return high (priority 1)
    job3 = queue.get_next_job()
    print(f"Job 3 retrieved: {job3[0]} (P={1})")

    # Should return normal (priority 0, FIFO within 0s)
    job4 = queue.get_next_job()
    print(f"Job 4 retrieved: {job4[0]} (P={0})")

    # Should return normal_2
    job5 = queue.get_next_job()
    print(f"Job 5 retrieved: {job5[0]} (P={0})")

    # Queue is empty
    job_none = queue.get_next_job()
    print(f"Job 6 retrieved: {job_none}")