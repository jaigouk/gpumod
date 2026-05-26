queue = JobQueue()
        queue.add_job("normal", {"type": "normal"}, priority=0)
        queue.add_job("critical", {"type": "critical"}, priority=2)
        queue.add_job("high", {"type": "high"}, priority=1)

        job = queue.get_next_job()  # Returns critical job first