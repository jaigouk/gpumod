queue.add_job("critical", ..., priority=2)
            queue.add_job("normal", ..., priority=0)
            job = queue.get_next_job() # Returns critical job first