import logging
from skippy.core.clustercontext import ClusterContext
from skippy.core.priorities import ResourcePriority
from skippy.core.model import Capacity




class MostRequestedPriority(ResourcePriority):
    def scorer(self, context: ClusterContext, requested: Capacity, allocatable: Capacity) -> int:
        cpu_fraction = self.fraction_of_capacity(requested.cpu_millis, allocatable.cpu_millis)
        memory_fraction = self.fraction_of_capacity(requested.memory, allocatable.memory)

        if cpu_fraction >= 1 or memory_fraction >= 1:
            return 0
        
        used_cpu = 1 - (allocatable.cpu_millis - requested.cpu_millis) / allocatable.cpu_millis
        used_mem = 1 - (allocatable.memory - requested.memory) / allocatable.memory

        avg_used = (used_cpu + used_mem) / 2
        score = int(avg_used * context.max_priority)
        return score
    
    @staticmethod
    def fraction_of_capacity(requested: int, capacity: int) -> float:
        if capacity == 0:
            capacity = 1
        return float(requested) / float(capacity)