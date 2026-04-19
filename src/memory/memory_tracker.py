"""
Memory Tracker: Tracks and reports memory usage
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field
import time
import psutil


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    timestamp: float
    allocated_bytes: int
    reserved_bytes: int
    peak_bytes: int
    num_allocations: int
    num_frees: int
    fragmentation: float
    utilization: float


class MemoryTracker:
    """
    Tracks memory allocation and usage patterns.

    Provides detailed statistics for memory optimization analysis.
    """

    def __init__(self, name: str = "default"):
        """
        Initialize MemoryTracker.

        Args:
            name: Name for this tracker instance
        """
        self.name = name
        self.allocated_bytes = 0
        self.reserved_bytes = 0
        self.peak_bytes = 0
        self.num_allocations = 0
        self.num_frees = 0
        self.allocation_history: List[Dict] = []
        self.start_time = time.time()

        try:
            self.process = psutil.Process()
        except Exception:
            self.process = None

    def allocate(self, size_bytes: int, tag: str = ""):
        """
        Record an allocation.

        Args:
            size_bytes: Size of allocation in bytes
            tag: Optional tag for tracking
        """
        self.allocated_bytes += size_bytes
        self.num_allocations += 1
        self.peak_bytes = max(self.peak_bytes, self.allocated_bytes)

        self.allocation_history.append({
            'type': 'alloc',
            'size': size_bytes,
            'tag': tag,
            'timestamp': time.time(),
            'total': self.allocated_bytes
        })

    def free(self, size_bytes: int, tag: str = ""):
        """
        Record a deallocation.

        Args:
            size_bytes: Size being freed
            tag: Optional tag for tracking
        """
        self.allocated_bytes = max(0, self.allocated_bytes - size_bytes)
        self.num_frees += 1

        self.allocation_history.append({
            'type': 'free',
            'size': size_bytes,
            'tag': tag,
            'timestamp': time.time(),
            'total': self.allocated_bytes
        })

    def get_current_stats(self) -> MemoryStats:
        """
        Get current memory statistics.

        Returns:
            MemoryStats instance
        """
        fragmentation = self._compute_fragmentation()
        utilization = self._compute_utilization()

        return MemoryStats(
            timestamp=time.time(),
            allocated_bytes=self.allocated_bytes,
            reserved_bytes=self.reserved_bytes,
            peak_bytes=self.peak_bytes,
            num_allocations=self.num_allocations,
            num_frees=self.num_frees,
            fragmentation=fragmentation,
            utilization=utilization
        )

    def _compute_fragmentation(self) -> float:
        """
        Estimate memory fragmentation (0.0 to 1.0).

        Returns:
            Fragmentation ratio
        """
        if self.num_allocations == 0:
            return 0.0

        recent_allocs = [h for h in self.allocation_history[-100:]
                       if h['type'] == 'alloc']
        recent_frees = [h for h in self.allocation_history[-100:]
                       if h['type'] == 'free']

        if not recent_frees:
            return 0.0

        freed = sum(h['size'] for h in recent_frees)
        if freed == 0:
            return 0.0

        holes = abs(sum(h['size'] for h in recent_allocs) - freed) / freed
        return min(1.0, holes)

    def _compute_utilization(self) -> float:
        """
        Compute memory utilization (0.0 to 1.0).

        Returns:
            Utilization ratio
        """
        if self.reserved_bytes == 0:
            return 0.0
        return self.allocated_bytes / self.reserved_bytes

    def get_system_memory(self) -> Dict[str, float]:
        """
        Get system memory information.

        Returns:
            Dictionary with system memory stats
        """
        if self.process is None:
            return {}

        mem_info = self.process.memory_info()
        sys_mem = psutil.virtual_memory()

        return {
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024),
            'system_total_mb': sys_mem.total / (1024 * 1024),
            'system_available_mb': sys_mem.available / (1024 * 1024),
            'system_used_percent': sys_mem.percent
        }

    def reset(self):
        """Reset all statistics."""
        self.allocated_bytes = 0
        self.reserved_bytes = 0
        self.peak_bytes = 0
        self.num_allocations = 0
        self.num_frees = 0
        self.allocation_history.clear()
        self.start_time = time.time()

    def print_summary(self):
        """Print a summary of memory usage."""
        stats = self.get_current_stats()
        sys_mem = self.get_system_memory()

        print(f"\n=== Memory Tracker: {self.name} ===")
        print(f"Allocations: {stats.num_allocations} allocs, {stats.num_frees} frees")
        print(f"Current: {stats.allocated_bytes / (1024*1024):.2f} MB")
        print(f"Peak: {stats.peak_bytes / (1024*1024):.2f} MB")
        print(f"Fragmentation: {stats.fragmentation:.2%}")
        print(f"Utilization: {stats.utilization:.2%}")

        if sys_mem:
            print(f"\nSystem Memory:")
            print(f"  RSS: {sys_mem.get('rss_mb', 0):.2f} MB")
            print(f"  System used: {sys_mem.get('system_used_percent', 0):.1f}%")

    def __repr__(self) -> str:
        return (f"MemoryTracker(name='{self.name}', "
                f"allocated={self.allocated_bytes / (1024*1024):.2f}MB, "
                f"peak={self.peak_bytes / (1024*1024):.2f}MB)")