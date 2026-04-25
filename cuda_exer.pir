<pir>
<meta>
name: cuda_exer
root: /home/daivy/projects/cuda_exer
profile: generic
lang: PY
</meta>
<units>
u0: tests/test_block_table.py type=PY role=lib module=tests
u1: tests/test_dynamic_batching.py type=PY role=lib module=tests
u2: tests/__init__.py type=PY role=lib module=tests
u3: tests/test_memory.py type=PY role=lib module=tests
u4: tests/test_bpha.py type=PY role=lib module=tests
u5: src/__init__.py type=PY role=lib module=src
u6: src/dynamicBatching/adaptive_batcher.py type=PY role=lib module=dynamicBatching
u7: src/dynamicBatching/__init__.py type=PY role=lib module=dynamicBatching
u8: src/dynamicBatching/queue_model.py type=PY role=lib module=dynamicBatching
u9: src/blockedTensor/__init__.py type=PY role=lib module=blockedTensor
u10: src/blockedTensor/blocked_tensor.py type=PY role=lib module=blockedTensor
u11: src/blockedTensor/layout.py type=PY role=lib module=blockedTensor
u12: src/bpha/bpha_compute.py type=PY role=lib module=bpha
u13: src/bpha/bpha_operator.py type=PY role=lib module=bpha
u14: src/bpha/__init__.py type=PY role=lib module=bpha
u15: src/memory/allocator.py type=PY role=lib module=memory
u16: src/memory/__init__.py type=PY role=lib module=memory
u17: src/memory/memory_tracker.py type=PY role=lib module=memory
u18: src/pagedAttention/paged_attention.py type=PY role=lib module=pagedAttention
u19: src/pagedAttention/paged_memory.py type=PY role=lib module=pagedAttention
u20: src/pagedAttention/__init__.py type=PY role=lib module=pagedAttention
u21: src/pagedAttention/block_table.py type=PY role=lib module=pagedAttention
u22: examples/__init__.py type=PY role=lib module=examples
u23: examples/example_dynamic_batching.py type=PY role=lib module=examples
u24: examples/example_block_table.py type=PY role=lib module=examples
u25: examples/example_blocked_tensor.py type=PY role=lib module=examples
u26: examples/example_memory_tracking.py type=PY role=lib module=examples
u27: examples/example_paged_attention.py type=PY role=lib module=examples
u28: benchmarks/run_benchmarks.py type=PY role=lib module=benchmarks
u29: benchmarks/bpha_benchmark.py type=PY role=lib module=benchmarks
u30: benchmarks/__init__.py type=PY role=lib module=benchmarks
u31: benchmarks/memory_benchmark.py type=PY role=lib module=benchmarks
u32: benchmarks/batching_benchmark.py type=PY role=lib module=benchmarks
</units>
<dependency-pool>
d0: import:[.adaptive_batcher]
d1: import:[.allocator]
d2: import:[.batching_benchmark]
d3: import:[.block_table]
d4: import:[.blockedTensor]
d5: import:[.blocked_tensor]
d6: import:[.bpha]
d7: import:[.bpha_benchmark]
d8: import:[.bpha_compute]
d9: import:[.bpha_operator]
d10: import:[.dynamicBatching]
d11: import:[.example_block_table]
d12: import:[.example_blocked_tensor]
d13: import:[.example_dynamic_batching]
d14: import:[.example_memory_tracking]
d15: import:[.example_paged_attention]
d16: import:[.layout]
d17: import:[.memory]
d18: import:[.memory_benchmark]
d19: import:[.memory_tracker]
d20: import:[.pagedAttention]
d21: import:[.paged_attention]
d22: import:[.paged_memory]
d23: import:[.queue_model]
d24: import:[batching_benchmark]
d25: import:[blockedTensor]
d26: import:[bpha]
d27: import:[bpha_benchmark]
d28: import:[dynamicBatching]
d29: import:[memory]
d30: import:[memory_benchmark]
d31: import:[numpy]
d32: import:[pagedAttention]
d33: import:[psutil]
d34: import:[stdlib:py]
d35: import:[torch.nn.functional]
d36: import:[torch.nn]
d37: import:[torch]
d38: import:[unittest]
</dependency-pool>
<dependencies>
u0->refs:[d34 d32 d38]
u1->refs:[d34 d38 d28]
u2->refs:[d34]
u3->refs:[d29 d34 d38]
u4->refs:[d31 d34 d38 d26 d37]
u5->refs:[d6 d10 d4 d17 d20]
u6->refs:[d34]
u7->refs:[d23 d0]
u8->refs:[d34]
u9->refs:[d16 d5]
u10->refs:[d31 d34]
u11->refs:[d34]
u12->refs:[d34 d35 d37]
u13->refs:[d36 d34 d35 d37]
u14->refs:[d9 d8]
u15->refs:[d34]
u16->refs:[d19 d1]
u17->refs:[d34 d33]
u18->refs:[d31 d34 d35 d37]
u19->refs:[d31 d34 d3]
u20->refs:[d21 d3 d22]
u21->refs:[d31 d34]
u22->refs:[d12 d13 d11 d15 d14]
u23->refs:[d34 d28]
u24->refs:[d34 d32]
u25->refs:[d31 d34 d25]
u26->refs:[d29 d34]
u27->refs:[d31 d34 d32 d37]
u28->refs:[d30 d34 d24 d27]
u29->refs:[d31 d34 d26 d37 d35]
u30->refs:[d2 d7 d18]
u31->refs:[d31 d34 d32]
u32->refs:[d31 d34 d28]
</dependencies>
<symbols>
TestBlockTable:u0 class
TestBlock:u0 class
TestM1M1Queue:u1 class
TestAdaptiveBatcher:u1 class
TestMemoryTracker:u3 class
TestBlockAllocator:u3 class
TestBPHAOperator:u4 class
TestBPHAForward:u4 class
TestBPHAComparison:u4 class
Request:u6 class
BatchDecision:u6 class
AdaptiveBatcher:u6 class
QueueStats:u8 class
M1M1Queue:u8 class
LayoutConstraint:u10 class
BlockedTensor:u10 class
BlockedTensorView:u10 class
ContiguityType:u11 class
AccessPattern:u11 class
TensorLayout:u11 class
bpha_forward:u12 func
bpha_backward:u12 func
compute_memory_efficiency:u12 func
BPHAOperator:u13 class
Allocation:u15 class
BlockAllocator:u15 class
MemoryStats:u17 class
MemoryTracker:u17 class
PagedAttention:u18 class
compare_with_standard_attention:u18 func
PagedMemoryManager:u19 class
Block:u21 class
BlockTable:u21 class
simulate_requests:u23 func
main:u23 func entry=true
main:u24 func entry=true
main:u25 func entry=true
main:u26 func entry=true
create_mock_kv_data:u27 func
main:u27 func entry=true
main:u28 func entry=true
standard_attention:u29 func
test_correctness:u29 func
benchmark_throughput:u29 func
simulate_contiguous_allocation:u31 func
benchmark_fragmentation:u31 func
benchmark_alloc_free:u31 func
test_queue_model:u32 func
verify_theoretical_formulas:u32 func
benchmark_adaptive_batching:u32 func
simulate_burst_traffic:u32 func
</symbols>
<profiles>
  active: ml-python
  ml-python:
    confidence: 0.8
    tags:
      - domain:ml
      - lib:numpy
      - lib:torch
      - runtime:cpython
      - stack:ml-python
</profiles>
</pir>