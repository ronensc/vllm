# Multi-Tier KV Offloading in vLLM

Based on:
[RFC]: Multi-tier KV offloading via the vLLM offloading connector
<https://github.com/vllm-project/vllm/issues/38260>

## Motivation

To date, vLLM offers native KV offloading to CPU memory but does not support further offloading from CPU memory to other tiers such as storage. Implementations for storage offload should either work directly with storage or implement their own CPU offloading as an additional tier. This document describes a high level design to natively support multi-tier KV offloading in vLLM.

The architecture supports a single primary tier, in most cases CPU DRAM, and multiple secondary tiers.

## Goals

- Allow simple and native integration of the current vLLM CPU offloading  with secondary tiers such as storage, or connection with other vLLM nodes for PD disaggregation settings or P2P communication of KV data.

- Utilize async gpu<->cpu transfers for primary tier and async lookup for secondary tier loads

- Support for HMA models out of the box

- Seamless support for cross TP offloading and transfers

- Simple, clean and performant implementation of PD communication via  the CPU tier

What we don't intend to support

- Direct GPU access (neither GPU-storage or GPU-GPU communication)

- Limited flexibility for variance in block size. While we allow vLLM block size to vary, CPU block size must be constant across all vLLM nodes (and a multiple of the underlying vLLM block size).

## High level Design

The design starting point is the current offloading connector design. In this design the offloading connector uses the vLLM V1 connector API and translates it to a simpler more abstract API for a backend. The multi-tier offloading has several key changes to this framework:

- It designates a single (CPU DRAM) backend as the primary tier but also  allows multiple backends to serve as a secondary tier. However, these backends differ from the current backend definition in that the source for offloading (and target for loading) is the CPU DRAM tier rather than the GPU HBM.

- Introduce a new component -- the **TieringManager** that serves as an orchestrator for the communication with the Primary and Secondary tiers. A more detailed description of this appears in the **TieringManager** section below.

- The GPU-CPU offloading is **managed by the scheduler side** offloading manager yet **executed by the worker side** backend. Namely, operations are scheduled and invoked by the scheduler thread, whereas the actual data migration is executed by the worker thread (as it is today in the offloading connector). In contrast, all **secondary tier operations are both managed and executed by the scheduler side only**.

- No matter what the TP rank of a vLLM is, the KV data in CPU DRAM will be stored in **canonical form**, that of TP rank 1. This is the foundation for supporting cross TP rank KV offloading support. For example, if the secondary tier is storage then all KV values will be stored in a single TP1 canonical form, no matter what the TP rank on the vLLM.

### Proposed Change

In order to facilitate the design changes above we are implementing the changes described below to the current offloading connector.

## The TieringManager

The native KV offloading in vLLM v1 currently supports offloading **from GPU memory** to an external location (like CPU memory). This RFC extends the design to allow offloading **from CPU memory** to additional tiers such as local storage, object storage, and remote nodes (P/D disaggregation).

![Tiering Architecture](https://github.com/user-attachments/assets/98ef7014-2071-4517-8eb4-6bf282f40599)

The `OffloadingConnector` interface is unchanged, it holds a single `OffloadingManager`. The new `TieringManager` implements that interface and orchestrates the tier hierarchy internally.

### Two tier types

**Primary Tier**: a single tier with exclusive access to GPU KV memory. The existing CPU Manager serves as the primary tier.

**Secondary Tier(s)**: one or more tiers with read/write access to the Primary Tier's CPU memory. No direct GPU access. Each secondary tier implements the `SecondaryTierManager` interface.

### SecondaryTierManager Interface

```python
class SecondaryTierManager(ABC):
    def lookup(block_hashes) -> int | None
    def submit_store(job_metadata: JobMetadata) -> None
    def submit_load(job_metadata: JobMetadata) -> None
    def get_finished() -> Iterable[JobResult]
    def touch(block_hashes)

@dataclass
class JobMetadata:
    job_id: int                         # unique job identifier
    block_hashes: list[BlockHash]       # which blocks are being transferred
    spec: CPUMemoryViewLoadStoreSpec    # memory views into CPU tensors + block IDs
```

`spec` is a zero-copy memory view into the primary tier's CPU tensors. For `submit_store` it is read-only (secondary tier reads from CPU); for `submit_load` it is writable (secondary tier writes into CPU).

### CPU Manager changes

Extend the CPU Manager to expose its worker's `cpu_tensors` so the `TieringManager` can pass **zero-copy memory views** to secondary tiers for direct reads and writes.

### Key Design Principles

1. **Always cascade to all tiers**: When a block is confirmed in the primary tier, it is asynchronously pushed to every secondary tier.
2. **Primary tier is the gateway**: Only the primary tier accesses GPU memory. Secondary tiers read/write CPU memory via memory views.
3. **Staged promotion**: Blocks in secondary tiers must be promoted to the primary tier before the GPU can access them. `lookup()` returns `None` while promotion is in progress (scheduler retries).
4. **Non-blocking scheduler methods**: All `SecondaryTierManager` methods run in the Scheduler process. `submit_store()` / `submit_load()` submit async jobs; `get_finished()` polls for completion.
5. **Secondary tiers own their evictions**: Each secondary tier manages its own eviction policy independently.

### Store Flow (Cascade)

```text
GPU → Primary Tier (CPU) → [all] Secondary Tiers
```

When `TieringManager.complete_store()` is called, the KV data is confirmed in CPU memory. The `TieringManager` calls `submit_store()` on every secondary tier to cascade the data asynchronously.

### Load Flow (Promotion)

```text
GPU ← Primary Tier (CPU) ← Secondary Tier
```

When `TieringManager.lookup()` is invoked:

1. Check primary tier first.
2. For remaining blocks, check each secondary tier in order.
3. On a hit, call `submit_load()` to initiate async promotion to the primary tier, then return `None` (retry later).

The `TieringManager` calls `get_finished()` on all secondary tiers each scheduling cycle to finalize completed jobs.

## Canonical CPU layout

The following Diagram depicts the general idea - the CPU memory layout is in canonical form, where each block holds all the KV data relevant to this block in consecutive memory layout. Loading and offloading from and to the secondary tier is always of a single memory buffer and is handled by a scheduler thread (rather than the GPU dedicated workers).

![Canonical CPU Layout](https://github.com/user-attachments/assets/0951d9b1-7eb3-4e89-90f9-cf88ffbec334)

The main changes required to facilitate this are:

- During initialization one of the worker side connectors allocates all CPU memory for the KV cache  and assigns the relevant parts of this memory to each of the other workers. This is done in shared memory between the scheduler and workers.

- Each worker is responsible for copying its relevant slice of the KV cache to the appropriate CPU memory locations.

- We mandate that the layout in CPU memory will be in canonical form. In a TP setting the division between workers is according to the  *heads* count -- each worker gets an equal share of the attention heads. As in the drawing above the, we ask that the heads divide the KV CPU memory into contiguous regions and thus can be mapped easily to various TP ranks. There is no constraint on the GPU memory layout, but we achieve better performance and simpler code if that is the case in the GPU as well. This is explained in RFC  <https://github.com/vllm-project/vllm/issues/27742> and implemented in PR <https://github.com/vllm-project/vllm/pull/27743>.  This layout change enables coalescing copies between GPU and CPU to fewer large contiguous memory copies and thus enables efficient use of the DMA for this purpose (see discussion in the following blog <https://vllm.ai/blog/kv-offloading-connector> ).

## Potential Secondary Tiers

- Storage -- This is the obvious secondary tier. Can include:

    - File System API

    - Object Storage

    - Key-Value Store

        - Can be either shared (remote) or local storage

- PD disaggregation -- The current PD connector ("NIXL connector") in vLLM is GPU to GPU communication. Having an alternative CPU to CPU implementation while introduces more hops and therefore more latency, has several benefits:

    - Quicker offloading on the P node, once moved to local CPU can release GPU memory.

    - Shorter time GPU memory required on the D node. Only need to allocate GPU buffers after KV data arrives on the D node CPU memory

    - Simple and clean cross TP handling. No need for complex grid of GPU to GPU chatter.

    - Potentially faster communication on the P to D leg as we will only need to communicate a single unified buffer (relevant to a TP setting).

- P2P -- a generalization of the PD setting is a general P2P communication between nodes.
