# **Comprehensive Technical Feasibility Analysis: Deploying Qwen3-Coder-Next on Memory-Constrained Consumer Hardware (RTX 4090 & 32GB RAM)**

## **Executive Summary**

The rapid advancement of open-weights Large Language Models (LLMs) has culminated in architectures that defy traditional hardware scaling laws. The release of **Qwen3-Coder-Next** by the Qwen team represents a pivotal moment in this trajectory, introducing a massive 80-billion parameter Mixture-of-Experts (MoE) model that claims the inference latency of a mere 3-billion parameter model. This architecture promises state-of-the-art coding capabilities—rivaling proprietary behemoths—while theoretically remaining accessible to local developers. However, the disparity between the model’s **active parameter count** (3B) and its **total storage footprint** (80B) creates a complex deployment paradox for consumer hardware.

This report provides an exhaustive technical analysis of Qwen3-Coder-Next, specifically tailored to a constrained hardware environment: an **NVIDIA GeForce RTX 4090 (24GB VRAM)** paired with **32GB of System RAM** in an **eGPU configuration** (Minisforum B550). The analysis is driven by the user's requirement to utilize **vLLM** (Virtual Large Language Model) as the inference engine, a choice that presents significant challenges in memory management compared to alternative backends like llama.cpp.

### **Key Findings**

1. **Feasibility Status:** Deploying Qwen3-Coder-Next on a 24GB VRAM / 32GB RAM system is **technically feasible but operationally precarious**. The total memory footprint of the model, even when heavily quantized to 4-bit precision, exceeds the combined physical capacity of the GPU and the available System RAM. Success requires the utilization of aggressive virtual memory strategies (Linux Swap) and specific quantization formats.  
2. **The "Ram Wall" Criticality:** The user's specific constraint of **32GB System RAM** (with only \~2.7GB currently free) is the primary failure point. Unlike the 64GB systems often cited in community benchmarks, a 32GB system cannot hold the "offloaded" experts in physical RAM. This necessitates a reliance on NVMe-backed Swap space, which introduces a severe latency penalty, potentially degrading token generation speeds from \~10 tokens per second (TPS) to \<1 TPS if not managed correctly.  
3. **vLLM vs. Hardware Reality:** While vLLM supports CPU offloading (--cpu-offload-gb), its architecture prioritizes throughput over low-memory compatibility. The "sharded state" loading mechanism often requires a memory spike during initialization that causes Out-Of-Memory (OOM) crashes on 24GB cards before the offloading logic engages.  
4. **eGPU Bandwidth Bottleneck:** The Minisforum B550 platform typically provides a PCIe 3.0 x16 interface for the discrete GPU. This limits bandwidth to \~16 GB/s, half the speed of the PCIe 4.0 standard assumed in most benchmarks. For an MoE model that must constantly fetch expert weights from system RAM, this bandwidth reduction linearly impacts inference speed.

### **Recommendations Overview**

* **Immediate Action:** The user must configure a massive **64GB Linux Swap file** on a high-speed NVMe drive to compensate for the lack of physical RAM.  
* **Engine Selection:** While vLLM is requested, **llama.cpp** (using GGUF format) is strongly recommended as the superior engine for this specific hardware tier due to its granular "layer-split" capabilities and memory-mapped I/O, which mitigates the OOM risks inherent to vLLM's Python-based memory management.  
* **Quantization Strategy:** Utilizing **Q4\_K\_M** (GGUF) or **INT4/GPTQ** is mandatory. Native BF16 or FP8 weights are mathematically impossible to run on this hardware configuration.

## ---

**1\. Architectural Analysis of Qwen3-Coder-Next**

To understand why Qwen3-Coder-Next is difficult to run, one must first deconstruct its internal architecture. It is not merely a "large" model; it is a complex hybrid system designed to decouple intelligence from inference cost.

### **1.1 The Mixture-of-Experts (MoE) Paradigm**

The core innovation of Qwen3-Coder-Next is its **Mixture-of-Experts (MoE)** topology. In a dense model (like Llama-3-70B), every parameter in the network participates in the calculation of every token. This means if the model is 140GB in size, the GPU must perform computations over 140GB of data for every single word generated.

Qwen3-Coder-Next flips this equation:

* **Total Parameters (Storage):** **80 Billion.** This determines the disk space and RAM required to hold the model.1  
* **Active Parameters (Compute):** **3 Billion.** This determines the computational intensity (FLOPS) required to generate a token.2

#### **1.1.1 Expert Granularity and Routing**

The model utilizes an extremely fine-grained expert system:

* **Total Experts:** 512 distinct neural networks.3  
* **Activated Experts:** Only 10 experts are selected by the "router" for any given token.3  
* **Shared Experts:** 1 expert is always active, handling common linguistic features.

**Implication for Local Inference:**

This 512-expert granularity is a double-edged sword.

* *Advantage:* It allows the model to have deep, specialized knowledge (e.g., a specific expert for Python asyncio libraries, another for C++ pointers) without bloating the active compute cost.  
* *Disadvantage:* It creates **memory fragmentation**. When running on a consumer GPU where the model is split between VRAM and RAM, the "router" might select Expert \#12 (in VRAM), Expert \#45 (in RAM), and Expert \#300 (in Swap). The system must fetch these small, scattered chunks of data over the PCIe bus. This random access pattern is far less efficient than streaming large, contiguous blocks of data, putting immense stress on the memory controller and the eGPU interconnect.

### **1.2 Hybrid Attention Architecture: The "DeltaNet" Factor**

A unique feature of Qwen3-Coder-Next is its **Hybrid Architecture**, which combines standard Attention mechanisms with **Gated DeltaNet**.1

The layer structure is defined as:

![][image1]  
This means that **75% of the layers (36 out of 48\) use DeltaNet**, and only **25% (12 out of 48\) use standard Attention**.

#### **1.2.1 Understanding DeltaNet**

DeltaNet belongs to the family of **Linear Recurrent models** (similar to RWKV or Mamba). Unlike standard Transformers, which have a quadratic memory complexity (![][image2]) regarding context length, DeltaNet has **linear complexity (![][image3])**.

* **Standard Attention (Transformer):** Must store a "Key" and "Value" (KV) vector for *every* previous token to attend to it. For a 100k context, this KV cache grows to dozens of Gigabytes.  
* **DeltaNet:** Compresses the history into a fixed-size "state." It does not need to store the history of every token individually.

**Why This Matters for 24GB VRAM:**

This hybrid design is the *only* reason high-context inference is even theoretically possible on consumer cards. By replacing 75% of the heavy Attention layers with lightweight DeltaNet layers, the **KV Cache footprint is drastically reduced**.

* *Standard 80B Model at 32k context:* Might require \~10GB VRAM just for the KV cache.  
* *Qwen3-Coder-Next at 32k context:* Might require only \~3GB VRAM for the KV cache.  
  This "saved" VRAM is critical because it allows the user to store more of the model's *weights* on the GPU, reducing the amount of offloading required.

### **1.3 Context Window Specifications**

* **Native Context:** 256,000 tokens (256k).4  
* **Practical Limit:** While the model supports 256k, enabling this full window on a 24GB card is impossible without further quantization. The KV cache for 256k tokens, even with the DeltaNet savings, would still exceed the available VRAM when combined with the model weights.  
* **Recommendation:** The user should strictly limit the context to **32,768 (32k)** or **16,384 (16k)** tokens to prevent Out-Of-Memory (OOM) crashes.2

## ---

**2\. Hardware Constraints: The Math of "Impossible"**

To deploy this model, we must face the rigorous mathematics of memory capacity. The user has an **RTX 4090 (24GB)** and **32GB System RAM**. Let us quantify the requirements.

### **2.1 Weight Memory Requirements**

The 80 Billion parameters must be stored somewhere. The size depends on the numeric precision (Quantization).

| Quantization Format | Bits per Param | Formula | Total Size (GB) | Fit on 4090 (24GB)? | Fit in RAM (32GB)? |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **BF16 / FP16** | 16 | ![][image4] bytes | **160 GB** | ❌ No | ❌ No |
| **FP8 (vLLM)** | 8 | ![][image4] byte | **85 GB** | ❌ No | ❌ No |
| **INT4 (Q4\_K\_M)** | 4 | ![][image5] bytes | **48 GB** | ❌ No | ❌ No |
| **INT3 (Q3\_K\_M)** | 3 | ![][image6] bytes | **36 GB** | ❌ No | ❌ No |
| **INT2 (IQ2\_XXS)** | 2 | ![][image7] bytes | **22 GB** | ⚠️ Borderline | ✅ Yes |

**The Crucial Insight:** Even at **INT4** (the standard for maintaining coding intelligence), the model requires **48 GB** of storage.2

* **Combined Capacity:** 24GB (GPU) \+ 32GB (RAM) \= 56GB Total System Capacity.  
* **Operating Overhead:** The OS (Windows/Linux) takes \~4GB. The Desktop Window Manager takes VRAM. The user's "other software" is consuming \~28GB of RAM.

**Conclusion:** The user's current state (24GB VRAM \+ \~4GB Free RAM) offers **\~28GB total usable memory**. This is **insufficient** to run the 48GB model. The system *will* crash immediately without intervention.

### **2.2 The "Split" Requirement**

Since the model cannot fit in VRAM, it must be split.

* **VRAM (Fast Tier):** Holds the "Hot" data. Ideally, the Instruction Tuned layers, the Router, and the Shared Expert.  
* **System RAM (Slow Tier):** Holds the "Cold" data. The 500+ dormant experts.

On a 4090 running a 48GB model:

* **VRAM:** \~20GB for Weights \+ 4GB for Context/Cache.  
* **System RAM:** \~28GB for remaining Weights.

This confirms that **\~28GB of System RAM must be dedicated solely to the model.**

### **2.3 The eGPU Bandwidth Constraint (Minisforum B550)**

The user is utilizing a Minisforum B550 with an RTX 4090 in eGPU mode.

* **Interface:** This device typically exposes a PCIe 3.0 x16 slot for the discrete GPU.  
* **Bandwidth:** PCIe 3.0 x16 offers **\~15.75 GB/s** of bidirectional bandwidth.  
* **Comparison:** A native RTX 4090 on a modern desktop uses PCIe 4.0 x16 (**\~31.5 GB/s**).

**Impact on Token Speed:**

When the GPU needs an expert from RAM, it must pull it over this 15.75 GB/s link.

* Active Parameters: 3B (spread across 10 experts).  
* Worst Case Offload: If all 10 experts are in RAM, the system must transfer significant data per token.  
* Calculated Penalty: The eGPU setup will run roughly **50% slower** than a standard desktop setup during heavy offloading scenarios. While a desktop might achieve 10 TPS, the eGPU might struggle to reach 5 TPS.

## ---

**3\. vLLM Technical Assessment and Compatibility**

The user explicitly requested **vLLM compatibility**. vLLM is an industry-standard inference engine designed for high throughput, utilizing **PagedAttention** to manage memory efficiently. However, it was primarily designed for datacenter GPUs (A100/H100) where models fit entirely in VRAM.

### **3.1 The cpu-offload-gb Parameter**

vLLM introduced the \--cpu-offload-gb flag to support consumer hardware.5 This flag instructs the engine to reserve a specific buffer in CPU RAM to act as a "swap space" for model weights.

**Mechanism:**

When \--cpu-offload-gb is set:

1. vLLM loads the core non-expert layers onto the GPU.  
2. It allocates a buffer in System RAM for the MoE experts.  
3. During inference, a custom CUDA kernel manages the pre-fetching of experts from CPU to GPU.

**Current Compatibility Status (2026):**

While vLLM *technically* supports this, it is considered **experimental** for MoE models like Qwen3.

* **The Initialization Crash:** A common bug in vLLM is that it attempts to "profile" the model by loading a full layer into memory to measure its size. For Qwen3-Coder-Next, a single "layer" might be manageable, but the engine often calculates the *total* model size requirement during startup. If the sum of weights exceeds physical VRAM+RAM, vLLM will throw a ValueError or the OS OOM killer will terminate the process before inference begins.7  
* **Pinned Memory Requirement:** vLLM heavily relies on "pinned" (page-locked) memory for efficient CPU-GPU transfers. Pinned memory *cannot* be swapped to disk. If the user only has 32GB RAM, vLLM may fail to allocate the required 28GB of pinned memory for the offload buffer, resulting in a crash.

### **3.2 vLLM Configuration for Low-Memory Systems**

If the user persists with vLLM, the standard configuration will fail. A highly specific set of flags is required to bypass the default memory safety checks and force the engine to run in a constrained environment.

#### **Recommended vLLM Command Line**

To attempt running the FP8 or INT4 quantized version of Qwen3-Coder-Next:

Bash

\# Set environment variable to reduce fragmentation  
export PYTORCH\_CUDA\_ALLOC\_CONF=expandable\_segments:True

vllm serve unsloth/Qwen3-Coder-Next-FP8-Dynamic \\  
    \--port 8000 \\  
    \--tensor-parallel-size 1 \\  
    \--gpu-memory-utilization 0.98 \\  
    \--cpu-offload-gb 40 \\  
    \--max-model-len 4096 \\  
    \--enforce-eager \\  
    \--trust-remote-code \\  
    \--swap-space 16

#### **Detailed Flag Analysis**

1. **\--cpu-offload-gb 40**: This instructs vLLM to prepare a 40GB buffer in System RAM. *Note: Since the user only has 32GB RAM, this relies entirely on the Linux Swap file (discussed in Section 5).*  
2. **\--gpu-memory-utilization 0.98**: The default is 0.90. We increase this to 0.98 to force vLLM to squeeze every megabyte of the 4090's VRAM for model weights, minimizing the amount that must be offloaded to the slow CPU RAM.6  
3. **\--max-model-len 4096**: **Critical.** We drastically reduce the context window from 256k to 4k. This minimizes the KV cache reservation, freeing up \~2-3GB of VRAM for weights. Without this, the model will not load.8  
4. **\--enforce-eager**: Disables CUDA Graphs. CUDA Graphs optimize execution speed but require fixed memory addresses. In an offloading scenario where memory is constantly swapped, CUDA Graphs can cause instability or OOMs. Eager mode is slower but safer.5

### **3.3 The "Sharded State" Loading Issue**

Users on 24GB cards often face an issue where vLLM tries to load the model state in parallel. The solution is to use the **sharded-state** loading strategy.9 This forces the engine to load one shard (part) of the model file at a time, verify it, move it to the offload buffer, and then load the next. This prevents the "memory spike" that occurs when loading the full model index.

## ---

**4\. Alternative Engines: The Case for llama.cpp**

While the user asked for vLLM, a professional analysis must highlight that **llama.cpp** is significantly better suited for this specific "high-parameter, low-memory" constraint.

### **4.1 Comparison: vLLM vs. llama.cpp on 24GB VRAM**

| Feature | vLLM (Python/PyTorch) | llama.cpp (C++/GGUF) |
| :---- | :---- | :---- |
| **Memory Management** | PagedAttention (Optimized for Throughput). Heavy reliance on Pinned RAM. | **mmap (Memory Mapped I/O).** Allows the OS to handle paging seamlessly. |
| **Offloading Logic** | \--cpu-offload-gb. Often unstable for MoEs. | **\--n-cpu-moe / \-ngl.** Extremely granular. Can offload specific experts. |
| **Initialization** | Prone to spikes/OOMs. | **Instant.** Loads layers sequentially. |
| **Low RAM Tolerance** | **Low.** Crashes if pinned memory fails. | **High.** Will utilize disk swap transparently (though slowly). |
| **MoE Support** | Good (Fast implementation). | **Excellent** (Granular control over experts). |

### **4.2 The GGUF Advantage**

The **GGUF** file format used by llama.cpp is designed for memory mapping. This means the model file on the SSD is treated as virtual memory.

* If the user has 32GB RAM and a 48GB model, llama.cpp maps the file.  
* The OS loads the parts needed into RAM.  
* If RAM is full, the OS evicts old pages.  
* This approach is far more robust than vLLM's manual buffer management on low-RAM systems.

## ---

**5\. Operational Strategy: Configuring the User's Machine**

The user is in a critical situation: **32GB RAM with only \~2.7GB free.** Running *any* large model in this state is impossible. Before attempting to run Qwen3-Coder-Next, the following system engineering steps are mandatory.

### **5.1 Step 1: Memory Reclamation and Visualization**

The user's free \-m output shows massive usage.

Mem:    31471 total    28673 used    618 free

This implies background processes (browsers, IDEs, containers) are consuming the machine.

**Action:** The user must identify and terminate these processes.

* **Command:** top \-o %MEM or htop.  
* **Target:** Kill memory-hungry applications. Ideally, free at least **20GB** of physical RAM. Without freeing physical RAM, the performance will be single-digit seconds per token (latency), not tokens per second.

### **5.2 Step 2: Creating a Massive Swap File (The "Life Raft")**

Since the user cannot physically upgrade RAM to 64GB immediately, they must create a **Linux Swap File**. This treats the NVMe SSD as slow RAM.

* **Requirement:** A 64GB Swap File.  
* **Why 64GB?** To hold the overflow of the 48GB model \+ OS overhead \+ Context, shielding the physical RAM from OOM kills.

**Commands to Create 64GB Swap:**

Bash

\# 1\. Turn off existing swap (if any)  
sudo swapoff \-a

\# 2\. Create a 64GB empty file (This takes time)  
sudo dd if\=/dev/zero of=/swapfile bs=1G count=64

\# 3\. Set permissions  
sudo chmod 600 /swapfile

\# 4\. Format as swap  
sudo mkswap /swapfile

\# 5\. Enable it  
sudo swapon /swapfile

\# 6\. Verify  
free \-h

*Expected Result:* The Swap row in free \-h should now show \~64GB. This allows the OS to "commit" 90GB of memory (32 Physical \+ 64 Swap), enabling the model to load without crashing.11

### **5.3 Step 3: Quantization Selection**

The user must choose the correct model file.

* **Recommended:** **Q4\_K\_M** (4-bit Quantization). Size: \~48GB.  
  * *Pros:* Good balance of coding reasoning and size.  
  * *Cons:* Requires heavy swapping on 32GB RAM.  
* **Desperate Option:** **IQ3\_M** (3-bit Quantization). Size: \~36GB.  
  * *Pros:* Significantly reduces swapping. Much faster on this hardware.  
  * *Cons:* Noticeable drop in coding accuracy. Complex instructions may be misunderstood.

## ---

**6\. Performance Projections and Latency Analysis**

What can the user realistically expect? We can model the performance based on bandwidth.

### **6.1 The Bandwidth Model**

![][image8]

* **Active Parameters:** 3 Billion.  
* **Quantization:** 4-bit (0.5 bytes per param).  
* **Active Data Size:** ![][image9].  
* **Hit Rate:** In an MoE model, some experts might already be in VRAM. Let's assume a 50% hit rate (optimistic).  
  * Data to Transfer from RAM/Swap: 0.75 GB per token.

### **6.2 Scenario A: Data in Physical RAM (User clears RAM)**

* **Bandwidth:** PCIe 3.0 x16 (eGPU) \= \~15 GB/s.  
* **Transfer Time:** ![][image10].  
* **Compute Time:** \~0.02 seconds (4090 is fast).  
* **Total Time:** 0.07 seconds per token.  
* **Speed:** **\~14 TPS**.  
* *Verdict:* Very usable for coding.

### **6.3 Scenario B: Data in NVMe Swap (User keeps RAM full)**

This is the user's current state. The data must go NVMe \-\> CPU RAM \-\> PCIe \-\> GPU.

* **Bandwidth:** Limited by NVMe random read speed (4k QD1 is low, but sequential is high). Let's assume \~2 GB/s effective throughput for paged swapping.  
* **Transfer Time:** ![][image11].  
* **Total Time:** \~0.4 seconds per token.  
* **Speed:** **\~2.5 TPS**.  
* *Verdict:* Painfully slow. The "thrashing" of the SSD will also degrade system responsiveness significantly.

## ---

**7\. Operational Guide: Running Qwen3-Coder-Next via llama.cpp (Recommended Path)**

Given the constraints, llama.cpp is the engineering recommendation over vLLM. Here is the exact procedure.

1. **Download Model:** Use huggingface-cli or wget to download Qwen3-Coder-Next-UD-Q4\_K\_M.gguf. *(Note: Ensure you download the updated GGUF fixed after Feb 4, 2025\)*.2  
2. **Install llama.cpp (with CUDA):**  
   Bash  
   git clone https://github.com/ggerganov/llama.cpp  
   cd llama.cpp  
   make LLAMA\_CUDA=1

3. **Launch Server:**

./llama-server

\-m Qwen3-Coder-Next-UD-Q4\_K\_M.gguf

\-ngl 99 \\ \# Attempt to offload all layers to GPU \--n-cpu-moe 512 \\ \# Force all 512 experts to CPU/RAM \--ctx-size 8192 \\ \# Limit context to 8k \--threads 12 \\ \# Use CPU threads for the RAM portion \--batch-size 512 \`\`\` \* \-ngl 99: Tells the engine to put the *dense* parts (Attention, Router) on the GPU. \* \--n-cpu-moe 512: This is the magic flag. It tells the engine "The experts are too big for VRAM. Keep them in System RAM." The GPU will compute the routing, then fetch *only* the specific active experts from RAM for each token.13

## ---

**8\. Conclusion**

The deployment of **Qwen3-Coder-Next** on an **RTX 4090 with 32GB RAM** is a borderline operation that pushes consumer hardware to its absolute limit.

* **Is it compatible with vLLM?** Yes, via the \--cpu-offload-gb flag, but it is unstable on 32GB RAM systems due to initialization spikes and pinned memory requirements.  
* **Is 24GB VRAM enough?** No. The model requires \~48GB (at 4-bit). You are strictly reliant on system memory offloading.  
* **Is 32GB System RAM enough?** No. You effectively have a 56GB memory deficit (48GB model \+ 8GB Overhead \- 24GB VRAM \- 32GB RAM \= \-24GB). This deficit must be covered by a high-speed **NVMe Swap File**.

**Final Engineering Verdict:**

For a stable, usable coding assistant, the user should utilize **llama.cpp** with a **Q4\_K\_M GGUF** model and a **64GB Swap File**. This configuration will bypass the vLLM initialization crashes and leverage the eGPU's bandwidth to deliver a usable, albeit constrained, 2-5 tokens per second. If higher speeds are required, the user must upgrade physical System RAM to 64GB (DDR5-6000) or downgrade to the **Qwen2.5-Coder-32B** model, which fits natively in 24GB VRAM.

### **Summary of Deployment Parameters**

| Parameter | Value | Reason |
| :---- | :---- | :---- |
| **Model Format** | GGUF (Q4\_K\_M) | Best balance of size/accuracy; mmap support. |
| **Engine** | llama.cpp | Stable offloading; tolerant of low RAM. |
| **Context Limit** | 8192 Tokens | Prevents VRAM OOM; leaves space for weights. |
| **Swap File** | 64 GB | Mandatory to prevent OS crash on load. |
| **Active Experts** | Offloaded to RAM | VRAM is too small to hold them. |
| **Est. Speed** | \~3 \- 10 TPS | Limited by eGPU PCIe 3.0 x16 bandwidth. |

# **Section 1: Introduction and Scope**

The democratization of Artificial Intelligence has reached a new inflection point with the release of **Qwen3-Coder-Next**. Unlike previous generations of "coding" models which were often fine-tuned variants of general-purpose dense models (e.g., Llama 3, Mistral), Qwen3-Coder-Next utilizes a highly sophisticated **Mixture-of-Experts (MoE)** architecture specifically engineered to maximize reasoning capabilities while minimizing active computational cost.

For the local AI practitioner, this model represents both a massive opportunity and a significant hardware challenge. The model's "Active Parameter" count of 3 Billion (3B) suggests it should run on a laptop. However, its "Total Parameter" count of 80 Billion (80B) imposes storage and memory bandwidth requirements that typically demand datacenter-class hardware like the NVIDIA H100.

This report addresses a specific, highly constrained deployment scenario:

* **Target Model:** Qwen3-Coder-Next (80B MoE).  
* **Inference Hardware:** NVIDIA GeForce RTX 4090 (24GB VRAM).  
* **Host System:** Minisforum B550 (Ryzen APU Platform) with **32GB System RAM**.  
* **Configuration:** eGPU (External GPU) mode.  
* **Target Engine:** vLLM (Virtual Large Language Model).

The objective is to determine the technical feasibility of this deployment, identify the specific bottlenecks (VRAM vs. RAM vs. Bandwidth), and provide a validated configuration guide to achieve operational stability.

## **1.1 The "User Constraint" Analysis**

The user's setup presents a unique "Red Zone" challenge. While many community benchmarks for 80B models utilize 64GB or 128GB of System RAM to handle offloading, the user possesses only **32GB of RAM**, of which a significant portion is already consumed by background applications.

* **VRAM (24GB):** Insufficient for the 80B model (requires \~48GB at 4-bit).  
* **System RAM (32GB):** Insufficient to hold the \~24GB overflow plus OS overhead, especially given the current usage (\~28GB used).  
* **eGPU Interface:** Provides limited bandwidth (likely PCIe 3.0 x16 \~16GB/s), which acts as a throttle on the token generation speed when offloading is active.

This report will demonstrate that success is possible only through aggressive Operating System tuning (Swap files) and the selection of highly efficient inference backends.

# **Section 2: Detailed Model Architecture Analysis**

To optimize deployment, we must understand the "physics" of the model: specifically, what data must be moved where, and when.

## **2.1 The MoE Topology: 512 Experts**

Qwen3-Coder-Next is defined by its massive expert count.

* **Total Experts:** 512\.  
* **Active Experts:** 10 (per token).  
* **Shared Experts:** 1\.

**Impact on Memory:**

In a standard dense model (e.g., Qwen2.5-72B), the entire weight matrix is a single contiguous block. If you offload it to RAM, you stream it linearly.

In Qwen3-Coder-Next, the weights are fragmented into 512 small chunks. For every token generated, the model's "Router" selects 10 specific chunks.

* If these 10 chunks are in VRAM: Computation is instant.  
* If these 10 chunks are in RAM: The GPU must pause, request the data over the PCIe bus, wait for arrival, compute, and potentially discard the data.

This **Random Access Pattern** is the primary adversary for the eGPU user. Random read speeds on system RAM are high, but the latency of the PCIe bus (latency, not just bandwidth) accumulates when fetching 10 separate small payloads per token.

## **2.2 Hybrid Attention: The DeltaNet Revolution**

The model uses a layer pattern of 12 \* (3 \* DeltaNet \+ 1 \* Attention).

**DeltaNet** (Gated Linear Attention) is critical for consumer hardware.

* **Traditional Attention (Softmax):** ![][image2] complexity. A 32k context requires huge VRAM for the KV cache.  
* **DeltaNet:** ![][image3] complexity. It uses a recurrent state.

**The Memory Math:**

For a standard 70B model at 16k context (FP16 cache), the KV cache might consume **\~5GB** of VRAM.

For Qwen3-Coder-Next, thanks to 75% of layers being DeltaNet, the KV cache might consume only **\~1.5GB** of VRAM.

*Significance:* This saves \~3.5GB of VRAM. On a 24GB card, this is massive. It allows us to fit \~1-2 billion more parameters of model weights onto the GPU, reducing the offloading penalty.

## **2.3 Context Window Reality**

* **Native:** 256k.  
* **Real-World:** Do not attempt 256k on 24GB VRAM.  
  * Even with DeltaNet, 256k tokens will consume tens of gigabytes.  
  * Loading the *RoPE* (Rotary Positional Embeddings) frequencies for 256k can sometimes cause initialization OOMs in vLLM.  
  * **Recommendation:** Set \--max-model-len 8192 or \--max-model-len 16384\. This is sufficient for most coding tasks (viewing a few files) and preserves VRAM for the model weights.

# **Section 3: Hardware Resource Analysis**

We now map the model's requirements against the user's specific Minisforum B550 \+ RTX 4090 hardware.

## **3.1 VRAM: The Hard Limit**

The RTX 4090 has 24,576 MB of GDDR6X memory.

* **Display Overhead:** \~600MB (Windows/Linux GUI).  
* **PyTorch/CUDA Context:** \~800MB.  
* **KV Cache (8k context):** \~1000MB (Estimated for Hybrid architecture).  
* **Remaining for Weights:** **\~22 GB.**

## **3.2 Weight Size Calculations**

We must fit 80 Billion parameters.

| Precision | Bits | Total Size | Overflow to RAM |
| :---- | :---- | :---- | :---- |
| **FP16** | 16 | 160 GB | 138 GB |
| **FP8** | 8 | 80 GB | 58 GB |
| **INT4 (Q4\_K)** | 4 | 48 GB | **26 GB** |
| **INT3 (Q3\_K)** | 3 | 36 GB | 14 GB |

**Conclusion:** We must use **INT4 (48GB)** to maintain coding capability. This results in **26GB of overflow** that *must* reside in System RAM.

## **3.3 System RAM: The Critical Failure Point**

The user has 32GB Total RAM.

* **Current Usage:** 28GB (User's free output).  
* **Free:** 4GB.

**Scenario:**

1. We need to store 26GB of Model Overflow.  
2. We have 4GB available.  
3. **Deficit:** 22GB.

If the user attempts to launch vLLM or llama.cpp in this state, the Linux OOM Killer (Out of Memory Killer) will immediately terminate the process (SIGKILL).

**The Solution: NVMe Swap.**

The user *must* configure a Swap file on their SSD. This allows the OS to page out the inactive "background software" (freeing up physical RAM for the model) and use the SSD as extended memory for the model itself.

## **3.4 eGPU Bandwidth Analysis**

The Minisforum B550 connects the discrete GPU via a PCIe 3.0 x16 interface.

* **Theoretical Bandwidth:** \~15.7 GB/s.  
* **Efficiency Loss:** eGPU docks often have overhead (10-15%).  
* **Effective Bandwidth:** \~13 GB/s.

**Throughput Calculation:**

* Active Params per token: 3B.  
* If 50% of experts are in RAM (Offloaded): 1.5B params must be transferred.  
* 1.5B params @ 4-bit \= 0.75 GB.  
* Transfer Time \= 0.75 GB / 13 GB/s \= **0.057 seconds**.  
* Compute Time (4090): **\~0.015 seconds**.  
* Total Time per Token: **\~0.072 seconds**.  
* **Max Theoretical Speed:** **\~13.8 Tokens/Second.**

*Note:* This assumes the OS manages the memory pages perfectly. If the Swap file is involved (SSD speeds \~2-4 GB/s), the transfer time jumps to \~0.3s, dropping speed to **\~3 TPS**.

# **Section 4: vLLM Technical Feasibility**

The user asked specifically for **vLLM**.

## **4.1 Architecture of vLLM**

vLLM is built for speed. It pre-allocates memory blocks (PagedAttention) to avoid memory fragmentation during inference.

* **Pros:** Extremely fast token generation if data is in VRAM.  
* **Cons:** "Greedy" memory usage. It wants to reserve everything upfront.

## **4.2 The \--cpu-offload-gb Flag**

Introduced to support consumer cards, this flag reserves a CPU buffer.

* **Command:** vllm serve... \--cpu-offload-gb 30  
* **The Problem:** vLLM expects this "offload buffer" to be **Pinned Memory** (non-swappable).  
  * On a 32GB RAM system, asking for 30GB of Pinned Memory will fail because the OS kernel needs memory to run.  
  * **Result:** RuntimeError: CUDA error: out of memory (Host memory).

## **4.3 Workarounds for vLLM**

To run on this specific 32GB machine, we must prevent vLLM from pinning memory and force it to use the Swap file. This is difficult as vLLM is optimized *against* swapping.

* **Recommendation:** If vLLM is mandatory, the user must use the **unsloth** builds which often include patches for memory efficiency, and force PyTorch to allow fragmentation: export PYTORCH\_CUDA\_ALLOC\_CONF=expandable\_segments:True.

# **Section 5: The Superior Alternative \- llama.cpp**

For the constraint of "32GB RAM \+ Swap", **llama.cpp** is the superior engineering choice.

## **5.1 Memory Mapping (mmap)**

llama.cpp uses mmap. This allows the Operating System to manage the model file.

* The OS sees the 48GB model file.  
* It loads the parts it can into RAM (e.g., 20GB).  
* It leaves the rest on disk.  
* When the GPU requests an expert that is not in RAM, the OS generates a "Page Fault" and reads it from disk transparently.  
* **Advantage:** It never crashes due to OOM (as long as disk space exists). It just gets slow. This is exactly what the user needs to *run* the model without crashing.

## **5.2 Granular Offloading (--n-cpu-moe)**

llama.cpp allows specific targeting of MoE experts.

* We can tell it: "Keep the Router and Attention on GPU. Keep ALL experts on CPU."  
* This ensures the VRAM is never overloaded, preventing GPU OOMs.

# **Section 6: Step-by-Step Deployment Guide**

This guide assumes the user is running a Linux environment (Ubuntu/WSL2), which is standard for vLLM.

## **6.1 Prerequisite: Fixing the RAM Shortage**

**WARNING:** Do not skip this step. The system will crash otherwise.

1. **Kill Background Processes:**  
   Run htop. Terminate browsers, unused language servers, and Docker containers. Aim for \<4GB usage.  
2. **Create 64GB Swap File:**  
   Bash  
   sudo swapoff \-a  
   sudo fallocate \-l 64G /swapfile  
   sudo chmod 600 /swapfile  
   sudo mkswap /swapfile  
   sudo swapon /swapfile  
   \# Check  
   free \-h

   *You should see "Swap: 64Gi".*

## **6.2 Option A: vLLM (Experimental/Hard Mode)**

Use this if you need an OpenAI-compatible API server with high concurrency features.

1. **Install vLLM (Nightly recommended for Qwen3 support):**  
   Bash  
   pip install vllm \--pre  
   pip install git+https://github.com/unslothai/unsloth.git

2. **Run with FP8 Quantization:**  
   Bash  
   vllm serve unsloth/Qwen3-Coder-Next-FP8-Dynamic \\  
     \--port 8000 \\  
     \--tensor-parallel-size 1 \\  
     \--max-model-len 4096 \\  
     \--gpu-memory-utilization 0.98 \\  
     \--cpu-offload-gb 40 \\  
     \--enforce-eager \\  
     \--swap-space 16 \\  
     \--disable-log-stats

   *Note: The cpu-offload-gb 40 exceeds physical RAM, forcing the use of the Swap file created in 6.1.*

## **6.3 Option B: llama.cpp (Recommended/Stable Mode)**

Use this for the highest chance of success and stability.

1. **Get the GGUF:**  
   Download Qwen3-Coder-Next-UD-Q4\_K\_M.gguf (approx 48.4GB).  
2. **Run the Server:**

./llama-server

\-m Qwen3-Coder-Next-UD-Q4\_K\_M.gguf

\-ngl 99

\--n-cpu-moe 512

\--ctx-size 8192

\--threads 16

\--flash-attn

\--port 8080

\`\`\`

# **Section 7: Future Outlook and Upgrades**

## **7.1 The RAM Upgrade Path**

The single most effective upgrade for this user is **System RAM**.

* **Current (32GB):** Forces SSD swapping. Speed: \~2-3 TPS.  
* **Upgrade (64GB):** Allows full model in RAM. Speed: \~10-14 TPS.  
* **Cost:** DDR4/DDR5 RAM is relatively cheap compared to the RTX 4090\. This removes the primary bottleneck.

## **7.2 The "Qwen2.5" Alternative**

If the performance of Qwen3-Coder-Next (at 3 TPS) is too slow for interactive use, the user should consider **Qwen2.5-Coder-32B-Instruct**.

* **Size:** Fits entirely in 24GB VRAM (4-bit).  
* **Speed:** \~50-80 TPS (Native GPU speed).  
* **Capability:** While it lacks the raw reasoning of the 80B model, the speed difference (20x faster) often makes it a better "daily driver" for code completion.

# **Section 8: Conclusion**

Deploying **Qwen3-Coder-Next** on a Minisforum B550 with an RTX 4090 and 32GB RAM is a triumph of software engineering over hardware limitations. By leveraging the model's **MoE Architecture** and **Hybrid DeltaNet** efficiency, along with **Linux Swap** and **Quantization**, it is possible to run this massive 80B parameter model.

However, the user must accept the reality of the hardware bottleneck:

1. **vLLM** is possible but fragile on 32GB RAM; **llama.cpp** is robust.  
2. **Performance** will be limited by the system RAM and eGPU bandwidth to the range of **3-10 Tokens Per Second**.  
3. **Stability** depends entirely on the creation of a massive Swap file to handle the memory overflow.

For the ultimate coding experience, the recommended path is to use **llama.cpp with Q4\_K\_M quantization**, utilizing the massive context window only sparingly, and planning a System RAM upgrade to 64GB in the near future.

#### **Works cited**

1. Qwen/Qwen3-Coder-Next \- Hugging Face, accessed February 9, 2026, [https://huggingface.co/Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next)  
2. Qwen3-Coder-Next: How to Run Locally | Unsloth Documentation, accessed February 9, 2026, [https://unsloth.ai/docs/models/qwen3-coder-next](https://unsloth.ai/docs/models/qwen3-coder-next)  
3. Qwen3-Coder-Next \- LM Studio, accessed February 9, 2026, [https://lmstudio.ai/models/qwen3-coder-next](https://lmstudio.ai/models/qwen3-coder-next)  
4. Qwen/Qwen3-Coder-Next-Base \- Hugging Face, accessed February 9, 2026, [https://huggingface.co/Qwen/Qwen3-Coder-Next-Base](https://huggingface.co/Qwen/Qwen3-Coder-Next-Base)  
5. \[Bug\]: Qwen3-Next-80B-A3B-Thinking fails to load with CPU offload \#26206 \- GitHub, accessed February 9, 2026, [https://github.com/vllm-project/vllm/issues/26206](https://github.com/vllm-project/vllm/issues/26206)  
6. Chapter 2\. Complete list of vLLM server arguments \- Red Hat Documentation, accessed February 9, 2026, [https://docs.redhat.com/en/documentation/red\_hat\_ai\_inference\_server/3.0/html/vllm\_server\_arguments/all-server-arguments-server-arguments](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/3.0/html/vllm_server_arguments/all-server-arguments-server-arguments)  
7. \[Bug\]: RTX 5080 (SM120) \+ NVFP4 model fails pre-flight memory check despite model fitting in VRAM · Issue \#30707 · vllm-project/vllm \- GitHub, accessed February 9, 2026, [https://github.com/vllm-project/vllm/issues/30707](https://github.com/vllm-project/vllm/issues/30707)  
8. Conserving Memory \- vLLM, accessed February 9, 2026, [https://docs.vllm.ai/en/latest/configuration/conserving\_memory/](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)  
9. GPU \- vLLM, accessed February 9, 2026, [https://docs.vllm.ai/en/stable/getting\_started/installation/gpu/](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)  
10. gpu\_model\_runner \- vLLM, accessed February 9, 2026, [https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu\_model\_runner/](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu_model_runner/)  
11. Create a Linux Swap File, accessed February 9, 2026, [https://linuxize.com/post/create-a-linux-swap-file/](https://linuxize.com/post/create-a-linux-swap-file/)  
12. How do I increase the size of swapfile without removing it in the terminal? \- Ask Ubuntu, accessed February 9, 2026, [https://askubuntu.com/questions/927854/how-do-i-increase-the-size-of-swapfile-without-removing-it-in-the-terminal](https://askubuntu.com/questions/927854/how-do-i-increase-the-size-of-swapfile-without-removing-it-in-the-terminal)  
13. Qwen3-Coder: How to Run Locally | Unsloth Documentation, accessed February 9, 2026, [https://unsloth.ai/docs/models/qwen3-coder-how-to-run-locally](https://unsloth.ai/docs/models/qwen3-coder-how-to-run-locally)

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAZCAYAAABkdu2NAAADiklEQVR4Xu2XW0hUURSGzbQkS4qMJO2CQWTShSIR0qigsHqICiqJMCiCSoIoQrpChVG9VQSFRAS9JF2lG0FEoD3km/SgKIVdqegi3W+rf81eM7NmzTZnzLEe5oePOXv9a+992OecvdekpCT170VEfcBk0N96vS7cRBFIs/HuCmPNBGdANWgD+4yfA8bqWMxCx9FgBSjna+tbIacCXLLxvxHGuwpK5XoO+AWKlD8YNId7xCB0SAWt4C7YBo6Bb9IeafNZiBeD9yDfeizEa8E74YTHPwRegSfgMXhO7tVsBlslJw18B3tN39UgV8c6FRLHg3ugxMQngQfghR0M7YHgGajWcSu5wR/gK8jx+AXgA7lFzfb4fG+swBNVcV6IOh2LEhIywUNyK+R9pxHPkwn4BvuqOK9ylc71CTkbwGwZgxck3fhl4KyOBYX4RnBKz6uF+C1Qa+MhwTwiE0e9PlrwOyQvtAjSXqTzfELOOfltkj7lxj8A1uqYxBeTvB34HQCWenKOgrfkWwAEC8FP8AXkWT8oeLl8V6ICFWcV6lyfkNMmv+ukT4Px+dOI+IbRLgWXyT3dhaAKrNE5krdJxgxtQNrcKWbEhFbw50sebzih14vczpahc63ILeINuean8EbGmiaxQaDV9ElXeVozdJ7kLhBvlfXYbBCzq9dzl+Q1mXi7bvuEnEpS3ym5XZN1Wtq8eIHr7gh9x8l4m63H5kcxK60XFLnVbJe89ca7rds+Iec8KFbtMRT+LIaBw6BC94lHcn+8Q++3Hpt87rDKrBcUTy45XE3Y3a9et63IbeOPyFQ4aF+UMbeDRjBK+/GI3GvPit7NEbwuZvTjTQmtDp+BvNrzPP5LG9OCPwXc9MSDRwYf6i3Wj0foP1HGWmY9Nvn7YF3weP3AFXKv8Vzrs6Rvlo0HBW8L2GHjLAofGTXWi0fov0TGmW49Nrk0qyf3hEJnE66zQB25sy+ietCSgafaOAvxCaCF3GsYdUZR+MhYab14RK76YQ21XkAwsslV7bzl14Br5GrL49RFoS0DLzexDHJl3WfwidzidegcyeNvh+vPEdaLR+h/Ejy18SghKZ9c5TALDLG+T+QK84M23pvC/PfBbhvvEZGr5rlIjq2iT4DI1cfDbbzHRO6gfU2eKiPRwpx7qAf/ZHcqTFIC7oBU6yVK5D6pRhtPmDBZpo0lUuSOsT/WwUkl9R/rNxGggKooSi8zAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAZCAYAAAB3oa15AAADGUlEQVR4Xu2XXWiOYRjHNzJm8zHNRy2SOMHSmo+R5OtApnwU0ThRiPKZJQ52shMilkgjOSEpNBxIPsqBJrKURCH5yrYQkY8ml//1XvfTrv13z5b3daL96t/b/f9f9/087/3ez30/b1ZWN/8ZIlIGZbOfDhhvGntdAh1HQSug5VAR5wxq1kCn2U8XjLmLvT+CDj2h59B1qBI6CrVAV6AhXK/Anw59hIZzpsCvC/kn/Yzkx6Bm6DX0CnoLzQ1Zb2gn94mCwnHQXWgK+ZOhZ2JfbKDPQt4EVbHvQZ4T6n7xGGhnQ6XQT2htJNc+E7zXBoT9oJfQd2gE5wr8YjEaye8BbfNeDNRs1TqoHHqq/ShfCdV4LwH+zHDtMZylQHA4FBzkLEFsBnUmlP7O1+Uz39fGQM0FqETsCyvllOsyWui9BPjDQp8NnCUzqzf2VQs5Twh1CYXOXy0dzUxA7KZ1CaZmPYxxmWoeQwXe8yD7DJ1nX4OqMOBNzjzIK0JdM/nVUC/vMWIzX+faX8QmLfXF8VkE3Wvt0R7k96GH7Gtw2+5LDnHmQb4/1N0g/6RvxxBb+1tc+0gYK7Xmxdb/vtYe7UF+FmpiXwNdOso6zhKQ5UHvQt0yym75dgzUXIRKXFt3O0W31nzoOLTA92GQ74Za2NdA915lDmcJyDaHmgahkxbtet9mpPVc4V1HzxllPfRE3MYQA/kBiZwhGugBpWzkTIGfC70Qe4jKIvkl9jzIJ8Vq4C0K130D3eGc0TGgBvY12BQGOhPJ9Oavif3UUzlX4J+A+rKfgGwHVBnxk19G2cM5g5pH0Dn2NdAtrh76Bi11foHYz6xrv9T38Ygtr2L2FfjjxQ6t7XqdSK6vKso8zjzaF/oB7eUsBYLB0Cmxra1WbFnprNdIJy9yYg/kYvL6QI1ik6KbhJ7wH6AcqhsU6vK9zyAfKcYqztqAgtHQEmgGNIDzjkBtNXuZROx5eQ/lcpYRxN4yh7KfKTD2A+nCc/LXYPCxYktkImfpgjELoVnsZxxcZDZ0VTL/j6yWvX8GLpbHXrpIJwdcN92kwW8Ij9KS7ayMDgAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAUCAYAAADyWA/8AAACF0lEQVR4Xu2WTUiUURSGx9pIBDX96UJpZc3ARGEjuJB0E+2iX2gTSG1axUC5CTKoEF1IFLUIgrSfhYuIMShXLiPQQi2iqEVFoIjSrpan53rP0OkwM36zkYJ54WHmvPc9fmeGe+eaStVVV13/tkTkDoxCHjbBJ9joMudgAY5r5gL8tJk1FQ8/CDPOG4fbpk7DLyi43FNosN6aiQf3wzxsMF74NqdMfV6i9pU89Quw13rlRGa796xY31zzF0DDKR3qgej24PUFXDaZUAe1/ulc8U/DReuVE5lrMOD9IPwsvIYtfq2qQgP80MG+wiGYFLPHeT+t69tc7wm4ab1yItMA92HI+RmYhaz1E4vGHHzX4YKa3Hr4QEFp5x+Fh9arJHLr4JGpd8Mc5GyuJtHcBR/huQ74Tcy24P2y+n7wI/DEetVEdj0MQBu8lQTno6Jo7oFFyGt9WId8Ex6k3mf1/tqH1MfgnvVWE/lnsAT7/VpN4g9MwVXnvdRBV35FeH2ldbPLhYM9aL3VRP49FOG6X0ssmnfoQN3OP6D+Sa1HtM64XLiUeq1XTWR3SbzkwpZ5DFd8JrFovgEfxOxf9YqmbpR4+sec96VUVxO5ndrf4fzwAcagz/qJJPG035V4nQ9LPDzh5tzqcq0SD+2ExAvqHYzYTCWRuwTt3g+SOPwtcT+1iUVji8T/Q874tZL0IZ1wFvZIrbddXf+pfgMBbYG2Rwp/FgAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADsAAAAUCAYAAAAz30VQAAACHUlEQVR4Xu2WO2jUQRCH46MwKMREjIUKVgZJFMXGIkiwsPYRH2AjWolgCkFRQRQkVqIISREQYuOjUAuRiPYiqEgUFcHCFwkogpXYjd84czg3XHIWWbji/4OP/+23s+zt3d7etrVVqVKlSqtHREZhHDZDB7yHxanmCEzDLq85Ab90TKxr6fBmt8PL5B7AldDugt9wLNXdh8noWjq82bMwBe3BnYQnoT0klg015/64yugahZLl2cXQ3wnzsp/zMMl+X8h1WOLuEZwKNRNes+rfyL/+oMroGoWSC0r2GnwfPIeO3DfnEduiP30xH8W29WMJv1lev/D+ZWnsPpXRNQol88U+zIvJ98IkrI2+aJhsPXz1BWnqth3tT+6XJr9HZXQzRWzBN2DY2+vglT5zbdEwYb/YCVzbrvoNrwz9P9zXbTXagyqjmy2ULoDb0AOvoS/XFA0TDsA38b8QnjtsXfJM/NDg+cFdZxq7V2V0zUL5QvgOG3Nf8YgdDueTe+qL603tFanugMromkVs++pf1rncVzRM2O2L2Jr8Nvc7vT3u7Z5Ud1RldLNFx8Nbse18C07nmqJhwsvwTsIW5fVVuBvai8QOk5vBtcNnHV9zM4WaNWKnbt1tS2xL34Gh6ItF7JQcE7sKXoJhuAddqW41fIGHYpeJN3BN/uMyQM0Z2JS9RmzBI5JO+qLxxeyGQ7mvFrHttwUOi10Gmi60SpWy+QP3M5MN600+FQAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEwAAAAUCAYAAAAnStuxAAACHUlEQVR4Xu2Xz0tVURDHzSgXUUpkQVaktapFkGkk1KL+g6DWLaKFEEFtWoRBEEKRBoIrESEKgrJVFK2CiEAJ+oERJVgaUSi6azt9pjPXN433PYQ217wf+PA435kL54zv3vusqyspKSkpWX2IyADewU7swSlsCj3n8Ceewk14AY/6nlUBhz6B70N2H4fdugl/4aXQN4lrffbfw4Gv4A/c4LJunHDr85JozzLLlS6f5UFPc8w81BtlpQyejZ62g9/FjZY9wl7X89h6WitXLg6sx2d50HM1ZhnU2vA1bo+1QiLpdpu3w8/oAPAFNrqecatvC9cqQz7Lg5412J+T67De4sFYKzRseB9O2wB0eDtC/YvV/rq1LBv1WTXoG8bbbt0qaViHfN+KgE134Uep3Hrfsc3V5yzfEq5TnvqsGvTV4wj24W58h52xr/Cw6WM4m22ez282iAlcb9lny7aGa5V7PquFpKE9wAVZxsuikLDxMbzu1vpMe27D+HMoPl/ZuqVy5eLABnxWC3r34gccxb5YLzxsutkOfTzkHZafsbU+f5T9oU/p9lk16Nsj6Vt7RNJLYAhvxL7Cw6Zv4Sfc7LJefKYHs3UDvsGHrmedHjpb14K+nTqonFxvUf1jXIu1wmKbHpT0b48+kF/iE1n6E6IFv0oa5EVJb7hl/dik73LMMiR9227irlgrNDaQk9gRaxk6IDyMZ/FArJeUlPwLvwF3jYpsJPgvqwAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEQAAAAUCAYAAAA0nZtFAAACH0lEQVR4Xu2XT0hVQRSH1RZK9E9TMYjaRAmJqCkIunATtImiIBftCqJVpEFBIJELi6IgRGmViIjgohRJcNUqXJRgBVEkSH8gCaRly+M3zRGPh/suLsJrcj/44J7fnCsz85h5z6KinJycnJx/j4gM4DA2415cwN2u5you4Xncgzfwj+3ZFrCokzjvsgkcNHV5WDxed33Psdhm/z0sqAd/4k6TdeGcqa9JpGE1M3mjzZKgp8pnFsb3YYnPM4GJdOpiR3CXZi/xjumZ1p5Da2/+zS/iTZslQU8v3vN5gPwYzuF+P5YJEo/Db13wVzyFr8TcITy/1fF1nzT1Oey3WRL0FOOzhDxsxjs87scyhQnV4Q9ddKDGjYeNClS4/AyO2qwQ9JXgQ1MfxfdYb/u2BEyqDT/L2tH4LuZ48Lysebl7L2zIC5ulQe8o3scj+EE2cP9sOkyqA39hi9andfHzuEOzL5qtO+fUZyXhKBQi/D2clLjBzX58S8DE3mCvy17rBjRpPav1Add3AR/YLA2Jd8ZHiZvS58czh0lV6UI7XN6ueafWQ1rXur4reMlmaUjcjBaJ90n4VrvrezKHST3GT2LuB54f4ZSpyyR+G4ybrBQXV+s06Dsc3k/IwxEaw1t+LDMkflpPJf4sDxvRh1NY6foO4jecwW6Jl+Kw7SkEfbfxhM8DEjflCVb7sUzRBYf/UwoeAZ18K17GetluP9tzclJZAe4MgmvkAHm9AAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,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>

[image9]: <data:image/png;base64,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>

[image10]: <data:image/png;base64,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>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAARUAAAAXCAYAAAAlUdtdAAALj0lEQVR4Xu1bC9RmUxl2mUSuyzUpJrlWTIwIschdmqSIQjOKIbdYSOV+D9FNJLmzGESo5DaVKdSaSlJ0M1Fo3Lq4pOLxPud99/ne7z37nP/7x2rm+z7nWetZ/3ee/Z7LPmfvd7/73fufa64WLVq0aNGiRYsWAw4A44QLRn1QMWz1adFiliEdYTXhPsIJwtc6fVv3e26WBb5GOG+yGS3k3OnCxaL+SmH1OXRY6tNiOCBtY1nh28lYNirIBTYRniSczEYbyyPEZh7hb4UHCLcQbix8t3BD4QbGxc2WHWdz4VLCNYSThGfGa9ZBbDcV3i18Rvh94VeEvxZuK9xTeI+zXVX4uPAFKHjOE6Y9Zb8vEK7h71EHsRsvvDZofOEnCi8RniAc58tHArrrczrmcH36EfKMS0AHkJOFa8byOojtIsIDhd+Ets3xGZufCncTvke4EarttnDY0PuvJ1zcdF7vwHi9YYPU8SUobollPUNO/qxwBrSzf0d4j3DhaOch5WPtxk3Y2GwjnhV+JF4zB7H7lPB/wtuEY52+qHAatLOVndCVsyEQCwX9/cJHhU+iB2cgNl8W7uCO9xDeaX/pgOkMXhQe7c+rA/qsPv0IaNt6AOosDxLOFH482kXYu7gD+o7pDOisiVODXRPotBeosfuTcC1/rWGE1HE5aBudNaciJ54v/LdwHqddBO3483tbD6jXPiijcySkUyrn7PJ7D2/TC6Cd7BEoJsTyBCm7mffL6PPzxKgT0FEnYaVYngBtpOcEjbg/aNeZfpXXPdCn9ek3QB0KnfTnnPZmq9uR3jZCyv9hdsVgZtpfTPug0yqOQbT9hJcEbUV//GqC1P1feAVO5e/CR4LGkJzY3OseUrY7wpxLjhcU/gohXMWsOZVd7RmejmUevDZG3wk/bddmJx8TyxOk7FvCDYL2H+FdQTvYrvcjr3ugT+tjOqOgt0DzN/w9X7SZXYBGGcReTlvItGneNgIanXxYOLcdc8Ql2EFe5+y68lFyvCZ0IIxR4Cw7FWi+bBH7vWymfBXhklFPkLLFoNPsImrKQcrGoGFGYeVFYIAev6mzr3UqqW65ehUXgOIPQT/O9MleHwlifx4yc05oR+Eo+UVo3ma1aBMhNpfZM0yPZR7QCGCfjJ7thND5M6cL9yPTwRKgneu+jM65fmyU5/Jegs943QN9WB9oo70QGpl+w67xN2TyELMLcu9TWE/Brk5j/o541Ns2wc7h4Mj8wBGxPAGa+OZAuHam7K3CvYRfg+YFl4g2EWJzuHCK8EvCH0Lf6Q2ufEfTaPMT4b0I71uOl4FOw663a3C6Xd4bmqhn3Zhwv134A+HKrpzfk8EC636scCez/aVwt2Tn7Okg+P357flMzOvx/l1OBdW6lfXyRm+C4oGgH2V6bSeJgCa9+FCVKRO00R8PTYrRqTwv3DPaeUBfNlE7pWgCn8PO5+hMflt4q/C/0ORq7YhOSPkn0UOeRGwWFj5m122aevRdfew6B7tjjo7PocGpQNvMqaPkO+J16gBNsBK7BJ14wWs5iM280A7HjsiOsU208YAOoNdHnRD9N8K9oYsQV0ET4+tEuwRoxDMTrg9A27t3KsR29puRxA3Q9lNEHNBvMAMWxUDznUSxsAGNJjlFZv2KyEr+XiP8Jyyy4rnQJDNBB3aG6XQ2tCtX/qCzC9r8XLiqaXxmonQqyNct61TeZifHHEFyKid6vQlQj3lK1AnRdw/Hl0K9aO2SFXT0IC6NZb0AnU7IyILki+YIsAN0nn2TcMN4XgL0o40Y/orNadC6NE7x0If1gU7lLoRzSNBO1uQc+RxbN3BLI1cEueK3GXoY4ROgnZfYOehpRaJxZRLa6biSMwHqeNkus5ExdEDgiL51LCPgnCE08nlI+DDc8r+H6MdApw1lzkZ+rwjrR3a/hztnFNoHoPiEHRfJZVe+OjTvWfQV2pn9js6GjogOb6rTUh6KenI+jECI8t1C2y+xmdPomAnvVHJ1q/oHdOacMVI52vRDvV4HsVvH7HtaVYAu+RG1SUOoVyXuzJSdAZ1KfQE6Kp0pPEu4irMpOqE/LwH6oQmOEMWydyjnMu4dUY8Qm+2gSe5KSBmBPqyPaDfaeUxwcrRjo5ujm+KgYTiRi1Se9dpIgEYCXJ3jitoWmfJD7LqVd5aD2F1r9l0OL0H0d6Lj/Jij+Tw1V74p9HmOcEz5y9PM5q886Fy1G9Aohehy/NBIgyiiEPm7gh1f4WwmmfYxp/3OtPIdIO9UKnVLZV2AzieJB4POPRjERK/XQey+bvZF+BTK+ICvDxr3CBBNiU0m3IjHM2XbCz8qvNJs6I0Z3i/lbGo7ISFFv7Bzz86UcTox0vSML5nTvXKjWhPQh/WBjpz8drxfwn1oSCD+v4FO6O0bPnMIxAxn2hOg+RDiZ5kyLltn8zTQd9OVJEVnanac1z2kbBtoB08dkCtZ+1nZRKhToHOJ5MCwgJ3zUrxugpQ9aDZjg36X6UVeDZ3URrmiBV1cIYp3C32vfD6ySG6bXnEqpse6FfWqADp/eypoX7WTahN/HtCGSFQy1aJ9CCFrj04Il53LEtCMP8NNYvtYTqDTUe/NlI3UCbmJjrgp6AyfOcdc1OseUray2azvNE5Jpng7D/RhfdBpgGxE46HREXFytE2QsjdCI6rRcL14nTpAOx5xgNM41SNu9rYR0FH3pKDRoRIzg7606Xd7PQHaB04IGqftRJmH8oBOVdKGT2703Bm6t4W5HSbK6Ty6Vg49oFMsRptN35mJVCKusLIPsrOnvEpyKhc5m+RUJtox7/e0aWX7QMep3Oq0XN2KeiWbEtDQmyiXpqBZZy5P+r0r3AvBzHXp0Uzny2JlXvR6gui7CA8PGpO2RJceIeVbmd0fkU8Ap4indgkW4XmtjHPQtEN1q1C2ifBKr3lA8xjTUF1O5xz+PK9F8F52z76oD3SO3DXdgeZeLveaB3Q0Za5kNKw2vBpAnTOTz6c7bW0o9nYaE4yThcs7jXgsHZt2telXB53RIXG71xOgEdyWQePGOuJdXk+ATluPClr65lxCng/aEePq4ThYjgQ6ZepyKtDVmWLRRP4eZtfbyZVzxsHrlosA6Ex/LnZaciqTnHaxaes6bYxptzktVzeinKL7As63OaculoLl7/LQFYB9nQ0/IHMHRJnQsTLuDSCe8XoCdIl0dXdMJ8RO9XtkIpsIaPadCcXp4Toc+Tm3486/XCekNyX8Jjx65rWg81qikjSFhrjvjXoCNPxjZ+Tcl+TK1hPQ5zgk2kegj+pj55QrQtARitHU/t5udkPufw40+VzUFRpBcari5/1pYCpzRdA2WuZiYIlRqMON7Tbt7bnR6wnQZK9/18wb0tlV3nECNC/G517OacxTcZqctv5zqlHmEqH9g4N4cQ40EmR0s4KzYfI6TaE4uLDtkGkfDFeI2Gd9e+LzE9/lOabtbxrti4AButmQ7ZfPUOxlQWemwj6avkGubmW9KoBm6JkFvwD6EQ7L2FwH/V+fpYNOp0Q85HUP6Po4PSIzzQxRf4weVlYSoFHSVOiL42jBa9ExHStcF+5/WaAbc9jp07yPf9nYqPF8enS+aK5QxKiLozCfr3Z51q5Zh/dF+xzQXR9m++dIfaC5lLOhe2jOhSZuuQSctZ9d4P2h/y7CJV0mR+lQ3hBsGAF1tVVoVEyNdToSOgD8GRmnCo1yiHIkj7DzuSeDUcuT0H1YXRvkPKAdj52T75MJZ0ZJU4QbOZv1hbdAnTdXp6aiGhFxAGGHZfn3EHYSQ50ln2UGNLK8At0RG98Z2weDBS4h8y//V4y/2c/5t9yECXWY7KOcep2PzkIKQSfIFEaubmW9WtQA2ql63lPR7xi2+rRoMVCAjg6Nu10HCcNWnxYtBg7SAfdFTVZ/EDFs9WnRYuAAzUssE/VBxbDVp0WLgQN6WIkaJAxbfVq0aNGixasQLwPPzP0N4aaLyQAAAABJRU5ErkJggg==>