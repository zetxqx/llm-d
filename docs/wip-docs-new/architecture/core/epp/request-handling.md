# EPP Request Handling


## Functionality

The EPP Request Handling component manages the lifecycle of an inference request before and after the scheduling phase. It handles parsing the request payload, preparing and managing state for the [Scheduler](scheduling.md), interacting with [Flow Control](flow-control.md), and processing the response from the model server. It is responsible for managing the state of the request throughout its full lifecycle.

## Design

### Architecture Overview

```mermaid
flowchart TD
    %% Custom Style Definitions
    classDef parsing fill:lightskyblue,stroke:dodgerblue,stroke-width:2px,color:black
    classDef flowcontrol fill:plum,stroke:purple,stroke-width:2px,color:black
    classDef statemgmt fill:orange,stroke:darkorange,stroke-width:2px,color:black
    classDef postsched fill:cyan,stroke:darkcyan,stroke-width:2px,color:black
    classDef rejection fill:tomato,stroke:red,stroke-width:2px,color:white
    classDef scheduler fill:lightgreen,stroke:forestgreen,stroke-width:3px,color:black,font-weight:bold

    %% Entry/Exit Points
    Req([Incoming Request])
    Client([Return to Client])
    Reject[Reject Request]:::rejection

    subgraph SystemPipeline ["EPP"]
        direction TB

        %% 1. Request/Response Parsing
        subgraph SubParsing ["RequestHandling Parsing"]
            Parse[Parser.ParseRequest]:::parsing
            ParseResp[Parser.ParseResponse]:::parsing
        end

        %% 2. Flow Control
        FC[FlowControl]:::flowcontrol

        %% 3. Request Handling
        subgraph SubHandling ["RequestHandling Control"]
            direction TB
            subgraph SubPre ["Pre-Schedule Handling"]
                Prep[DataProducer.PrepareRequestData]:::statemgmt
                Admit[Admitter.AdmitRequest]:::statemgmt
            end

            subgraph SubPostSched ["Post-Scheduling Handling"]
                PreReq[PreRequest.PreRequest]:::postsched
                RespHead[ResponseHeaderProcessor.ResponseHeader]:::postsched
                RespBody[ResponseBodyProcessor.ResponseBody]:::postsched
            end
        end

        %% 4. Scheduler
        Sched[Scheduler.Schedule]:::scheduler    
    end
    Forward[Selected endpoints]

    %% Connections
    Req --> Parse
    Parse -->|"InferenceRequest"| FC
    
    %% Rejection Paths
    FC --> Prep
    FC -->|"Rejected/Evicted"| Reject
    
    Prep --> Admit
    Admit -->|"Admit"| Sched
    Admit -->|"Denied"| Reject
    
    %% Success Paths
    Sched -->|"SchedulingResult"| PreReq
    
    PreReq --> Forward
    Forward --> RespHead
    RespHead --> ParseResp
    ParseResp --> RespBody
    RespBody --> Client
```


#### Core Components

*   **Parser**: Responsible for parsing the request and response payloads to InferenceRequest, a structured internal representation of the incoming request.

* **Pre-Scheduling**: 
    * **DataProducer**: A pluggable extension that allows customizing request pre-processing and producing per-request state needed for scheduling, such as tokenization, prefix-cache matches, predicted processing latency etc..
    * **Admitter**: Decides whether to admit a request based on criteria like latency SLOs. Runs after data production but before scheduling.
* **Post-Scheduling**: 
    * **PreRequest**: Executes after `SchedulingResult` is generated but before passing the request back to the proxy to forward it to the model server.
    * **ResponseHeaderProcessor**: Triggered after response headers are successfully received. 
    * **ResponseBodyProcessor**: The primary interface for processing response data. It handles both streaming and non-streaming responses. For streaming responses, it processes each data chunk, with `EndOfStream` (EOS) set to true on the final chunk. For non-streaming responses, it runs exactly once with `EndOfStream` set to true.

---

### Concrete Plugins

#### Parsers
*   **[`openai-parser`](placeholder-link)**: The default parser supporting the OpenAI API. It parses request payloads to extract model name and prompts, and response payloads to extract usage data (tokens). It supports the following endpoints:
    *   `/conversations`
    *   `/responses`
    *   `/chat/completions`
    *   `/completions`
    *   `/embeddings`
*   **[`vllmgrpc-parser`](placeholder-link)**: A parser designed to handle requests specifically for the vLLM gRPC API. It supports:
    *   `Generate`
    *   `Embed`
*   **[`passthrough-parser`](placeholder-link)**: A model-agnostic parser that supports any request format by passing the request body through without interpretation.

#### Request Control Plugins

##### Admitter Plugins
*   **[`latency-slo-admitter`](placeholder-link)**: Rejects sheddable requests (priority < 0) when no endpoint can meet latency SLO constraints.

##### Data Producers
*   **[`predicted-latency-producer`](placeholder-link)**: Trains XGBoost models via a sidecar and generates per-endpoint TTFT/TPOT predictions. It calculates SLO headroom, collects training data, and tracks per-endpoint running request queues.
*   **[`inflight-load-producer`](placeholder-link)**: Tracks the number of in-flight requests and estimated tokens for each endpoint. It increments counts in `PreRequest` and decrements them in `ResponseBodyProcessor` on end-of-stream.
*   **[`approx-prefix-cache-producer`](placeholder-link)**: Prepares data for approximate prefix cache aware scheduling by hashing prompts in blocks and matching them against an indexer of cached prefixes on servers.

