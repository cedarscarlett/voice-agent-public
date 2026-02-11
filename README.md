## Overview

Real-time voice agents often fail in production because they are implemented
like chatbots rather than streaming systems.
This repository demonstrates the orchestration architecture required to build
**interruptible, observable, deterministic voice agents** that operate reliably
across browser and telephony environments.

Instead of focusing on conversational behavior, this project focuses on
**systems correctness under real-time constraints** — coordinating ASR, LLM,
and TTS streams while preserving low latency, run isolation, and
interruptibility.

This architecture enables voice agents capable of handling multi-turn
conversations over phone systems while remaining responsive to user
interruptions — a requirement for applications like scheduling, customer
service automation, and outbound qualification calls.


## Real-Time Voice Agent Orchestrator

This repository implements a **streaming, event-driven orchestration system for
real-time voice agents**.
Rather than treating voice interaction as a chatbot problem, the system models
it as a **deterministic streaming pipeline** coordinating ASR, LLM, TTS, and
telephony audio in a low-latency environment.

The project focuses on the **architecture required to make voice agents
reliable and interruptible**, including:

- reducer-based orchestration
- run-scoped execution and cancellation
- streaming token and audio pipelines
- chunked speech synthesis
- tool-call continuation
- telephony audio compatibility (μ-law 8 kHz)
- structured observability
- low-latency interaction targets


The goal of this repository is not to provide a framework or SDK, but to
**demonstrate the systems design required to build robust real-time AI voice
interactions**.

---

## Technology Stack

This project is implemented in **Python** using an asynchronous, event-driven
architecture.

The system integrates real provider APIs through adapter layers, including:

- Streaming ASR provider
- Streaming LLM provider
- Text-to-Speech provider
- Telephony media streams
- WebRTC/browser audio sessions

Core technologies used in the repository include:

- Python asyncio
- streaming HTTP / websocket APIs
- PCM16 audio pipeline
- μ-law 8 kHz telephony audio conversion
- structured JSON logging

Providers are intentionally isolated behind adapters so the orchestration layer
remains provider-agnostic.

---

## System Overview

At a high level, the voice agent is implemented as a **streaming pipeline
coordinated by a deterministic orchestrator**.

The system processes audio and language incrementally rather than turn-by-turn:

```
User Audio
   ↓
ASR Adapter (streaming transcription)
   ↓
Orchestrator (event reducer + state machine)
   ↓
LLM Adapter (streaming tokens)
   ↓
TTS Adapter (chunked synthesis)
   ↓
Audio Output (browser or telephony)
```

Each stage operates independently and communicates through **typed events**,
allowing the orchestrator to coordinate multiple asynchronous services without
coupling provider logic to application state.

---

---

### Session State Model

The orchestrator maintains a session state machine representing the lifecycle
of a voice interaction.

A simplified flow looks like:

```
Idle
 ↓
Listening
 ↓
Transcribing
 ↓
Generating
 ↓
Speaking
 ↓
Idle
```

Interruptions may occur during generation or speech:

```
Speaking → Interrupted → Listening
Generating → Cancelled → Listening
```

State transitions are driven exclusively by reducer events, ensuring consistent
behavior across streaming services.

### Event-Driven Orchestration

The core of the system is a **reducer-driven orchestrator** that processes
events from:

- ASR
- LLM
- TTS
- timers
- user audio input
- tool results
- telephony audio streams

Every state transition is produced by reducing an event into a new immutable
session state:

```
(state, event) → (new_state, commands)
```

Commands emitted by the reducer drive external services (LLM calls, TTS
synthesis, timers, etc.), while adapters convert provider responses back into
events.

This architecture keeps orchestration **deterministic, observable, and
testable**.

---

### Streaming Interaction Model

Unlike turn-based chat systems, the agent operates continuously:

- ASR produces partial transcripts
- the LLM streams tokens
- text is chunked for TTS
- TTS produces audio segments
- speech can be interrupted (barge-in)
- runs can be cancelled or superseded

The orchestrator maintains correctness across these overlapping streams using
**run-scoped execution IDs** and explicit state transitions.

---

### Provider Isolation via Adapters

External services are integrated through adapters that translate provider APIs
into internal events.
This keeps orchestration logic independent of any specific vendor.

Examples include:

- streaming ASR adapters
- streaming LLM adapters
- chunked TTS adapters
- telephony audio codecs

Adapters are responsible only for I/O translation — **never orchestration
decisions**.

---

### Telephony Compatibility

The system supports both browser audio sessions and telephony sessions by
normalizing audio into a consistent internal representation (PCM16).
Telephony audio is converted to and from μ-law 8 kHz where required.

This allows the same orchestration pipeline to operate across:

- WebRTC/browser audio
- Twilio media streams
- local microphone sessions

without changing orchestration logic.

---

---

## Design Goals

This project is designed to explore the engineering constraints of real-time AI
voice systems.

Key goals include:

- Deterministic orchestration under concurrent streaming conditions
- Safe interruption of speech generation
- Provider independence through adapters
- Observability of real-time pipelines
- Telephony and browser compatibility using the same core system
- Minimal coupling between infrastructure and orchestration logic

The repository intentionally prioritizes **correctness, debuggability, and
architectural clarity over feature completeness**.

## Architectural Principles

This project is built around a small set of architectural constraints intended
to make real-time voice interaction predictable, debuggable, and interruptible.

---

### Deterministic Orchestration

The orchestrator is implemented as a **pure event reducer**:

```
(state, event) → (new_state, commands)
```

External services never mutate session state directly.
Instead, adapters emit events that are reduced into new state snapshots.

This ensures:

- reproducible behavior
- explicit state transitions
- testable orchestration logic
- easier debugging of streaming interactions

The reducer is the single authority over session state.

---

### Streaming-First Design

Voice interaction is treated as a **continuous streaming process**, not a
sequence of discrete turns.

The system is designed to operate correctly while:

- ASR is still transcribing
- the LLM is still generating tokens
- TTS is still synthesizing audio
- the user may interrupt speech

All components are built to function incrementally and concurrently.

---

### Run-Scoped Execution

Each external service interaction is associated with a **run ID**.

Run IDs allow the orchestrator to:

- cancel superseded runs
- ignore stale provider responses
- safely handle interruptions (barge-in)
- coordinate overlapping streaming operations

This prevents race conditions between ASR, LLM, and TTS streams.

---

### Provider Isolation

All third-party services are accessed through **adapters**.

Adapters:
- translate provider APIs into internal events
- handle streaming I/O
- perform format conversion
- emit lifecycle events

Adapters do **not**:
- manage orchestration state
- implement conversation logic
- coordinate other services

This separation keeps orchestration logic stable even when providers change.

---

### Observability-First Design

Streaming systems are difficult to debug without visibility into state
transitions.

The orchestrator emits structured log events for:

- reducer transitions
- run lifecycle changes
- adapter activity
- timing and latency data
- audio pipeline events

Logs are session-scoped and designed to support replay and debugging of
real-time interactions.

---

### Interruptibility as a First-Class Constraint

Voice interaction must remain responsive to the user.

The system is designed so that:

- user speech can interrupt TTS playback
- new LLM runs supersede old ones
- audio output can be cancelled safely
- partial responses remain coherent

Interruptibility is handled at the orchestration level rather than within
individual adapters.
---

## Component Breakdown

The repository is organized around a small set of clearly defined system
components. Each component has a single responsibility within the streaming
voice pipeline.

---

### Orchestrator

The orchestrator coordinates all real-time interaction.

Responsibilities include:

- receiving events from adapters and runtime
- reducing events into new session state
- emitting commands for external services
- managing run lifecycles
- enforcing interruptibility rules
- coordinating streaming ASR → LLM → TTS flow

The orchestrator contains **no provider-specific logic** and operates entirely
on internal events and state.

---

### Reducer

The reducer is the core state transition mechanism:

```
(state, event) → (new_state, commands)
```

It defines how the system reacts to:

- ASR partials and finals
- LLM tokens and completion
- TTS chunk completion
- tool results
- timers
- user audio activity
- run cancellation

The reducer is intentionally deterministic and side-effect free.
All external actions are represented as emitted commands.

---

### Runtime / Gateway

The runtime (or gateway) wires the orchestrator to adapters and external
services.

Responsibilities include:

- session initialization
- adapter construction
- command execution
- routing adapter events back into the orchestrator
- managing session lifecycle

This layer acts as the boundary between orchestration logic and infrastructure.

---

### ASR Adapter

The ASR adapter handles streaming speech recognition.

Responsibilities:

- sending audio frames to the provider
- receiving transcription results
- emitting ASR events
- maintaining provider session state

The adapter exposes transcription updates as internal events without exposing
provider APIs to the orchestrator.

---

### LLM Adapter

The LLM adapter handles streaming text generation.

Responsibilities:

- sending message context to the model
- streaming tokens
- emitting LLMToken / LLMDone events
- tracking run IDs

The adapter does not interpret tokens or manage conversation logic.

---

### TTS Adapter

The TTS adapter converts text chunks into audio output.

Responsibilities:

- synthesizing audio from text segments
- emitting chunk completion events
- converting provider output into PCM16 audio
- supporting run-scoped cancellation

Speech synthesis is driven entirely by commands emitted from the reducer.

---

### Telephony Codec Layer

The telephony codec layer handles audio format conversion required for phone
calls.

Responsibilities include:

- μ-law ↔ PCM16 conversion
- 8 kHz ↔ 16 kHz handling
- framing audio for telephony media streams

This allows the same internal audio pipeline to operate across browser and
telephony environments.

---

### Logging / Observability

The system emits structured log events for all major lifecycle transitions.

Logs include:

- reducer transitions
- adapter activity
- run ID changes
- timing and latency signals
- audio pipeline events

Logging is designed to make streaming interactions inspectable and debuggable.

---
---

## Demo Overview

This repository includes a working demonstration of the orchestration system
operating in real time.

The demo is intentionally simple from a product perspective — its purpose is
to exercise the full streaming pipeline rather than showcase conversational
capabilities.

The demo demonstrates:

- real-time speech input
- streaming transcription
- streaming LLM responses
- chunked speech synthesis
- interruptible audio playback (barge-in)
- tool-call continuation
- structured logging
- run lifecycle management

---

### Browser Voice Session

In browser mode, the system captures microphone audio and streams it through
the full pipeline:

```
Microphone → ASR → Orchestrator → LLM → TTS → Speakers
```

This mode demonstrates:

- low-latency streaming interaction
- incremental transcription handling
- token streaming from the LLM
- chunked TTS playback
- interruption during speech output

---

### Telephony Session

The demo also supports voice interaction over telephony media streams.

In this mode, the system handles:

- μ-law 8 kHz audio input/output
- telephony framing
- codec conversion to PCM16
- identical orchestration logic as browser sessions

This demonstrates that the orchestrator is **transport-agnostic**, operating
independently of the audio source.

---

### Tool Call Continuation

The demo includes a simple tool integration to demonstrate orchestration across
LLM tool calls.

The flow is:

1. User asks for an action
2. LLM emits a tool call
3. Tool executes
4. Result is returned to the orchestrator
5. LLM resumes generation
6. TTS continues speaking the response

This verifies correct coordination of:

- run IDs
- streaming continuation
- assistant text accumulation
- TTS chunking across tool boundaries

---

### Observability During Demo

During a session, the system produces structured logs describing:

- state transitions
- run lifecycle events
- adapter activity
- timing signals
- audio pipeline activity

These logs are intended to make the behavior of the streaming system
transparent during development.

---
---

## Running Locally

The demo can be run locally using a Python environment with API credentials
configured for the selected providers.

---

### Requirements

- Python 3.10+
- Microphone access (for browser demo)
- API keys for configured providers (ASR, LLM, TTS)
- Optional: Twilio account for telephony demo

---

### Setup

Clone the repository:

```
git clone <repo-url>
cd <repo-name>
```

Create and activate a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

### Configuration

Set required environment variables for your providers. For example:

```
export LLM_API_KEY=...
export ASR_API_KEY=...
export TTS_API_KEY=...
```

Configuration values are read by the runtime when adapters are constructed.

---

### Start the Server

Run the gateway/server:

```
python main.py
```

The server will start a local session endpoint and initialize the orchestrator.

---

### Browser Demo

Open the browser client and start a voice session.
Microphone audio will stream through the full pipeline.

---

### Telephony Demo (Optional)

If telephony is configured:

1. Start the local server
2. Expose it using a tunnel (e.g., ngrok)
3. Configure your telephony provider webhook to the tunnel URL
4. Call the configured number

The telephony session will use the same orchestration pipeline as the browser
demo.

---

### Logs

Structured logs are written during execution and can be inspected to observe:

- reducer transitions
- adapter lifecycle events
- run IDs
- streaming activity

---

---

## Why not Bland / Vapi / Retell?

Frameworks and hosted platforms make it easy to build voice agent demos.
This project explores what those abstractions hide: the orchestration
complexity required for reliable real-time voice interaction.

Most voice agent platforms optimize for:

- rapid prototyping
- managed infrastructure
- simplified conversational pipelines
- vendor-managed orchestration

Those are valuable goals. However, they abstract away the systems-level
concerns that become critical in production environments.

This repository focuses on those underlying problems directly.

---

### Orchestration Control

Framework-based voice agents typically rely on callback chains or
internal orchestration engines that are not fully observable or
deterministic.

This project instead implements:

- explicit event-driven orchestration
- reducer-based state transitions
- run-scoped execution control
- deterministic cancellation behavior

This makes streaming behavior predictable and debuggable.

---

### Streaming Correctness

Voice interaction involves multiple concurrent streams:

- ASR partial transcripts
- LLM token generation
- TTS audio synthesis
- user interruptions

Many frameworks treat these as sequential steps.
This repository treats them as overlapping asynchronous processes
coordinated by a state machine.

---

### Interruptibility

Interrupting speech safely requires coordination across multiple
services.

This project implements interruption handling at the orchestration
layer through:

- run ID gating
- command cancellation
- reducer-managed state transitions

Rather than relying on provider-specific interruption features.

---

### Provider Independence

Hosted platforms often couple orchestration logic to specific vendors.

This project isolates providers behind adapters so the system can
operate with:

- different ASR providers
- different LLM providers
- different TTS providers
- browser or telephony audio

without modifying orchestration logic.

---

### Observability

Debugging real-time voice systems requires visibility into state
transitions and streaming behavior.

This repository exposes:

- reducer transitions
- run lifecycle events
- adapter activity
- timing signals

as structured logs, making the streaming pipeline inspectable.

---

### Project Scope

This repository is not intended to replace voice-agent frameworks or
hosted platforms.

Instead, it serves as a reference architecture for the orchestration
layer underlying real-time voice systems, demonstrating the engineering
challenges that production voice agents must solve.
