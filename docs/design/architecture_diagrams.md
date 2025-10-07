# PowerGrid 2.0: Implementation Design Diagrams

**Purpose**: Visual guides for 3-person team implementation
**Date**: October 2025
**Version**: 1.0

---

## Diagram Index

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Class Hierarchy Diagram](#2-class-hierarchy-diagram)
3. [Agent Lifecycle Sequence](#3-agent-lifecycle-sequence)
4. [Environment Step Flow](#4-environment-step-flow)
5. [Data Flow Architecture](#5-data-flow-architecture)
6. [Module Dependency Graph](#6-module-dependency-graph)
7. [Experiment Pipeline](#7-experiment-pipeline)
8. [Baseline Integration Architecture](#8-baseline-integration-architecture)
9. [Implementation Phases](#9-implementation-phases)
10. [Team Responsibilities Matrix](#10-team-responsibilities-matrix)

---

## 1. System Architecture Overview

```mermaid
graph TB
    subgraph "External Interfaces"
        User[Researcher/User]
        RLlib[RLlib/Ray]
        SB3[Stable-Baselines3]
        YAML[YAML Configs]
    end

    subgraph "PowerGrid Core - Month 1-2 Implementation"
        subgraph "Agent Layer - agents/"
            Agent[Agent ABC<br/>observe, act, communicate]
            DevAgent[DeviceAgent<br/>wraps Device]
            GridAgent[GridCoordinatorAgent<br/>manages sub-agents]
        end

        subgraph "Environment Layer - envs/"
            MAEnv[MultiAgentPowerGridEnv<br/>PettingZoo ParallelEnv]
            HierEnv[HierarchicalEnv<br/>2/3-level control]
        end

        subgraph "Device Layer - devices/"
            Device[Device ABC]
            DG[DG]
            ESS[ESS]
            RES[RES]
            Shunt[Shunt]
            Grid[Grid]
        end

        subgraph "Core Primitives - core/"
            State[DeviceState<br/>dataclass]
            Action[Action<br/>continuous/discrete]
        end
    end

    subgraph "Data & Configuration - Month 3"
        subgraph "Datasets - datasets/"
            Loader[DatasetLoader]
            CAISO[CAISO 2020-2024]
            ERCOT[ERCOT Data]
        end

        subgraph "Config System - config/"
            Schema[ConfigSchema]
            Factory[AgentFactory]
            Validator[ConfigValidator]
        end
    end

    subgraph "Baselines & Experiments - Month 3-4"
        subgraph "Algorithms - algorithms/"
            RLAlgos[RL Algorithms<br/>PPO, SAC, MAPPO]
            Classical[Classical<br/>MPC, OPF]
        end

        subgraph "Benchmarks - benchmarks/"
            Tasks[5 Benchmark Tasks]
            Metrics[Metrics Logger]
            Runner[Experiment Runner]
        end
    end

    subgraph "Physics Backend"
        PP[PandaPower<br/>AC/DC Power Flow]
    end

    %% External connections
    User -->|writes| YAML
    YAML -->|loads| Schema
    User -->|trains| RLlib
    User -->|trains| SB3

    %% Core connections
    Agent --> DevAgent
    Agent --> GridAgent
    DevAgent -->|wraps| Device
    Device --> DG
    Device --> ESS
    Device --> RES
    Device --> Shunt
    Device --> Grid

    Device -->|has| State
    Device -->|has| Action

    MAEnv -->|manages| GridAgent
    MAEnv -->|registers| DevAgent
    HierEnv -->|extends| MAEnv

    %% Data flow
    Loader -->|loads| CAISO
    Loader -->|loads| ERCOT
    MAEnv -->|uses| Loader

    Schema -->|creates| Factory
    Factory -->|instantiates| DevAgent
    Factory -->|instantiates| GridAgent

    %% Baseline connections
    RLlib -->|trains on| MAEnv
    SB3 -->|trains on| MAEnv
    RLAlgos -->|wraps| RLlib
    RLAlgos -->|wraps| SB3
    Classical -->|baselines| MAEnv

    Tasks -->|evaluates on| MAEnv
    Runner -->|runs| Tasks
    Runner -->|logs| Metrics

    %% Physics
    MAEnv -->|power flow| PP
    Classical -->|optimization| PP

    style Agent fill:#e8f5e9
    style DevAgent fill:#e8f5e9
    style GridAgent fill:#e8f5e9
    style MAEnv fill:#fff4e1
    style HierEnv fill:#fff4e1
    style Device fill:#f3e5f5
    style PP fill:#e3f2fd
```

---

## 2. Class Hierarchy Diagram

```mermaid
classDiagram
    class Agent {
        <<abstract>>
        +agent_id: str
        +action_space: Space
        +observation_space: Space
        +observe(global_state) Observation
        +act(observation) Action
        +receive_message(msg, sender)
        +send_message(msg, recipients)
        +reset(seed)
    }

    class DeviceAgent {
        +device: Device
        +policy: Policy
        +mailbox: List[Message]
        +observe(global_state) Observation
        +act(observation) Action
        +set_action(action)
    }

    class GridCoordinatorAgent {
        +sub_agents: List[DeviceAgent]
        +protocol: Protocol
        +coordination_freq: int
        +observe(global_state) Observation
        +act(observation) Dict[AgentID, Action]
        +coordinate(observations) Dict[AgentID, Setpoint]
    }

    class Device {
        <<abstract>>
        +name: str
        +bus: str
        +state: DeviceState
        +action: Action
        +cost: float
        +safety: float
        +set_action_space()
        +update_state()
        +update_cost_safety()
        +reset()
    }

    class DG {
        +p_max: float
        +fuel_cost: float
        +min_up_time: int
        +update_state()
        +update_cost_safety()
    }

    class ESS {
        +capacity: float
        +soc: float
        +max_p_mw: float
        +update_state()
        +update_cost_safety()
    }

    class RES {
        +type: str (solar/wind)
        +scaling: float
        +update_state()
    }

    class DeviceState {
        <<dataclass>>
        +P: float
        +Q: float
        +on: int
        +soc: float (optional)
        +as_vector() ndarray
    }

    class Action {
        <<dataclass>>
        +c: ndarray (continuous)
        +d: ndarray (discrete)
        +dim_c: int
        +dim_d: int
        +range: ndarray
        +sample()
    }

    class MultiAgentPowerGridEnv {
        <<PettingZoo ParallelEnv>>
        +net: pandapower.Network
        +agents: Dict[str, Agent]
        +protocol: Protocol
        +dataset: Dict
        +possible_agents: List[str]
        +action_spaces: Dict
        +observation_spaces: Dict
        +step(actions) Tuple
        +reset() Tuple
        -_solve_power_flow() bool
        -_compute_rewards() Dict
    }

    class HierarchicalEnv {
        +device_agents: Dict
        +grid_agents: Dict
        +system_agent: Agent
        +hierarchy_depth: int
        +step(actions) Tuple
        -_hierarchical_step() Tuple
    }

    class Protocol {
        <<abstract>>
        +coordinate(observations) Dict
    }

    class PriceSignalProtocol {
        +lmp_solver: str
        +coordinate(observations) Dict
    }

    class ADMMProtocol {
        +max_iters: int
        +rho: float
        +coordinate(observations) Dict
    }

    Agent <|-- DeviceAgent
    Agent <|-- GridCoordinatorAgent

    DeviceAgent o-- Device
    DeviceAgent o-- Policy
    GridCoordinatorAgent o-- DeviceAgent
    GridCoordinatorAgent o-- Protocol

    Device <|-- DG
    Device <|-- ESS
    Device <|-- RES
    Device o-- DeviceState
    Device o-- Action

    MultiAgentPowerGridEnv o-- Agent
    MultiAgentPowerGridEnv o-- Protocol
    HierarchicalEnv --|> MultiAgentPowerGridEnv

    Protocol <|-- PriceSignalProtocol
    Protocol <|-- ADMMProtocol
```

---

## 3. Agent Lifecycle Sequence

```mermaid
sequenceDiagram
    participant User
    participant Config as ConfigLoader
    participant Factory as AgentFactory
    participant Env as MultiAgentPowerGridEnv
    participant Agent as DeviceAgent
    participant Device as Device (DG/ESS)
    participant PP as PandaPower

    Note over User,PP: Initialization Phase (Month 2)

    User->>Config: load("config.yaml")
    Config->>Config: parse YAML
    Config->>Factory: create_agents(agent_configs)

    loop For each agent config
        Factory->>Device: __init__(params)
        Device->>Device: set_action_space()
        Factory->>Agent: __init__(device, policy)
        Factory-->>Config: agent instance
    end

    Config->>Env: __init__(net, agents, protocol)
    Env->>PP: attach devices to net
    Env->>Env: build action/obs spaces
    Config-->>User: env instance

    Note over User,PP: Training Loop (Month 4)

    User->>Env: reset(seed=42)
    Env->>Agent: reset(rng)
    Agent->>Device: reset(rng)
    Device->>Device: initialize state (SOC, etc.)
    Env->>PP: runpp(net)
    Env->>Agent: observe(global_state)
    Agent->>Device: state.as_vector()
    Agent-->>Env: observation
    Env-->>User: obs_dict, info

    loop Training steps
        User->>User: policy.forward(obs)
        User->>Env: step(actions)

        loop For each agent
            Env->>Agent: set_action(action)
            Agent->>Device: action.c = vals
        end

        loop For each device
            Env->>Device: update_state()
            Device->>Device: P = f(action)
            Device->>Device: SOC += P*dt
        end

        Env->>PP: sync devices to net
        Env->>PP: runpp(net)
        PP-->>Env: converged, results

        loop For each device
            Env->>Device: update_cost_safety()
            Device->>Device: cost = fuel * P
            Device->>Device: safety = constraint_viol
        end

        Env->>Env: aggregate rewards
        Env->>Agent: observe(global_state)
        Agent-->>Env: observation
        Env-->>User: obs, rewards, dones, info

        User->>User: policy.update(experience)
    end
```

---

## 4. Environment Step Flow

```mermaid
flowchart TD
    Start([step actions])

    subgraph "1. Action Distribution"
        A1[Parse action_dict by agent_id]
        A2[For each DeviceAgent:<br/>agent.set_action]
        A3[Device.action.c/d = values]
    end

    subgraph "2. State Update - Physics"
        B1[For each Device:<br/>device.update_state]
        B2{Device Type?}
        B3[DG: P = P_min + action * range<br/>Q = tan × P]
        B4[ESS: P = action × P_max<br/>SOC += P × dt]
        B5[RES: P = action × scaling<br/>Q = reactive power]
    end

    subgraph "3. Pandapower Sync"
        C1[Push device states to net]
        C2[net.sgen.at = DG.state.P]
        C3[net.storage.at = ESS.state.SOC]
        C4[net.shunt.at = Shunt.Q]
    end

    subgraph "4. Power Flow Solution"
        D1[pp.runpp]
        D2{Converged?}
        D3[Get results:<br/>vm_pu, va_degree, loading]
        D4[Set failed flag]
    end

    subgraph "5. Cost & Safety"
        E1[For each Device:<br/>update_cost_safety]
        E2[DG: cost = fuel_cost × P]
        E3[ESS: safety = SOC_violation]
        E4[Voltage: safety = |V - 1.0|]
    end

    subgraph "6. Reward Computation"
        F1[Call _reward_and_safety]
        F2[Per-device rewards]
        F3[Aggregate: sum or weighted]
        F4[Apply safety penalty]
    end

    subgraph "7. Observation"
        G1[For each Agent:<br/>agent.observe]
        G2[Local: device state]
        G3[Global: bus voltages]
        G4[Messages: mailbox]
    end

    subgraph "8. Communication Protocol"
        H1{Protocol enabled?}
        H2[protocol.coordinate]
        H3[Broadcast messages]
        H4[Update agent mailboxes]
    end

    End([Return: obs, rewards, dones, info])

    Start --> A1
    A1 --> A2
    A2 --> A3
    A3 --> B1

    B1 --> B2
    B2 --> B3
    B2 --> B4
    B2 --> B5
    B3 --> C1
    B4 --> C1
    B5 --> C1

    C1 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> D1

    D1 --> D2
    D2 -->|Yes| D3
    D2 -->|No| D4
    D3 --> E1
    D4 --> E1

    E1 --> E2
    E1 --> E3
    E1 --> E4
    E2 --> F1
    E3 --> F1
    E4 --> F1

    F1 --> F2
    F2 --> F3
    F3 --> F4
    F4 --> G1

    G1 --> G2
    G2 --> G3
    G3 --> G4
    G4 --> H1

    H1 -->|Yes| H2
    H1 -->|No| End
    H2 --> H3
    H3 --> H4
    H4 --> End

    style Start fill:#e8f5e9
    style End fill:#e8f5e9
    style D1 fill:#e3f2fd
    style D2 fill:#fff3e0
```

---

## 5. Data Flow Architecture

```mermaid
graph LR
    subgraph "Data Sources - Month 3 Week 9"
        CAISO[(CAISO OASIS<br/>2020-2024<br/>Load/Solar/LMP)]
        ERCOT[(ERCOT<br/>Load/Wind)]
        NREL[(NREL NSRDB<br/>Solar Profiles)]
    end

    subgraph "Data Pipeline - datasets/"
        Download[DataDownloader<br/>auto-fetch from APIs]
        Preprocess[Preprocessor<br/>clean, interpolate]
        Align[TimeseriesAligner<br/>match timestamps]
        Split[TrainTestSplitter<br/>2020-2022 train<br/>2023-2024 test]
        Cache[DataCache<br/>~/.powergrid/data/]
    end

    subgraph "Dataset Objects"
        TSDataset[TimeseriesDataset<br/>load[t], solar[t], price[t]]
        SpatialDS[SpatialDataset<br/>bus locations, topology]
    end

    subgraph "Environment Integration"
        Env[MultiAgentPowerGridEnv]
        Loader[DatasetLoader]
    end

    subgraph "Device Consumption"
        RES[RES Device<br/>scaling = solar[t]]
        Grid[Grid Device<br/>price = lmp[t]]
        Load[Load Scaling<br/>net.load.scaling = load[t]]
    end

    CAISO -->|download| Download
    ERCOT -->|download| Download
    NREL -->|download| Download

    Download --> Preprocess
    Preprocess --> Align
    Align --> Split
    Split --> Cache

    Cache --> TSDataset
    Cache --> SpatialDS

    TSDataset --> Loader
    SpatialDS --> Loader
    Loader --> Env

    Env --> RES
    Env --> Grid
    Env --> Load

    style CAISO fill:#e3f2fd
    style Cache fill:#fff3e0
    style Env fill:#fff4e1
```

---

## 6. Module Dependency Graph

```mermaid
graph TD
    subgraph "Layer 1: Primitives (No dependencies)"
        Core[core/<br/>state.py, actions.py]
        Utils[utils/<br/>utils.py, cost.py, safety.py]
    end

    subgraph "Layer 2: Devices (Depends on Layer 1)"
        Devices[devices/<br/>base.py, generator.py,<br/>storage.py, grid.py]
    end

    subgraph "Layer 3: Agents (Depends on Layer 2)"
        Agents[agents/<br/>base.py, device_agent.py,<br/>grid_agent.py]
    end

    subgraph "Layer 4: Protocols (Depends on Layer 3)"
        Protocols[communication/<br/>protocols.py, market.py,<br/>consensus.py]
    end

    subgraph "Layer 5: Environments (Depends on Layer 3-4)"
        Envs[envs/<br/>multi_agent/base.py,<br/>hierarchical.py]
    end

    subgraph "Layer 6: Data (Independent)"
        Datasets[datasets/<br/>loaders.py, preprocessors.py]
    end

    subgraph "Layer 7: Configuration (Depends on Layer 3,5)"
        Config[config/<br/>loader.py, schema.py,<br/>factory.py]
    end

    subgraph "Layer 8: Baselines (Depends on Layer 5)"
        Algos[algorithms/<br/>rl_algorithms.py,<br/>mpc.py, opf.py]
    end

    subgraph "Layer 9: Benchmarks (Depends on Layer 5,8)"
        Bench[benchmarks/<br/>tasks.py, runner.py,<br/>metrics.py]
    end

    subgraph "External"
        PP[PandaPower]
        Gym[Gymnasium]
        PZ[PettingZoo]
        RLlib[RLlib]
        SB3[Stable-Baselines3]
    end

    Core --> Devices
    Utils --> Devices

    Devices --> Agents

    Agents --> Protocols
    Agents --> Envs
    Protocols --> Envs

    Envs --> Config
    Agents --> Config

    Envs --> Algos
    Algos --> Bench
    Envs --> Bench

    Datasets --> Envs

    PP --> Devices
    PP --> Envs
    Gym --> Agents
    PZ --> Envs
    RLlib --> Algos
    SB3 --> Algos

    style Core fill:#e8f5e9
    style Devices fill:#f3e5f5
    style Agents fill:#e8f5e9
    style Envs fill:#fff4e1
    style PP fill:#e3f2fd
```

**Implementation Order** (follow layer numbers):
1. Core primitives (Week 1)
2. Devices (Week 1-2)
3. Agents (Week 2-3)
4. Protocols (Week 7, can defer)
5. Environments (Week 4-6)
6. Datasets (Week 9)
7. Configuration (Week 5-6)
8. Baselines (Week 10-11)
9. Benchmarks (Week 12-13)

---

## 7. Experiment Pipeline

```mermaid
flowchart LR
    subgraph "Setup - Month 3"
        Config[config.yaml<br/>5 tasks × 7 algos]
        Scripts[experiment_scripts/<br/>train.py, eval.py]
    end

    subgraph "Execution - Month 4 Week 13-14"
        Ray[Ray Cluster<br/>4 GPUs]
        Parallel[Parallel Trainer<br/>25 jobs/day]
        Monitor[WandB/TensorBoard<br/>live monitoring]
    end

    subgraph "Training"
        T1[Task 1: Voltage<br/>IEEE34, 5 agents]
        T2[Task 2: Dispatch<br/>IEEE123, 15 agents]
        T3[Task 3: Peak Shaving<br/>CIGRE, 8 agents]
        T4[Task 4: Resilience<br/>IEEE34+faults]
        T5[Task 5: Scalability<br/>200-bus, 100 agents]
    end

    subgraph "Storage"
        Logs[Training Logs<br/>CSV, JSON]
        Ckpts[Checkpoints<br/>model_*.pkl]
        TB[TensorBoard<br/>tfevents]
    end

    subgraph "Analysis - Month 4 Week 15"
        Load[Load Results]
        Stats[Statistical Tests<br/>t-test, bootstrap]
        Plots[Generate Plots<br/>matplotlib, seaborn]
    end

    subgraph "Outputs"
        Figs[Figures/<br/>learning_curves.pdf<br/>scalability.pdf]
        Tables[Tables/<br/>performance.csv<br/>ablations.csv]
    end

    Config --> Scripts
    Scripts --> Ray
    Ray --> Parallel

    Parallel --> T1
    Parallel --> T2
    Parallel --> T3
    Parallel --> T4
    Parallel --> T5

    T1 --> Monitor
    T2 --> Monitor
    T3 --> Monitor
    T4 --> Monitor
    T5 --> Monitor

    T1 --> Logs
    T2 --> Logs
    T3 --> Logs
    T4 --> Logs
    T5 --> Logs

    T1 --> Ckpts
    T2 --> Ckpts
    T3 --> Ckpts
    T4 --> Ckpts
    T5 --> Ckpts

    T1 --> TB
    T2 --> TB
    T3 --> TB
    T4 --> TB
    T5 --> TB

    Logs --> Load
    Ckpts --> Load
    TB --> Load

    Load --> Stats
    Stats --> Plots

    Plots --> Figs
    Stats --> Tables

    style Ray fill:#e3f2fd
    style Monitor fill:#fff3e0
    style Figs fill:#e8f5e9
    style Tables fill:#e8f5e9
```

**Key Metrics to Track**:
- Episode return (primary)
- Steps to convergence
- Training time
- Memory usage
- Safety violations (%)
- Power flow convergence rate

---

## 8. Baseline Integration Architecture

```mermaid
graph TB
    subgraph "Baseline Interface"
        BaseAlgo[BaseAlgorithm ABC<br/>train, evaluate, save, load]
    end

    subgraph "RL Baselines - algorithms/rl/"
        RLWrapper[RLAlgorithmWrapper<br/>wraps RLlib/SB3]

        subgraph "Single-Agent"
            PPO[PPO]
            SAC[SAC]
        end

        subgraph "Multi-Agent"
            IPPO[IPPO<br/>independent learners]
            MAPPO[MAPPO<br/>shared critic]
            QMIX[QMIX<br/>value factorization]
        end

        subgraph "Hierarchical"
            HMAPPO[H-MAPPO<br/>2-level coordinator]
        end
    end

    subgraph "Classical Baselines - algorithms/classical/"
        MPC[MPC<br/>Model Predictive Control]
        OPF[OPF<br/>Optimal Power Flow]
        RBC[RBC<br/>Rule-Based Control]
    end

    subgraph "External Libraries"
        RLlib_ext[RLlib]
        SB3_ext[Stable-Baselines3]
        CVXPY[CVXPY<br/>optimization]
        PP_ext[PandaPower<br/>OPF solver]
    end

    subgraph "Evaluation"
        Evaluator[Evaluator<br/>run rollouts, compute metrics]
        Metrics[Metrics Logger<br/>reward, time, memory]
    end

    BaseAlgo --> RLWrapper
    BaseAlgo --> MPC
    BaseAlgo --> OPF
    BaseAlgo --> RBC

    RLWrapper --> PPO
    RLWrapper --> SAC
    RLWrapper --> IPPO
    RLWrapper --> MAPPO
    RLWrapper --> QMIX
    RLWrapper --> HMAPPO

    PPO --> RLlib_ext
    SAC --> SB3_ext
    IPPO --> RLlib_ext
    MAPPO --> RLlib_ext
    QMIX --> RLlib_ext

    MPC --> CVXPY
    OPF --> PP_ext

    PPO --> Evaluator
    SAC --> Evaluator
    IPPO --> Evaluator
    MAPPO --> Evaluator
    MPC --> Evaluator
    OPF --> Evaluator

    Evaluator --> Metrics

    style BaseAlgo fill:#e8f5e9
    style RLWrapper fill:#fff4e1
    style MPC fill:#e3f2fd
    style OPF fill:#e3f2fd
```

**Baseline Implementation Priority**:
1. **Week 10**: PPO, IPPO, MAPPO (RLlib)
2. **Week 11**: SAC (SB3), MPC (CVXPY)
3. **Week 12**: OPF (PandaPower), H-MAPPO
4. **Defer to NeurIPS**: QMIX, MADDPG

---

## 9. Implementation Phases

```mermaid
gantt
    title PowerGrid 2.0 Implementation Timeline
    dateFormat YYYY-MM-DD

    section Month 1: Foundation
    Agent Base Classes           :a1, 2025-10-01, 7d
    Device Refactor             :a2, after a1, 7d
    Unit Tests                  :a3, after a2, 7d
    Code Review & Merge         :a4, after a3, 7d

    section Month 2: Multi-Agent
    PettingZoo Environment      :b1, 2025-11-01, 7d
    IEEE 13/34 Examples         :b2, after b1, 7d
    Hierarchical Example        :b3, after b2, 7d
    Documentation               :b4, after b3, 7d

    section Month 3: Data & Baselines
    CAISO Data Pipeline         :c1, 2025-12-01, 7d
    RLlib Integration           :c2, after c1, 7d
    Classical Baselines         :c3, after c2, 7d
    Dry Run Experiments         :c4, after c3, 7d

    section Month 4: Experiments
    Main Experiments (175 runs) :d1, 2026-01-01, 14d
    Ablation Studies            :d2, after d1, 7d
    Paper Draft                 :d3, after d2, 14d
    Submit e-Energy             :milestone, d4, 2026-01-31, 0d
```

---

## 10. Team Responsibilities Matrix

```mermaid
graph TB
    subgraph "Architect - Core Infrastructure"
        A1[Week 1-4: Agent Abstraction<br/>agents/base.py, device_agent.py]
        A2[Week 5-8: PettingZoo Env<br/>envs/multi_agent/base.py]
        A3[Week 9-12: Baselines<br/>RLlib, hierarchical]
        A4[Week 13-18: Experiments & Paper<br/>Run 175 trials, write paper]
    end

    subgraph "Domain Engineer - Devices & Data"
        D1[Week 1-4: Device Refactor<br/>Convert to DeviceAgent]
        D2[Week 5-8: Examples<br/>IEEE 13/34, YAML configs]
        D3[Week 9-12: Datasets<br/>CAISO integration, MPC]
        D4[Week 13-18: Analysis<br/>Plots, tables, statistics]
    end

    subgraph "DevOps - Testing & Docs"
        O1[Week 1-4: Test Infrastructure<br/>pytest, CI/CD]
        O2[Week 5-8: Integration Tests<br/>RLlib, SB3 compatibility]
        O3[Week 9-12: Training Scripts<br/>Ray cluster, monitoring]
        O4[Week 13-18: Reproducibility<br/>Docker, docs, supplement]
    end

    subgraph "Collaboration Points"
        C1[Week 4: Code Review<br/>All review agent PR]
        C2[Week 8: Demo<br/>Train MAPPO on IEEE34]
        C3[Week 12: Dry Run<br/>Test all baselines]
        C4[Week 17: Internal Review<br/>3 reviewers on paper]
    end

    A1 -.->|Reviews| C1
    D1 -.->|Reviews| C1
    O1 -.->|Reviews| C1

    A2 -.->|Demo| C2
    D2 -.->|Provides configs| C2
    O2 -.->|Runs tests| C2

    A3 -.->|Baselines| C3
    D3 -.->|Data ready| C3
    O3 -.->|Scripts ready| C3

    A4 -.->|Draft| C4
    D4 -.->|Figures| C4
    O4 -.->|Supplement| C4

    style A1 fill:#e3f2fd
    style A2 fill:#e3f2fd
    style A3 fill:#e3f2fd
    style A4 fill:#e3f2fd

    style D1 fill:#fff3e0
    style D2 fill:#fff3e0
    style D3 fill:#fff3e0
    style D4 fill:#fff3e0

    style O1 fill:#e8f5e9
    style O2 fill:#e8f5e9
    style O3 fill:#e8f5e9
    style O4 fill:#e8f5e9
```

---

## Quick Reference: Key Files to Implement

### Month 1 (Agent Abstraction)
```
powergrid/
├── agents/
│   ├── __init__.py
│   ├── base.py              # Agent ABC (Architect, Week 1)
│   ├── device_agent.py      # DeviceAgent wrapper (Architect, Week 1-2)
│   ├── grid_agent.py        # GridCoordinatorAgent (Architect, Week 2-3)
│   └── policies.py          # Policy interfaces (Architect, Week 3)
├── core/
│   ├── state.py             # DeviceState (existing, minor updates)
│   └── actions.py           # Action (existing, minor updates)
└── devices/
    ├── base.py              # Device ABC (Domain, refactor Week 1-2)
    ├── generator.py         # DG as agent (Domain, Week 2)
    ├── storage.py           # ESS as agent (Domain, Week 2)
    └── ...                  # Other devices (Domain, Week 3)
```

### Month 2 (Multi-Agent Environment)
```
powergrid/
├── envs/
│   ├── multi_agent/
│   │   ├── __init__.py
│   │   ├── base.py          # MultiAgentPowerGridEnv (Architect, Week 5-6)
│   │   ├── ieee13.py        # 3-agent example (Domain, Week 5)
│   │   └── ieee34.py        # 5-agent example (Domain, Week 6)
│   └── hierarchical/
│       ├── __init__.py
│       └── base.py          # HierarchicalEnv (Domain, Week 7)
└── config/
    ├── __init__.py
    ├── loader.py            # ConfigLoader (Architect, Week 6-7)
    ├── schema.py            # YAML schema (DevOps, Week 6)
    └── factory.py           # AgentFactory (Architect, Week 7)
```

### Month 3 (Data & Baselines)
```
powergrid/
├── datasets/
│   ├── __init__.py
│   ├── loaders.py           # DatasetLoader (Domain, Week 9)
│   ├── preprocessors.py     # Preprocessing (Domain, Week 9)
│   └── caiso.py             # CAISO dataset (Domain, Week 9)
├── algorithms/
│   ├── __init__.py
│   ├── base.py              # BaseAlgorithm ABC (Architect, Week 10)
│   ├── rl/
│   │   ├── rllib_wrapper.py # RLlib integration (Architect, Week 10)
│   │   └── sb3_wrapper.py   # SB3 integration (Architect, Week 10)
│   └── classical/
│       ├── mpc.py           # MPC baseline (Domain, Week 11)
│       └── opf.py           # OPF baseline (Domain, Week 11)
└── benchmarks/
    ├── __init__.py
    ├── tasks.py             # 5 benchmark tasks (Domain, Week 11)
    ├── runner.py            # Experiment runner (DevOps, Week 11)
    └── metrics.py           # Metrics logger (DevOps, Week 11)
```

### Month 4 (Experiments)
```
experiments/
├── configs/
│   ├── task1_voltage.yaml   # Config for Task 1 (Domain, Week 12)
│   ├── task2_dispatch.yaml  # Config for Task 2 (Domain, Week 12)
│   └── ...
├── scripts/
│   ├── train.py             # Training script (DevOps, Week 12)
│   ├── eval.py              # Evaluation script (DevOps, Week 12)
│   └── analysis.py          # Statistical analysis (Domain, Week 15)
├── results/
│   ├── logs/                # Training logs (auto-generated)
│   ├── checkpoints/         # Model checkpoints (auto-generated)
│   └── figures/             # Plots for paper (Domain, Week 15)
└── paper/
    ├── paper.tex            # LaTeX source (All, Week 16-17)
    ├── figures/             # Camera-ready figures (Domain, Week 15)
    └── supplement.pdf       # Supplementary material (DevOps, Week 17)
```

---

## Additional Visual Aids

### Communication Protocol Flow
```mermaid
sequenceDiagram
    participant A1 as Agent 1 (DG)
    participant A2 as Agent 2 (ESS)
    participant A3 as Agent 3 (RES)
    participant Protocol as PriceSignalProtocol
    participant Coordinator as GridCoordinator

    Note over A1,Coordinator: Step 1: Observations
    A1->>Coordinator: obs_1 (local state)
    A2->>Coordinator: obs_2 (local state)
    A3->>Coordinator: obs_3 (local state)

    Note over A1,Coordinator: Step 2: Coordination
    Coordinator->>Protocol: coordinate(all_obs)
    Protocol->>Protocol: solve_opf() or compute_lmp()
    Protocol-->>Coordinator: price_signals

    Note over A1,Coordinator: Step 3: Message Passing
    Coordinator->>A1: send_message(price=50.2)
    Coordinator->>A2: send_message(price=50.2)
    Coordinator->>A3: send_message(price=50.2)

    Note over A1,Coordinator: Step 4: Individual Actions
    A1->>A1: act(obs, price) → reduce P
    A2->>A2: act(obs, price) → discharge
    A3->>A3: act(obs, price) → maintain

    A1->>Coordinator: action_1
    A2->>Coordinator: action_2
    A3->>Coordinator: action_3
```

### Hierarchical Control Flow
```mermaid
graph TD
    subgraph "Level 3: System Operator (Hourly)"
        ISO[ISO Agent<br/>Set prices, reserves]
    end

    subgraph "Level 2: Microgrid Controllers (15 min)"
        MG1[MG1 Coordinator<br/>Local optimization]
        MG2[MG2 Coordinator<br/>Local optimization]
    end

    subgraph "Level 1: Device Agents (1 min)"
        DG1[DG Agent]
        ESS1[ESS Agent]
        RES1[RES Agent]

        DG2[DG Agent]
        ESS2[ESS Agent]
    end

    ISO -->|price signal| MG1
    ISO -->|price signal| MG2

    MG1 -->|P setpoint| DG1
    MG1 -->|P setpoint| ESS1
    MG1 -->|Q setpoint| RES1

    MG2 -->|P setpoint| DG2
    MG2 -->|P setpoint| ESS2

    DG1 -->|state| MG1
    ESS1 -->|state| MG1
    RES1 -->|state| MG1

    MG1 -->|aggregated state| ISO
    MG2 -->|aggregated state| ISO

    style ISO fill:#e8f5e9
    style MG1 fill:#fff4e1
    style MG2 fill:#fff4e1
    style DG1 fill:#e3f2fd
    style ESS1 fill:#e3f2fd
    style RES1 fill:#e3f2fd
```

---

## Usage Guide for Team

### For Architect
**Focus on**: Diagrams 1, 2, 3, 4, 6, 8
- Use diagram 2 (Class Hierarchy) to design inheritance structure
- Use diagram 3 (Agent Lifecycle) for initialization logic
- Use diagram 4 (Step Flow) for environment implementation
- Use diagram 6 (Dependencies) to avoid circular imports

### For Domain Engineer
**Focus on**: Diagrams 1, 4, 5, 7, 9
- Use diagram 4 (Step Flow) to understand device state updates
- Use diagram 5 (Data Flow) for dataset integration
- Use diagram 7 (Experiment Pipeline) for benchmark setup
- Use diagram 9 (Timeline) to track progress

### For DevOps
**Focus on**: Diagrams 6, 7, 9, 10
- Use diagram 6 (Dependencies) for test coverage
- Use diagram 7 (Experiment Pipeline) for infrastructure setup
- Use diagram 9 (Timeline) for CI/CD milestones
- Use diagram 10 (Responsibilities) to coordinate reviews

---

## Next Steps

1. **Print or display** these diagrams during kickoff meeting
2. **Reference diagram numbers** in code comments (e.g., "See Diagram 4, Step 2")
3. **Update diagrams** as design evolves (living document)
4. **Use for onboarding** new contributors

---

**Document Version**: 1.0
**Last Updated**: October 2025
**Status**: Ready for Implementation
