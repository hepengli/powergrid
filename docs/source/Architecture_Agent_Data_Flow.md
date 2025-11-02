# 🔁 2. Agent Data-Flow — Observation to Action Loop

The following diagram illustrates how state data moves through the agent layer, producing actions
that affect the simulated grid and completing a closed control loop.

```{mermaid}
flowchart LR    
    subgraph F["Feature Providers"]
        FP1["ElectricalProvider"]
        FP2["ThermalProvider"]
        FP3["PriceProvider"]
    end

    subgraph S["State Object"]
        S1["Aggregated Features"]
    end

    subgraph A["Agent Layer"]
        BA["BaseAgent Interface"]
        DA["DeviceAgent"]
        GA["GridAgent"]
        CA["CustomAgent"]
    end

    subgraph AC["Action Layer"]
        ACT["Action (Setpoints, Constraints)"]
    end

    subgraph ENV["Grid Environment (pandapower / Physical Grid)"]
        OBS["Telemetry, Voltages, Currents"]
        PHY["Grid Dynamics & Constraints"]
    end

    %% Dataflow
    FP1 --> S1
    FP2 --> S1
    FP3 --> S1
    S1 --> DA
    DA --> ACT
    ACT --> ENV
    ENV -->|Telemetry / Observations| DA
    DA -->|Reports & Coordination| GA
    CA -->|Overrides or Extends| DA
    GA -->|Supervisory Commands| DA
```
