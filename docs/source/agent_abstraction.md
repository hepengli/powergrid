
# 🧩 1. Agent Abstraction — Class Diagram (with caption)
```{mermaid}
classDiagram
    class BaseAgent {
        +agent_id: str
        +observation_space: Space
        +action_space: Space
        +observe()
        +act()
        +send_message()
        +receive_message()
    }

    class DeviceAgent {
        +state: State
        +action: Action
        +FeatureProviders: list
        +update_state()
        +execute_action()
    }

    class GridAgent {
        +devices: List~DeviceAgent~
        +coordinate()
        +aggregate_control()
    }

    class CustomAgent {
        +custom_logic()
        +plugin_model()
    }

    class State {
        +features: List~FeatureProvider~
        +to_vector()
    }

    class Action {
        +setpoints
        +constraints
    }

    class FeatureProvider {
        <<abstract>>
        +get_feature()
    }

    class ElectricalProvider {
        +voltage
        +current
        +power
    }

    class ThermalProvider {
        +temperature
        +thermal_limit
    }

    class PriceProvider {
        +market_price
        +tariff_signal
    }

    %% Relationships
    BaseAgent <|-- DeviceAgent
    BaseAgent <|-- GridAgent
    BaseAgent <|-- CustomAgent
    DeviceAgent --> State
    DeviceAgent --> Action
    State --> FeatureProvider
    FeatureProvider <|-- ElectricalProvider
    FeatureProvider <|-- ThermalProvider
    FeatureProvider <|-- PriceProvider
    GridAgent --> DeviceAgent : manages
    CustomAgent --> DeviceAgent : extends or replaces
```
