"""Demo script showcasing the agent abstraction layer.

This example demonstrates:
1. Creating DeviceAgents from existing devices
2. Hierarchical coordination with GridAgent
3. Price-based coordination protocol
4. Observation extraction and action computation
"""

import numpy as np
from powergrid.agents import (
    DeviceAgent,
    GridAgent,
    PriceSignalProtocol,
    Observation,
)
from powergrid.devices.storage import ESS
from powergrid.devices.generator import DG


def main():
    print("=" * 60)
    print("PowerGrid Agent Abstraction Layer Demo")
    print("=" * 60)

    # ========================================
    # Step 1: Create Devices
    # ========================================
    print("\n[1] Creating devices...")

    ess = ESS(
        name="ess_1",
        bus=800,
        min_p_mw=-0.5,
        max_p_mw=0.5,
        capacity=1.0,
        init_soc=0.5,
    )

    dg = DG(
        name="dg_1",
        bus="806",
        min_p_mw=0.0,
        max_p_mw=0.5,
    )

    print(f"  - Created ESS: {ess.name} (capacity={ess.capacity} MWh)")
    print(f"  - Created DG: {dg.name} (max_p={dg.max_p_mw} MW)")

    # ========================================
    # Step 2: Wrap as DeviceAgents
    # ========================================
    print("\n[2] Wrapping devices as agents...")

    ess_agent = DeviceAgent(device=ess, partial_obs=False)
    dg_agent = DeviceAgent(device=dg, partial_obs=False)

    print(f"  - {ess_agent}")
    print(f"    Action space: {ess_agent.action_space}")
    print(f"  - {dg_agent}")
    print(f"    Action space: {dg_agent.action_space}")

    # ========================================
    # Step 3: Create GridCoordinator
    # ========================================
    print("\n[3] Creating grid coordinator with price-based protocol...")

    protocol = PriceSignalProtocol(initial_price=50.0)
    coordinator = GridAgent(
        agent_id="mg_controller",
        subordinates=[ess_agent, dg_agent],
        protocol=protocol,
    )

    print(f"  - {coordinator}")

    # ========================================
    # Step 4: Reset All Agents
    # ========================================
    print("\n[4] Resetting agents...")

    coordinator.reset(seed=42)
    print(f"  - Coordinator reset")
    print(f"  - ESS SOC: {ess.state.soc:.3f}")
    print(f"  - DG P: {dg.state.P:.3f} MW")

    # ========================================
    # Step 5: Simulate One Timestep
    # ========================================
    print("\n[5] Running one timestep...")

    # Create global state
    global_state = {
        "bus_vm": {800: 1.05, "806": 1.03},
        "bus_va": {800: 0.0, "806": 0.1},
        "converged": True,
        "dataset": {
            "price": 60.0,  # High price - ESS should discharge
            "load": 1.0,
        },
    }

    print(f"  - Global state: price={global_state['dataset']['price']} $/MWh")

    # Coordinator observes and acts
    coordinator_obs = coordinator.observe(global_state)
    coordinator.act(coordinator_obs)

    print(f"  - Coordinator acted (broadcast price signal)")

    # Device agents observe and act
    print("\n  Device agent actions:")

    ess_obs = ess_agent.observe(global_state)
    ess_action = ess_agent.act(ess_obs)
    print(f"    - ESS action: {ess_action}")
    print(f"      Mailbox: {len(ess_agent.mailbox)} messages")
    if ess_agent.mailbox:
        print(f"      Price signal: {ess_agent.mailbox[0].content.get('price')}")

    dg_obs = dg_agent.observe(global_state)
    dg_action = dg_agent.act(dg_obs)
    print(f"    - DG action: {dg_action}")
    print(f"      Mailbox: {len(dg_agent.mailbox)} messages")

    # ========================================
    # Step 6: Check Observation Structure
    # ========================================
    print("\n[6] Observation structure:")

    print(f"  ESS Observation:")
    print(f"    - Local state: {list(ess_obs.local.keys())}")
    print(f"    - Global info: {list(ess_obs.global_info.keys())}")
    print(f"    - SOC: {ess_obs.local.get('soc', 'N/A')}")
    print(f"    - Bus voltage: {ess_obs.global_info.get('bus_voltage', 'N/A')}")

    # ========================================
    # Step 7: Demonstrate Observation Vector
    # ========================================
    print("\n[7] Converting observation to vector for RL:")

    obs_vec = ess_obs.as_vector()
    print(f"  - Vector shape: {obs_vec.shape}")
    print(f"  - Vector: {obs_vec}")

    # ========================================
    # Step 8: Multiple Timesteps
    # ========================================
    print("\n[8] Running 5 timesteps with varying prices...")

    prices = [40.0, 50.0, 70.0, 60.0, 45.0]

    for t, price in enumerate(prices):
        global_state["dataset"]["price"] = price

        # Update timesteps
        coordinator.update_timestep(float(t))
        ess_agent.update_timestep(float(t))
        dg_agent.update_timestep(float(t))

        # Clear mailboxes
        ess_agent.clear_mailbox()
        dg_agent.clear_mailbox()

        # Step
        coordinator_obs = coordinator.observe(global_state)
        coordinator.act(coordinator_obs)

        ess_obs = ess_agent.observe(global_state)
        ess_action = ess_agent.act(ess_obs)

        dg_obs = dg_agent.observe(global_state)
        dg_action = dg_agent.act(dg_obs)

        print(f"  t={t}: price=${price:.1f}/MWh, ESS action={ess_action[0]:.3f} MW")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
