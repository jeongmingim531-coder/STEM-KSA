# STEM-KSA Energy Harvesting Simulation

This repository contains a Python-based simulation of a closed-loop energy-harvesting system demonstrating analogues of the four fundamental forces:

- **Gravity** drives a 20 mm steel ball through a transparent kinetic spiral beneath a pressure-sensitive floor tile.
- **Electromagnetism** is captured via Halbach-array-enhanced induction coils and an electromagnetic elevator reset.
- **Strong force analogue** is modelled with a metamaterial lattice of spring-mass-magnetic interactions that stores and releases energy.
- **Weak force analogue** introduces stochastic decay events within the lattice that destabilise the stored energy and modulate the release profile.

## Running the simulation

1. Ensure you have Python 3.9+ available (the simulation depends only on the standard library).
2. Execute the main script:

   ```bash
   python simulation/main.py
   ```

## Outputs

Running the script produces artefacts in `simulation/outputs/`:

- `system_overview.svg`: 3D render and graph montage showing the foot press, spiral, elevator, lattice dynamics, voltage/power traces, and energy plots.
- `spiral_data.csv`, `lattice_data.csv`, `decay_events.csv`: Detailed time histories of the spiral motion, lattice dynamics, and weak-force-inspired decay events.
- `spiral_track.stl`, `lattice.stl`: Geometry exports for the spiral track and metamaterial lattice that can be inspected in CAD/3D tools.
- Console output summarises harvested energy, elevator energy cost, and net cycle energy balance.

The simulation logs induced EMF, current, power, cumulative harvested energy, energy stored/released in the strong-force analogue, and decay-triggered events, allowing further analysis of the interacting subsystems.
