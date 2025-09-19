import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass
class FootPressEvent:
    force_newtons: float
    deflection_mm: float
    duration_s: float


@dataclass
class SpiralConfig:
    radius_start_m: float = 0.15
    radius_end_m: float = 0.05
    turns: float = 6.0
    drop_height_m: float = 0.18
    tilt_deg: float = 5.0
    ball_radius_m: float = 0.01
    ball_density_kg_m3: float = 7850.0
    load_resistance_ohm: float = 10.0
    halbach_peak_tesla: float = 0.35
    coil_length_m: float = 0.04
    coil_density_per_turn: int = 140


@dataclass
class RackPinionConfig:
    gear_radius_m: float = 0.015
    generator_efficiency: float = 0.72
    gear_efficiency: float = 0.88


@dataclass
class ElevatorConfig:
    lift_height_m: float = 0.18
    lift_velocity_m_s: float = 0.2
    motor_efficiency: float = 0.65


@dataclass
class LatticeConfig:
    grid_dims: Tuple[int, int, int] = (3, 3, 3)
    rest_spacing_m: float = 0.025
    spring_k: float = 1200.0
    repulsive_strength: float = 2.5e4
    repulsive_exponent: float = 4.5
    compression_mm: float = 7.0
    release_time_s: float = 0.35
    dt: float = 1e-3
    damping: float = 0.06
    generator_coupling: float = 0.55
    decay_rate_hz: float = 0.8
    decay_reduction: float = 0.4


@dataclass
class SimulationOutputs:
    spiral_data: List[Dict[str, float]]
    lattice_data: List[Dict[str, float]]
    decay_events: List[Dict[str, float]]
    total_harvested_j: float
    elevator_energy_cost_j: float
    figures: List[Path]
    stl_paths: List[Path]
    csv_paths: List[Path]


def vec_add(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_scale(a: Sequence[float], s: float) -> Tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def vec_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec_cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec_norm(a: Sequence[float]) -> float:
    return math.sqrt(vec_dot(a, a))


def vec_normalize(a: Sequence[float]) -> Tuple[float, float, float]:
    norm = vec_norm(a)
    if norm == 0:
        return (0.0, 0.0, 0.0)
    return (a[0] / norm, a[1] / norm, a[2] / norm)


class EnergyHarvesterSimulation:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures: List[Path] = []
        self.csv_paths: List[Path] = []
        self.stl_paths: List[Path] = []

    def run(self) -> SimulationOutputs:
        foot_press = FootPressEvent(force_newtons=500.0, deflection_mm=4.0, duration_s=0.35)

        spiral_cfg = SpiralConfig()
        spiral_data, energy_from_spiral = self.simulate_spiral(spiral_cfg)

        rack_cfg = RackPinionConfig()
        generator_energy = self.rack_pinion_energy(spiral_data, rack_cfg)

        elevator_cfg = ElevatorConfig()
        elevator_energy_cost = self.elevator_energy(spiral_cfg, elevator_cfg)

        lattice_cfg = LatticeConfig()
        lattice_data, decay_events = self.simulate_lattice(lattice_cfg)

        self.generate_svgs(spiral_data, lattice_data, decay_events, foot_press, spiral_cfg)
        self.export_geometry(spiral_cfg, lattice_cfg)

        spiral_csv = self.output_dir / "spiral_data.csv"
        self.write_csv(spiral_csv, spiral_data)
        self.csv_paths.append(spiral_csv)

        lattice_csv = self.output_dir / "lattice_data.csv"
        self.write_csv(lattice_csv, lattice_data)
        self.csv_paths.append(lattice_csv)

        decay_csv = self.output_dir / "decay_events.csv"
        self.write_csv(decay_csv, decay_events)
        self.csv_paths.append(decay_csv)

        total_harvested = energy_from_spiral + generator_energy

        return SimulationOutputs(
            spiral_data=spiral_data,
            lattice_data=lattice_data,
            decay_events=decay_events,
            total_harvested_j=total_harvested,
            elevator_energy_cost_j=elevator_energy_cost,
            figures=self.figures,
            stl_paths=self.stl_paths,
            csv_paths=self.csv_paths,
        )

    def simulate_spiral(self, cfg: SpiralConfig) -> Tuple[List[Dict[str, float]], float]:
        ball_volume = (4.0 / 3.0) * math.pi * cfg.ball_radius_m ** 3
        ball_mass = ball_volume * cfg.ball_density_kg_m3
        g = 9.80665
        total_angle = cfg.turns * 2 * math.pi

        steps = int(2000 * cfg.turns)
        dt = 0.002
        spiral_data: List[Dict[str, float]] = []
        sin_tilt = math.sin(math.radians(cfg.tilt_deg))
        inertia_factor = 1 + (2 / 5)
        tangential_acc = g * sin_tilt / inertia_factor
        coil_positions = [i * total_angle / 23 for i in range(24)]
        coil_strengths = [math.cos(i * math.pi / 23) ** 2 for i in range(24)]

        energy_harvested = 0.0
        cumulative_path = 0.0
        velocity = 0.0

        for step in range(steps):
            t = step * dt
            angle = total_angle * step / (steps - 1)
            radius = cfg.radius_start_m + (cfg.radius_end_m - cfg.radius_start_m) * step / (steps - 1)
            height = -cfg.drop_height_m * step / (steps - 1)

            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = height

            if step > 0:
                prev = spiral_data[-1]
                prev_pos = (prev["x_m"], prev["y_m"], prev["z_m"])
                cur_pos = (x, y, z)
                segment = vec_norm(vec_sub(cur_pos, prev_pos))
                cumulative_path += segment
                velocity += tangential_acc * dt
            emf = 0.0
            for coil_angle, strength in zip(coil_positions, coil_strengths):
                delta = angle - coil_angle
                emf += strength * math.exp(-0.5 * (delta / 0.12) ** 2)
            emf *= cfg.halbach_peak_tesla * cfg.coil_length_m * velocity * cfg.coil_density_per_turn
            current = emf / cfg.load_resistance_ohm
            power = emf * current
            energy_harvested += power * dt

            spiral_data.append({
                "time_s": t,
                "angle_rad": angle,
                "x_m": x,
                "y_m": y,
                "z_m": z,
                "velocity_m_s": velocity,
                "path_length_m": cumulative_path,
                "emf_V": emf,
                "current_A": current,
                "power_W": power,
                "cumulative_energy_J": energy_harvested,
                "ball_mass_kg": ball_mass,
            })

        return spiral_data, energy_harvested

    def rack_pinion_energy(self, spiral_data: List[Dict[str, float]], cfg: RackPinionConfig) -> float:
        velocity = spiral_data[-1]["velocity_m_s"]
        mass = spiral_data[-1]["ball_mass_kg"]
        translational_ke = 0.5 * mass * velocity ** 2
        rotational_ke = (2.0 / 5.0) * mass * (velocity ** 2)
        available_energy = translational_ke + rotational_ke
        mechanical = available_energy * cfg.gear_efficiency
        electrical = mechanical * cfg.generator_efficiency
        return electrical

    def elevator_energy(self, cfg: SpiralConfig, elevator_cfg: ElevatorConfig) -> float:
        mass = cfg.ball_density_kg_m3 * (4.0 / 3.0) * math.pi * (cfg.ball_radius_m ** 3)
        potential = mass * 9.80665 * elevator_cfg.lift_height_m
        electrical_cost = potential / elevator_cfg.motor_efficiency
        return electrical_cost

    def simulate_lattice(self, cfg: LatticeConfig) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        nx, ny, nz = cfg.grid_dims
        total_nodes = nx * ny * nz
        mass = 0.035
        rest = cfg.rest_spacing_m
        positions: List[List[float]] = []
        velocities: List[List[float]] = []

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    positions.append([ix * rest, iy * rest, iz * rest])
                    velocities.append([0.0, 0.0, 0.0])

        max_z = max(p[2] for p in positions)
        compression = cfg.compression_mm / 1000.0
        for p in positions:
            if abs(p[2] - max_z) < 1e-9:
                p[2] -= compression

        connections: List[Tuple[int, int]] = []
        for i in range(total_nodes):
            for j in range(i + 1, total_nodes):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dz = positions[i][2] - positions[j][2]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist <= math.sqrt(2) * rest + 1e-6:
                    connections.append((i, j))

        k_values = [cfg.spring_k for _ in connections]
        lattice_records: List[Dict[str, float]] = []
        decay_events: List[Dict[str, float]] = []
        generator_energy = 0.0
        dt = cfg.dt
        steps = int(cfg.release_time_s / dt)

        for step in range(steps):
            time = step * dt
            forces = [[0.0, 0.0, 0.0] for _ in range(total_nodes)]
            potential_energy = 0.0

            for conn_idx, (i, j) in enumerate(connections):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dz = positions[i][2] - positions[j][2]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist == 0:
                    continue
                direction = [dx / dist, dy / dist, dz / dist]
                stretch = dist - rest

                spring_force = -k_values[conn_idx] * stretch
                repulsive_force = 0.0
                if dist < 1.5 * rest:
                    repulsive_force = cfg.repulsive_strength / (dist ** cfg.repulsive_exponent)

                total_force = spring_force + repulsive_force
                fx = total_force * direction[0]
                fy = total_force * direction[1]
                fz = total_force * direction[2]
                forces[i][0] += fx
                forces[i][1] += fy
                forces[i][2] += fz
                forces[j][0] -= fx
                forces[j][1] -= fy
                forces[j][2] -= fz

                potential_energy += 0.5 * k_values[conn_idx] * stretch * stretch
                if repulsive_force:
                    potential_energy += 0.5 * cfg.repulsive_strength / ((dist + 1e-9) ** (cfg.repulsive_exponent - 1))

                if random.random() < cfg.decay_rate_hz * dt:
                    k_values[conn_idx] *= (1.0 - cfg.decay_reduction)
                    decay_events.append({
                        "time_s": time,
                        "connection": float(conn_idx),
                        "new_k": k_values[conn_idx],
                    })

            kinetic_energy = 0.0
            for idx in range(total_nodes):
                ax = forces[idx][0] / mass - cfg.damping * velocities[idx][0]
                ay = forces[idx][1] / mass - cfg.damping * velocities[idx][1]
                az = forces[idx][2] / mass - cfg.damping * velocities[idx][2]

                velocities[idx][0] += ax * dt
                velocities[idx][1] += ay * dt
                velocities[idx][2] += az * dt

                positions[idx][0] += velocities[idx][0] * dt
                positions[idx][1] += velocities[idx][1] * dt
                positions[idx][2] += velocities[idx][2] * dt

                kinetic_energy += 0.5 * mass * (
                    velocities[idx][0] ** 2 + velocities[idx][1] ** 2 + velocities[idx][2] ** 2
                )

            generator_energy += cfg.generator_coupling * kinetic_energy * dt

            lattice_records.append({
                "time_s": time,
                "potential_energy_J": potential_energy,
                "kinetic_energy_J": kinetic_energy,
                "generator_energy_J": generator_energy,
            })

        return lattice_records, decay_events

    def write_csv(self, path: Path, rows: List[Dict[str, float]]) -> None:
        if not rows:
            path.write_text("")
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def generate_svgs(
        self,
        spiral_data: List[Dict[str, float]],
        lattice_data: List[Dict[str, float]],
        decay_events: List[Dict[str, float]],
        foot_press: FootPressEvent,
        spiral_cfg: SpiralConfig,
    ) -> None:
        width, height = 1200, 900
        svg_path = self.output_dir / "system_overview.svg"

        def project(point: Tuple[float, float, float]) -> Tuple[float, float]:
            ax = math.radians(35)
            ay = math.radians(25)
            cx = math.cos(ax)
            sx = math.sin(ax)
            cy = math.cos(ay)
            sy = math.sin(ay)
            x, y, z = point
            xp = cy * x + sy * z
            zp = -sy * x + cy * z
            yp = cx * y - sx * zp
            return xp, yp

        def norm_values(values: List[float]) -> Tuple[float, float]:
            return min(values), max(values) if values else (0.0, 1.0)

        spiral_points = [(d["x_m"], d["y_m"], d["z_m"]) for d in spiral_data]
        proj_points = [project(p) for p in spiral_points]
        xs = [p[0] for p in proj_points]
        ys = [p[1] for p in proj_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        def map_to_panel(pt: Tuple[float, float], panel: Tuple[int, int, int, int]) -> Tuple[float, float]:
            px, py = pt
            x0, y0, w, h = panel
            x = x0 + (px - min_x) / (max_x - min_x + 1e-9) * w
            y = y0 + h - (py - min_y) / (max_y - min_y + 1e-9) * h
            return x, y

        # Graph scaling
        times = [d["time_s"] for d in spiral_data]
        voltages = [d["emf_V"] for d in spiral_data]
        powers = [d["power_W"] for d in spiral_data]
        energies = [d["cumulative_energy_J"] for d in spiral_data]
        lattice_times = [d["time_s"] for d in lattice_data]
        lattice_potential = [d["potential_energy_J"] for d in lattice_data]
        lattice_generator = [d["generator_energy_J"] for d in lattice_data]

        def line_points(x_data: List[float], y_data: List[float], panel: Tuple[int, int, int, int]) -> List[Tuple[float, float]]:
            if not x_data:
                return []
            x0, y0, w, h = panel
            min_xd, max_xd = min(x_data), max(x_data)
            min_yd, max_yd = min(y_data), max(y_data)
            span_x = max(max_xd - min_xd, 1e-9)
            span_y = max(max_yd - min_yd, 1e-9)
            pts = []
            for x_val, y_val in zip(x_data, y_data):
                px = x0 + (x_val - min_xd) / span_x * w
                py = y0 + h - (y_val - min_yd) / span_y * h
                pts.append((px, py))
            return pts

        with svg_path.open("w") as svg:
            svg.write(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>\n")
            svg.write("<style>text{font-family:Arial,sans-serif;font-size:16px;fill:#111;}" "</style>\n")

            # Panel definitions
            panel_overview = (50, 60, 500, 450)
            panel_volt = (600, 60, 550, 220)
            panel_energy = (600, 340, 550, 170)
            panel_lattice = (50, 560, 1100, 280)

            svg.write("<text x='50' y='35'>Energy Harvesting System Overview</text>\n")
            svg.write(
                f"<text x='55' y='55'>Foot press: {foot_press.force_newtons:.0f} N, deflection {foot_press.deflection_mm:.1f} mm</text>\n"
            )

            # Draw floor tile
            tile_points = [(-0.115, -0.115, 0.01), (0.115, -0.115, 0.01), (0.115, 0.115, 0.01), (-0.115, 0.115, 0.01)]
            proj_tile = [map_to_panel(project(p), panel_overview) for p in tile_points]
            tile_path = " ".join(f"{x:.1f},{y:.1f}" for x, y in proj_tile)
            svg.write(f"<polygon points='{tile_path}' fill='none' stroke='#888' stroke-dasharray='6,4' stroke-width='2'/>\n")

            # Spiral path
            spiral_coords = " ".join(
                f"{map_to_panel(pt, panel_overview)[0]:.1f},{map_to_panel(pt, panel_overview)[1]:.1f}"
                for pt in proj_points
            )
            svg.write(f"<polyline points='{spiral_coords}' fill='none' stroke='#1f77b4' stroke-width='2'/>\n")

            # Ball positions along spiral
            total_pts = len(spiral_points)
            for idx in range(0, total_pts, max(1, total_pts // 12)):
                bx, by = map_to_panel(proj_points[idx], panel_overview)
                svg.write(f"<circle cx='{bx:.1f}' cy='{by:.1f}' r='4' fill='#ff8c00'/>\n")

            # Elevator line
            elevator_bottom = (spiral_cfg.radius_end_m + 0.04, 0.0, spiral_points[-1][2])
            elevator_top = (spiral_cfg.radius_end_m + 0.04, 0.0, 0.02)
            e0 = map_to_panel(project(elevator_bottom), panel_overview)
            e1 = map_to_panel(project(elevator_top), panel_overview)
            svg.write(f"<line x1='{e0[0]:.1f}' y1='{e0[1]:.1f}' x2='{e1[0]:.1f}' y2='{e1[1]:.1f}' stroke='#2ca02c' stroke-width='4'/>\n")
            svg.write(
                f"<text x='{e1[0]:.1f}' y='{e1[1]-10:.1f}' fill='#2ca02c'>Magnetic elevator</text>\n"
            )

            # Lattice block representation
            lattice_rect = (panel_overview[0] + 60, panel_overview[1] + panel_overview[3] - 120, 120, 120)
            svg.write(
                f"<rect x='{lattice_rect[0]}' y='{lattice_rect[1]}' width='{lattice_rect[2]}' height='{lattice_rect[3]}' fill='rgba(128,0,0,0.15)' stroke='#800000' stroke-width='2'/>\n"
            )
            svg.write(
                f"<text x='{lattice_rect[0]}' y='{lattice_rect[1]-10}'>Metamaterial lattice</text>\n"
            )

            # Voltage and power graph
            volt_pts = line_points(times, voltages, panel_volt)
            power_pts = line_points(times, powers, panel_volt)
            svg.write("<text x='600' y='50'>Voltage & Power vs Time</text>\n")
            svg.write(self.svg_polyline(volt_pts, "#1f77b4", 2))
            svg.write(self.svg_polyline(power_pts, "#ff7f0e", 2))
            svg.write("<text x='605' y='80' fill='#1f77b4'>Voltage</text>\n")
            svg.write("<text x='605' y='100' fill='#ff7f0e'>Power</text>\n")

            # Energy graph
            energy_pts = line_points(times, energies, panel_energy)
            svg.write("<text x='600' y='320'>Cumulative Energy Harvested</text>\n")
            svg.write(self.svg_polyline(energy_pts, "#9467bd", 2))

            # Lattice energy and generator capture
            lattice_potential_pts = line_points(lattice_times, lattice_potential, panel_lattice)
            lattice_generator_pts = line_points(lattice_times, lattice_generator, panel_lattice)
            svg.write("<text x='50' y='540'>Lattice Energy Dynamics</text>\n")
            svg.write(self.svg_polyline(lattice_potential_pts, "#d62728", 2))
            svg.write(self.svg_polyline(lattice_generator_pts, "#2ca02c", 2))

            # Decay events markers
            for event in decay_events:
                t = event["time_s"]
                if lattice_times:
                    x0, y0, w, h = panel_lattice
                    min_t, max_t = min(lattice_times), max(lattice_times)
                    span_t = max(max_t - min_t, 1e-9)
                    px = x0 + (t - min_t) / span_t * w
                    py = y0 + 10
                    svg.write(
                        f"<rect x='{px-4:.1f}' y='{py:.1f}' width='8' height='8' fill='#ff0000' stroke='none'/>\n"
                    )

            svg.write("</svg>\n")

        self.figures.append(svg_path)

    def svg_polyline(self, points: List[Tuple[float, float]], color: str, width: int) -> str:
        if not points:
            return ""
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        return f"<polyline points='{coords}' fill='none' stroke='{color}' stroke-width='{width}'/>\n"

    def export_geometry(self, spiral_cfg: SpiralConfig, lattice_cfg: LatticeConfig) -> None:
        spiral_stl = self.output_dir / "spiral_track.stl"
        lattice_stl = self.output_dir / "lattice.stl"
        self.write_spiral_stl(spiral_cfg, spiral_stl)
        self.write_lattice_stl(lattice_cfg, lattice_stl)
        self.stl_paths.extend([spiral_stl, lattice_stl])

    def write_spiral_stl(self, cfg: SpiralConfig, path: Path) -> None:
        total_angle = cfg.turns * 2 * math.pi
        steps = 400
        track_width = 0.015
        track_thickness = 0.004
        vertices: List[Tuple[float, float, float]] = []

        for i in range(steps):
            angle = total_angle * i / (steps - 1)
            radius = cfg.radius_start_m + (cfg.radius_end_m - cfg.radius_start_m) * i / (steps - 1)
            height = -cfg.drop_height_m * i / (steps - 1)
            center = (radius * math.cos(angle), radius * math.sin(angle), height)
            tangent = vec_normalize((-radius * math.sin(angle), radius * math.cos(angle), -cfg.drop_height_m / total_angle))
            normal = vec_normalize((math.cos(angle), math.sin(angle), 0.0))
            binormal = vec_normalize(vec_cross(tangent, normal))
            offset1 = vec_scale(normal, track_width / 2)
            offset2 = vec_scale(normal, -track_width / 2)
            bottom_shift = vec_scale(binormal, -track_thickness)
            vertices.extend([
                vec_add(center, offset1),
                vec_add(center, offset2),
                vec_add(vec_add(center, offset1), bottom_shift),
                vec_add(vec_add(center, offset2), bottom_shift),
            ])

        triangles: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]] = []
        for i in range(0, len(vertices) - 4, 4):
            v0, v1, v2, v3 = vertices[i:i + 4]
            v4, v5, v6, v7 = vertices[i + 4:i + 8]
            triangles.extend([
                (v0, v4, v1), (v1, v4, v5),
                (v1, v5, v3), (v3, v5, v7),
                (v0, v2, v4), (v4, v2, v6),
                (v2, v3, v6), (v3, v7, v6),
            ])

        with path.open("w") as f:
            f.write("solid spiral\n")
            for tri in triangles:
                normal = vec_cross(vec_sub(tri[1], tri[0]), vec_sub(tri[2], tri[0]))
                norm = vec_norm(normal)
                if norm == 0:
                    normal = (0.0, 0.0, 1.0)
                else:
                    normal = (normal[0] / norm, normal[1] / norm, normal[2] / norm)
                f.write(
                    f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n"
                    "    outer loop\n"
                    f"      vertex {tri[0][0]:.6e} {tri[0][1]:.6e} {tri[0][2]:.6e}\n"
                    f"      vertex {tri[1][0]:.6e} {tri[1][1]:.6e} {tri[1][2]:.6e}\n"
                    f"      vertex {tri[2][0]:.6e} {tri[2][1]:.6e} {tri[2][2]:.6e}\n"
                    "    endloop\n  endfacet\n"
                )
            f.write("endsolid spiral\n")

    def write_lattice_stl(self, cfg: LatticeConfig, path: Path) -> None:
        nx, ny, nz = cfg.grid_dims
        spacing = cfg.rest_spacing_m
        strut = spacing * 0.08
        cubes: List[List[Tuple[float, float, float]]] = []

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    cx = ix * spacing
                    cy = iy * spacing
                    cz = iz * spacing
                    d = strut / 2
                    cube = [
                        (cx + d, cy + d, cz + d),
                        (cx + d, cy - d, cz + d),
                        (cx - d, cy - d, cz + d),
                        (cx - d, cy + d, cz + d),
                        (cx + d, cy + d, cz - d),
                        (cx + d, cy - d, cz - d),
                        (cx - d, cy - d, cz - d),
                        (cx - d, cy + d, cz - d),
                    ]
                    cubes.append(cube)

        faces = [
            (0, 1, 2), (0, 2, 3),
            (4, 7, 6), (4, 6, 5),
            (0, 4, 5), (0, 5, 1),
            (1, 5, 6), (1, 6, 2),
            (2, 6, 7), (2, 7, 3),
            (3, 7, 4), (3, 4, 0),
        ]

        with path.open("w") as f:
            f.write("solid lattice\n")
            for cube in cubes:
                for face in faces:
                    v0, v1, v2 = cube[face[0]], cube[face[1]], cube[face[2]]
                    normal = vec_cross(vec_sub(v1, v0), vec_sub(v2, v0))
                    norm = vec_norm(normal)
                    if norm == 0:
                        normal = (0.0, 0.0, 1.0)
                    else:
                        normal = (normal[0] / norm, normal[1] / norm, normal[2] / norm)
                    f.write(
                        f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n"
                        "    outer loop\n"
                        f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n"
                        f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n"
                        f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n"
                        "    endloop\n  endfacet\n"
                    )
            f.write("endsolid lattice\n")


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "outputs"
    sim = EnergyHarvesterSimulation(output_dir)
    outputs = sim.run()
    print(f"Total harvested energy (J): {outputs.total_harvested_j:.6f}")
    print(f"Elevator energy cost (J): {outputs.elevator_energy_cost_j:.6f}")
    net = outputs.total_harvested_j - outputs.elevator_energy_cost_j
    print(f"Net cycle energy (J): {net:.6f}")
    print("Figures:")
    for fig in outputs.figures:
        print(f" - {fig}")
    print("STL files:")
    for stl in outputs.stl_paths:
        print(f" - {stl}")
    print("CSV logs:")
    for csv in outputs.csv_paths:
        print(f" - {csv}")


if __name__ == "__main__":
    main()
