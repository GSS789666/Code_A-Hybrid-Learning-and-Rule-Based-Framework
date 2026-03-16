# MRTA-Benchmark dataset: 250K Optimal Multi-Robot Task Allocation Instances with Heterogeneous Robots, Precedence Constraints & Dynamic Coalitions

**Author**: Jakob Bichler  
**Contact**: j.d.bichler@gmail.com  
**Institution**: TU Delft – MSc Robotics, Autonomous Multi-Robot Lab  
**License**: CC BY 4.0  

---

## 📦 General Introduction

This dataset contains 250,000 optimally solved instances of multi-robot task assignment problems with heterogeneous robots and dynamic coalition formation. It was created as part of the Sadcher scheduling framework, developed during a master thesis at TU Delft.

Each problem instance specifies task requirements, robot capabilities, spatial task layouts, precedence constraints, and travel/execution times. Each corresponding solution file describes the optimal schedule minimizing the makespan, generated using a MILP solver.

---

## 📁 File Structure and Naming Convention

- `problems/`: JSON files with input data for each instance.  
- `solutions/`: JSON files with corresponding optimal task schedules.

Filenames are indexed numerically and match across folders, e.g.:

```
problems/instance_000001.json
solutions/instance_000001.json
```

---

## 🧪 Methodological Details

- **Instance generation**: Randomized task/robot features sampled under constraints to ensure full skill coverage and valid scheduling.
- **Solving method**: Mixed Integer Linear Programming (MILP) using PuLP and CBC solver.
- **Resources**: Dataset generated using DelftBlue HPC cluster (48 cores).
- **Software**:
  - Python 3.9+
  - PuLP (MILP modeling)
  - CBC solver
  - NumPy, Matplotlib (visualization)

---

## 📊 File Format Description

### 🔹 Problem Instance (`problems/`)
JSON object with:
- `Q`: Robot skill matrix, shape `(N x S)`, binary.
- `R`: Task requirement matrix, shape `((M+2) x S)`, binary, includes start/end dummy tasks.
- `T_e`: Execution times per task, length `M+2`.
- `T_t`: Travel time matrix, shape `(M+2 x M+2)`, symmetric.
- `task_locations`: List of `[x, y]` coordinates, length `M+2`.
- `precedence_constraints`: List of `[i, j]` pairs enforcing task `i` before `j`.

### 🔹 Solution File (`solutions/`)
JSON object with:
- `makespan`: Total schedule duration.
- `n_tasks`, `n_robots`: Problem size.
- `robot_schedules`: Dictionary mapping robot IDs to:
  - `task`: Index of the assigned task.
  - `start_time`, `end_time`: Scheduled time window.

---

## 🧮 Units and Conventions

- **Distances**: Euclidean; unitless (assumed meters if needed).
- **Times**: Arbitrary unit (can be scaled); consistent across execution and travel times.
- **Skills**: Binary vectors; `1` indicates possession or requirement.
- **Missing data**: Not applicable—instances are fully specified and validated.

---

## 🔐 License

This dataset is released under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/). You are free to use, adapt, and distribute it with proper attribution.

---

For questions or suggestions, please contact: j.d.bichler@gmail.com
