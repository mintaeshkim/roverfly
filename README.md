# Quadrotor Trajectory Tracking

This repository provides a reinforcement learning (RL) framework for quadrotor trajectory tracking. The framework is built using MuJoCo and Stable Baselines3. It supports curriculum learning, domain randomization, and sim-to-real gap reduction techniques. It is designed for training policies that can generalize to real-world applications.

## Features

### 1. Quadrotor Trajectory Tracking
- The environment is designed for training reinforcement learning policies to track reference trajectories with a quadrotor.
- Supports different trajectory types, including setpoint stabilization and continuous curve tracking.
- Includes a reward function that encourages accurate position and velocity tracking.

### 2. Curriculum Learning
- The environment dynamically adjusts trajectory difficulty based on agent performance.
- Starts with simple setpoint stabilization and gradually progresses to complex curved trajectories.
- The difficulty is updated using historical performance metrics.

### 3. Sim-to-Real Gap & Domain Randomization
- Implements **domain randomization** to improve policy generalization:
  - Random perturbations in quadrotor mass and inertia.
  - Random initial position and velocity offsets.
  - Actuator delay modeling.
- Reduces the **sim-to-real gap** by incorporating realistic noise and delays.

### 4. Observation Space
- Includes **state history** (past observations) for improved temporal understanding.
- Uses a **high-dimensional observation vector (o_dim = 198)** consisting of:
  - Current quadrotor state (position, velocity, orientation, angular velocity)
  - Past observations from a rolling buffer
  - Future reference trajectory points for predictive tracking

### 5. Action Space
- Control inputs are represented as **Collective Thrust and Body Rates (CTBR)**:
  - **Action space:** `[Thrust, Roll Rate, Pitch Rate, Yaw Rate]` in normalized range.
  - Supports bounded and unbounded action modes.
- Actions are processed through a **low-level PID controller** to generate rotor thrust commands.

### 6. Low-Level Controller
- The environment includes a **PID-based low-level controller** for attitude stabilization.
- Converts **body rate commands** into **individual rotor thrusts** using the quadrotor's control allocation matrix.
- Includes **actuator dynamics** (rise and fall time constants) for realistic motor response.

## Environment Structure

### **State Representation**
The environment maintains a structured observation space with history buffers and predictive information:
```
Observation = [
    Current state (position, velocity, orientation, angular velocity),
    Feedback (position, velocity),
    Prior action,
    I/O history (position, velocity, actions),
    Future reference trajectory points
]
```

### **Action Representation**
```
Action = [Thrust, Roll Rate, Pitch Rate, Yaw Rate]
```
- If `is_action_bound = True`: Thrust values are constrained between `[0.2, 0.8]`
- If `is_action_bound = False`: Actions are in `[-1, 1]` normalized range

## Installation & Setup
### **Requirements**
- MuJoCo
- NumPy
- Gymnasium

### **Setup Instructions**
```sh
pip install -r requirements.txt
```

## **Citing**

```

```