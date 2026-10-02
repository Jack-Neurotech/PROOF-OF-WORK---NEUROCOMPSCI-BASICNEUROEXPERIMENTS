# PROOF-OF-WORK---NEUROCOMPSCI-BASICNEUROEXPERIMENTS

# 1. What Is the Project and What Does It Do?

* **Project:** Basic Computational Neuroscience Experiments

* **Purpose:** A Python-based computational neuroscience project designed to build and test increasingly complex models of neuronal behavior, synaptic communication, neural networks, and learning.

* **Primary goal:** Use mathematical models and Python simulations to understand how individual neurons generate electrical activity, how neurons communicate through synapses, how activity propagates through networks, and how synaptic connections change through learning.

* **The project is organized into progressively more complex experimental stages:**

  * Passive membrane dynamics
  * Leaky Integrate-and-Fire neuron models
  * Synaptic transmission
  * Multi-neuron networks
  * Synaptic plasticity
  * Repeated learning experiments
  * Network-level learning

* **The project contains multiple experimental branches:**

  * `main`
  * `Computational-Neuron-Simulator---BROADER-SIMULATON`
  * `Synaptic-Communication`

* **The Computational Neuron Simulator branch contains experiments involving:**

  * Passive membrane behavior.
  * Current injection and membrane response.
  * Membrane time constants.
  * Numerical convergence.
  * Leaky Integrate-and-Fire neurons.
  * Subthreshold membrane behavior.
  * Threshold-based spiking.
  * Refractory periods.
  * Firing-rate measurements.
  * Synaptic transmission.
  * Multi-neuron networks.
  * Synaptic propagation through networks.

* **The passive membrane experiments investigate:**

  * How a neuron's membrane voltage responds to input current.
  * How membrane voltage changes over time.
  * How quickly the membrane approaches a new voltage.
  * How the simulation behaves as numerical resolution changes.

* **The Leaky Integrate-and-Fire experiments investigate:**

  * Subthreshold membrane integration.
  * Threshold detection.
  * Spike generation.
  * Voltage reset.
  * Refractory behavior.
  * Firing rate as input changes.

* **The synaptic experiments investigate:**

  * How an action potential can produce a synaptic signal.
  * How synaptic input can influence another neuron.
  * How signals are represented numerically.
  * How synaptic effects can be incorporated into neuron simulations.

* **The network experiments investigate:**

  * Multiple neurons operating together.
  * Excitatory feed-forward connectivity.
  * Synaptic current propagation.
  * Spike propagation through connected neurons.
  * Network activity.
  * Whether activity in one neuron produces activity in downstream neurons.

* **The Synaptic Communication branch extends the project into synaptic learning and plasticity.**

* **The branch contains experiments investigating:**

  * STDP timing relationships.
  * Synaptic weight changes.
  * Weight dynamics.
  * Competing synapses.
  * Network-level synaptic weights.
  * Repeated learning.
  * Emergent network behavior.

* **Experiment 001 investigates STDP timing:**

  * Measures how the timing difference between pre-synaptic and post-synaptic spikes affects synaptic weight change.
  * Produces an STDP timing curve.
  * Tests both potentiation and depression.

* **Experiment 002 investigates synaptic weight learning:**

  * Repeatedly applies spike-timing relationships.
  * Measures how synaptic strength changes across learning.
  * Stores the resulting weight trajectory.

* **Experiment 003 investigates weight dynamics and timing:**

  * Examines the relationship between spike timing and changing synaptic strength.
  * Produces timing and weight-dynamics measurements.

* **Experiment 004 investigates competing synapses:**

  * Examines multiple synaptic connections.
  * Measures differences between synaptic weight changes.
  * Demonstrates that separate synapses can experience different learning dynamics.

* **Experiment 005 investigates network weights:**

  * Extends synaptic learning to multiple network connections.
  * Tracks how network-level synaptic weights change.
  * Produces final and network-wide weight measurements.

* **Experiment 006 investigates repeated network learning:**

  * Repeatedly runs the neural network.
  * Applies synaptic plasticity after each trial.
  * Tracks synaptic weights across trials.
  * Measures changes in spike timing.
  * Examines the resulting network behavior.

* **The project therefore progresses from:**

  * Individual membrane
  * → Individual neuron
  * → Spike generation
  * → Synaptic transmission
  * → Multiple neurons
  * → Network propagation
  * → Synaptic plasticity
  * → Repeated learning
  * → Network-level learning

* **In simple terms:**

  * **Mathematical equations → neuron simulation → spikes → synaptic communication → neural networks → synaptic learning**

* **What the project demonstrates:**

  * Python programming
  * Scientific computing
  * Numerical simulation
  * Computational neuroscience
  * Differential-equation-based modeling
  * Neuron modeling
  * Spike generation
  * Synaptic communication
  * Neural networks
  * Synaptic plasticity
  * Spike-timing-dependent plasticity
  * Experimental data generation
  * Data visualization
  * Reproducible computational experiments

---

# 2. Python Principles, General Structure, Packages, and Associated Principles

# A. Python Principles and the General Structure

The project uses fundamental Python programming concepts to turn mathematical descriptions of neurons and synapses into executable computational experiments.

### Variables

* Variables store numerical parameters and experimental results.

```python
dt = 0.1

duration = 500.0

threshold = -55.0
```

These variables can represent:

* Time step.
* Simulation duration.
* Membrane threshold.
* Current.
* Voltage.
* Synaptic weight.
* Spike time.

---

### Data Types

The experiments use several basic Python data types.

* **Numbers** — voltages, currents, times, weights, and firing rates.
* **Strings** — experiment names and labels.
* **Lists** — collections of spike times or weight histories.
* **Arrays** — large collections of numerical measurements.
* **Dictionaries** — structured experimental results.
* **Objects** — neurons, synapses, and other computational components.

Example:

```python
weight = 0.5
```

The variable contains a numerical synaptic weight.

---

### Lists

Lists allow multiple values to be stored together.

```python
spike_times = []
```

Spike times can then be added to the list as the simulation progresses.

A learning experiment can similarly maintain a history:

```python
weight_history = []
```

This allows the experiment to track how synaptic strength changes over repeated trials.

---

### Indexing

Indexing allows a specific value to be retrieved from a collection.

```python
spike_times[0]
```

This accesses the first recorded spike time.

Indexing is useful for examining:

* Individual spikes.
* Individual trials.
* Individual neurons.
* Individual measurements.

---

### Loops

Loops allow the same computational experiment to be repeated.

```python
for trial in range(20):
    run_trial()
```

This is particularly important for learning experiments because synaptic plasticity is applied repeatedly.

The result is a sequence such as:

```text
Trial 1
Trial 2
Trial 3
...
Trial 20
```

---

### Functions

Functions divide the experiments into reusable computational operations.

```python
def simulate_neuron(...):
    ...
```

A function can:

1. Receive parameters.
2. Perform calculations.
3. Produce results.

The general computational structure is:

**Input → Calculation → Output**

This allows the same neuron or synaptic model to be tested under different conditions.

---

### Conditional Statements

Conditional statements allow the simulation to make decisions.

For example, a neuron can determine whether its voltage has reached threshold.

```python
if voltage >= threshold:
    spike = True
```

This allows the mathematical model to produce discrete events such as action potentials.

---

### Boolean Logic

Boolean logic allows the program to determine whether conditions are true or false.

Examples include:

```python
voltage >= threshold
```

or:

```python
spike_detected and synaptic_connection_exists
```

Boolean conditions are used to control events inside the simulations.

---

### Arrays

Arrays store large collections of numerical measurements.

A simulation can store:

```text
time
voltage
current
spike state
synaptic weight
```

across thousands of simulation steps.

This makes it possible to analyze the complete trajectory of the simulated neuron.

---

### Numerical Integration

Numerical integration is used to approximate how a neuron's voltage changes over time.

Instead of solving the mathematical equation symbolically at every point, the simulation advances through small time steps.

Conceptually:

```text
Current voltage
      ↓
Calculate voltage change
      ↓
Advance time
      ↓
New voltage
      ↓
Repeat
```

This is fundamental to computational neuron simulation.

---

### Object-Oriented Programming

Object-oriented programming allows the project to represent biological components as computational objects.

A neuron can be represented as an object containing:

* Membrane parameters.
* Voltage.
* Threshold.
* Spike state.
* Simulation methods.

A synapse can similarly contain:

* Pre-synaptic neuron.
* Post-synaptic neuron.
* Synaptic weight.
* Plasticity parameters.

This allows biological components to become computational components.

---

### Modular Programming

The project is divided into separate experiments and components rather than putting every model into one Python file.

Examples include:

```text
001_passive_membrane
002_lif
003_synapses
004_networks
Synaptic COMMUNICATION
```

This organization allows individual concepts to be developed and tested independently.

---

### Data Flow

The computational experiments follow a progression from mathematical parameters to simulated measurements.

```text
Model parameters
      ↓
Mathematical calculation
      ↓
Numerical simulation
      ↓
Voltage / spike data
      ↓
Synaptic interaction
      ↓
Network activity
      ↓
Learning
      ↓
Results
```

---

### File Input and Output

The experiments generate structured results.

The repository contains experimental data files such as:

```text
.csv
```

and visualization files such as:

```text
.png
```

This separates the computational experiment from the resulting scientific measurements.

---

### Data Recording

The simulations record variables throughout the experiment instead of only storing the final result.

Examples include:

* Voltage history.
* Spike times.
* Synaptic weights.
* Weight changes.
* Timing relationships.
* Network activity.

This makes it possible to analyze how the system changes over time.

---

### Reproducible Experiments

Each experiment is organized as a defined computational procedure.

The same:

* Parameters
* Equations
* Time step
* Network structure
* Learning rule

can be run again to reproduce the experiment.

This is an important principle of computational neuroscience.

---

# B. Packages and the Python Principles Associated With Them

## NumPy

**Purpose:**

* Numerical computing.
* Numerical arrays.
* Mathematical calculations.
* Simulation data.

### Python principles associated with NumPy

* Variables
* Arrays
* Indexing
* Mathematical operations
* Functions
* Iteration

NumPy provides the numerical foundation needed to simulate voltage, current, spike timing, and synaptic weights.

---

## Matplotlib

**Purpose:**

* Visualization of simulation results.
* Plotting membrane voltage.
* Plotting firing rates.
* Plotting synaptic weights.
* Plotting timing relationships.
* Visualizing learning behavior.

### Python principles associated with Matplotlib

* Functions
* Objects
* Methods
* Variables
* Arrays

The general relationship is:

**Simulation data → Matplotlib → Scientific visualization**

---

## Pandas

**Purpose:**

* Organizing experimental results into tabular data.
* Saving numerical results as CSV files.
* Reading and analyzing experimental datasets.

### Python principles associated with Pandas

* Objects
* Tables
* Data structures
* Functions
* Indexing
* File input/output

Pandas is particularly useful when simulation results need to be stored and analyzed outside the original Python execution.

---

## Python Standard Library

The experiments also use functionality provided by Python itself rather than requiring a specialized neuroscience package.

This includes concepts such as:

* Loops
* Functions
* Lists
* Dictionaries
* Conditional statements
* File operations
* Numerical control
* Program execution

This demonstrates that computational neuroscience models can be constructed from fundamental Python programming principles combined with scientific packages.

---

# How the Python Structure Supports the Neuroscience

The project connects programming concepts directly to biological mechanisms.

**Python variable**

→ represents a biological parameter.

**Function**

→ represents a computational process.

**Loop**

→ represents repeated simulation steps or experimental trials.

**Conditional**

→ represents a biological event such as reaching threshold.

**Array**

→ represents measurements across time.

**Object**

→ represents a computational neuron or synapse.

**Network**

→ represents interactions between multiple computational neurons.

**Weight**

→ represents the strength of a synaptic connection.

**Learning rule**

→ changes the synaptic weight according to spike timing.

The result is a computational representation of progressively more complex neural systems.

---

# Overall Project Structure

The project progresses through several levels of computational neuroscience:

**Level 1 — Passive Membrane**

* Current response
* Membrane voltage
* Time constant
* Numerical convergence

↓

**Level 2 — Leaky Integrate-and-Fire**

* Subthreshold integration
* Threshold
* Spiking
* Reset
* Refractory period
* Firing rate

↓

**Level 3 — Synapses**

* Synaptic transmission
* Synaptic input
* Communication between neurons

↓

**Level 4 — Networks**

* Multiple neurons
* Connectivity
* Synaptic currents
* Spike propagation
* Network activity

↓

**Level 5 — Synaptic Plasticity**

* Spike timing
* Potentiation
* Depression
* Synaptic weight changes

↓

**Level 6 — Network Learning**

* Repeated trials
* Changing synaptic weights
* Changing spike timing
* Network-level learning
* Emergent behavior

---

# Scientific Concept Being Demonstrated

The central idea of the repository is that increasingly complex neural behavior can be constructed from simpler computational components.

```text
Membrane dynamics
        ↓
Neuron dynamics
        ↓
Action potentials
        ↓
Synaptic transmission
        ↓
Neural networks
        ↓
Synaptic plasticity
        ↓
Learning
```

The repository therefore functions as a progression from **basic mathematical neuron models toward computational models of neural communication and learning**.
