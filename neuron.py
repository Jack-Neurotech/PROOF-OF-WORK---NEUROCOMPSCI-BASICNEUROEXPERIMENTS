# ==========================================
# NEURON SIMULATION
# DETERMINISTIC PIECEWISE ACTION POTENTIAL
# ==========================================

import matplotlib.pyplot as plt


# ==========================================
# MEMBRANE PARAMETERS
# ==========================================

E_L = -70.0
V_rest = -70.0
V_threshold = -55.0
V_peak = 30.0


# ==========================================
# CURRENT PARAMETERS
# ==========================================

R = 10.0
tau_m = 10.0
C_m = 1.0


# ==========================================
# SIMULATION PARAMETERS
# ==========================================

dt = 0.01
simulation_time = 50.0


# ==========================================
# INITIAL CONDITIONS
# ==========================================

V = V_rest
t = 0.0


# ==========================================
# STORE RESULTS
# ==========================================

times = []
voltages = []


# ==========================================
# SIMULATION LOOP
# ==========================================

while t <= simulation_time:

    # ======================================
    # PHASE 1
    # SUBTHRESHOLD
    # ======================================

    if t < 20.0:

        I_input = 1.5

        dVdt = (
            -(V - E_L)
            + R * I_input
        ) / tau_m

        V = V + dVdt * dt


    # ======================================
    # PHASE 2
    # DEPOLARIZATION
    # ======================================

    elif t < 21.0:

        I_Na = 100.0

        V = V + (
            I_Na / C_m
        ) * dt


        # Prevent overshooting the peak
        if V > V_peak:
            V = V_peak


    # ======================================
    # PHASE 3
    # PEAK
    # ======================================

    elif t < 21.5:

        I_Na = 0.0
        I_K = 0.0

        V = V + (
            (I_Na + I_K) / C_m
        ) * dt


        V = V_peak


    # ======================================
    # PHASE 4
    # REPOLARIZATION
    # ======================================

    elif t < 23.0:

        I_Na = 0.0
        I_K = -70.0

        V = V + (
            I_K / C_m
        ) * dt


        if V < -70.0:
            V = -70.0


    # ======================================
    # PHASE 5
    # HYPERPOLARIZATION
    # ======================================

    elif t < 28.0:

        I_Na = 0.0
        I_K = -1.0

        V = V + (
            I_K / C_m
        ) * dt


        if V < -75.0:
            V = -75.0


    # ======================================
    # PHASE 6
    # RETURN TO BASELINE
    # ======================================

    else:

        I_input = 0.0

        dVdt = (
            -(V - E_L)
            + R * I_input
        ) / tau_m

        V = V + dVdt * dt


    # ======================================
    # SAVE CURRENT STATE
    # ======================================

    times.append(t)
    voltages.append(V)


    # ======================================
    # ADVANCE TIME
    # ======================================

    t += dt


# ==========================================
# GRAPH
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    times,
    voltages,
    label="Membrane Voltage"
)


# ==========================================
# REFERENCE LINES
# ==========================================

plt.axhline(
    V_threshold,
    linestyle="--",
    label="Threshold"
)

plt.axhline(
    V_rest,
    linestyle="--",
    label="Resting Potential"
)


# ==========================================
# GRAPH LABELS
# ==========================================

plt.xlabel("Time (ms)")
plt.ylabel("Membrane Voltage (mV)")

plt.title(
    "Deterministic Piecewise Action Potential"
)

plt.legend()
plt.grid(True)

plt.show()
