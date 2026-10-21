### T1_parameters

Baseline parameters (units of wake wavelength and shedding period) and trout scaling (Beal et al. 2006: lambda = 20.8 cm, f = 2.74 Hz, L = 17.8 cm).

| quantity | model | trout |
|---|---|---|
| wake wavelength lambda | 1 | 20.8 cm |
| shedding period T | 1 | 0.365 s |
| free stream U_inf | 1.149 | 0.655 m/s (3.68 L/s) |
| velocity deficit delta_0, decay l_delta | 1.3, 0.9 | - |
| vortex velocity W_0, rise x_r, decay l_w | 0.35, 0.35, 1.5 | - |
| lift c_L, drag c_D, lift-to-pitch sigma | 3.0, 0.15, 8.0 | - |
| body natural frequency Omega/omega_w | 1.2 | 3.29 Hz |
| body damping ratio zeta | 0.2 | - |
| CPG amplitude rate mu, shear c, gain g | 2 pi, 0, 16 | - |
| holding position (stable, continuation) | 0.50 to 1.21 | 0.58 to 1.41 L |

### T2_index_verification

Verification of the index condition sgn det(I - M_int) sgn(-a_L) > 0. Branch identity by homotopy in all rows; rows (12) use a fine homotopy and a_L fitted over x_L +/- 0.005.

| source | stable orbits | violations | index < 0 | of which stable | node: a_L<0 | saddle: a_L>0 |
|---|---|---|---|---|---|---|
| parameter sets (08) | 465 | 0 | 486 | 0 | 62/62 | 403/403 |
| model structures (09) | 921 | 0 | 737 | 0 | 127/127 | 794/794 |
| continued branches (10) | 2119 | 0 | - | - | 484/530 | 1635/1685 |
| re-evaluated 08 (12) | 455 | 0 | 66 | 0 | - | - |
| re-evaluated 09 (12) | 913 | 0 | 92 | 0 | - | - |

Violations exclude marginal cases |a_L| < 0.02; there were none. Rows (10) have no unstable orbits (only stable points were classified).

### T3_model_structures

Robustness to model structure (Script 09): stable station-holding orbits found, of which tested on the saddle and node branches, index violations, and the ranges where stable holding occurs.

| variant | states | stable orbits | tested saddle | tested node | violations | eps | omega_c/omega_w | x/lambda |
|---|---|---|---|---|---|---|---|---|
| baseline | 6 | 27 | 22 | 1 | 0 | 0.75 to 4.75 | 1.020 to 1.133 | 0.54 to 1.10 |
| linear lift + lever arm | 6 | 163 | 103 | 0 | 0 | 0.75 to 4.75 | 1.010 to 1.120 | 0.25 to 1.47 |
| added reactive thrust | 6 | 73 | 72 | 0 | 0 | 0.75 to 4.75 | 1.020 to 1.128 | 1.24 to 1.70 |
| reactive thrust only | 6 | 0 | 0 | 0 | 0 | n/a | n/a | n/a |
| lift lag | 7 | 39 | 11 | 0 | 0 | 0.75 to 4.75 | 1.015 to 1.113 | 0.53 to 1.12 |
| lift lag + added mass | 7 | 10 | 4 | 0 | 0 | 0.75 to 2.25 | 1.015 to 1.045 | 0.53 to 1.07 |
| pitch + free heave | 8 | 27 | 22 | 0 | 0 | 0.75 to 4.25 | 1.020 to 1.120 | 0.52 to 1.03 |
| pitch + resonant heave | 8 | 0 | 0 | 0 | 0 | n/a | n/a | n/a |
| pitch + stiff heave | 8 | 22 | 17 | 1 | 0 | 0.75 to 4.25 | 1.020 to 1.123 | 0.53 to 1.05 |
| two-mode body | 8 | 36 | 31 | 3 | 0 | 0.75 to 4.75 | 1.015 to 1.145 | 0.53 to 1.16 |
| van der Pol CPG | 6 | 1 | 1 | 0 | 0 | 0.75 to 0.75 | 1.100 to 1.100 | 0.55 to 0.55 |
| lateral line + proprio | 6 | 7 | 6 | 0 | 0 | 0.25 to 1.25 | 1.010 to 1.065 | 0.54 to 0.96 |
| lateral line only | 6 | 9 | 9 | 0 | 0 | 0.25 to 3.25 | 0.977 to 1.057 | 0.56 to 1.17 |
| plateau deficit | 6 | 10 | 3 | 3 | 0 | 0.75 to 4.25 | 1.020 to 1.115 | 0.79 to 0.97 |
| algebraic decay | 6 | 74 | 74 | 0 | 0 | 0.75 to 4.75 | 1.020 to 1.175 | 1.21 to 2.51 |
| bump envelope | 6 | 7 | 7 | 0 | 0 | 0.75 to 3.75 | 1.020 to 1.100 | 0.69 to 0.88 |
| joint LHS (20 samples, 20 with holding) | 6 | 705 | 412 | 119 | 0 | - | - | - |

### T4_body_tuning

Body-tuning window (Script 06): stable station-holding orbits by body stiffness and damping (in brackets: on the node branch).

| Omega/omega_w | zeta = 0.1 | zeta = 0.2 | zeta = 0.4 |
|---|---|---|---|
| 0.8 | - | 0 (0) | - |
| 0.9 | - | 0 (0) | - |
| 1.0 | - | 5 (3) | - |
| 1.1 | 62 (61) | 19 (0) | - |
| 1.2 | 58 (0) | 26 (0) | 6 (0) |
| 1.3 | 42 (1) | 24 (0) | - |
| 1.4 | 18 (0) | 18 (1) | - |
| 1.5 | 5 (0) | 2 (0) | - |
| 1.6 | - | 2 (0) | - |

### T5_continuation

Stable station-holding area in the (omega_c/omega_w, eps) plane by neural delay and fixed-position branch (Script 10). Ends of stable segments: torus 224, domain: closed-loop 120, fold 178; all 282 half-branches close into loops.

| delay | total | saddle only | node only | no lock only | node and saddle | node and no lock |
|---|---|---|---|---|---|---|
| 0.000 | 0.0522 | 0.0522 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1.571 | 0.0537 | 0.0281 | 0.0003 | 0.0000 | 0.0169 | 0.0206 |
| 3.142 | 0 | - | - | - | - | - |
| 4.712 | 0 | - | - | - | - | - |

### T6_bouts

Station-holding bouts under noise (Script 11): median bout in shedding periods (fraction still holding after 2000 periods), starting on stable orbits; D is the phase diffusion per period (wake amplitude noise has standard deviation sqrt(D)).

| noise | delay 0, saddle | delay pi/2, saddle | delay pi/2, node |
|---|---|---|---|
| joint D=0.003 | 42 (0.00) | 68 (0.00) | 78 (0.00) |
| joint D=0.01 | 26 (0.00) | 26 (0.00) | 32 (0.00) |
| joint D=0.03 | 19 (0.00) | 17 (0.00) | 20 (0.00) |
| joint D=0.1 | 14 (0.00) | 12 (0.00) | 15 (0.00) |
| wake phase only D=0.01 | 33 (0.00) | 50 (0.00) | 54 (0.00) |
| wake amplitude only D=0.01 | 62 (0.10) | > 2000 (0.51) | 333 (0.13) |
| CPG phase only D=0.01 | 36 (0.00) | 39 (0.00) | 52 (0.00) |
| exponent of median vs D | -0.32 | -0.49 | -0.47 |

### T7_lateral_line

Lateral-line removal (Script 11): fish holding station with lateral-line plus proprioceptive input, then lateral line switched off for 400 periods.

| quantity | value |
|---|---|
| runs holding with lateral line | 96 |
| still holding after removal | 22 (23%) |
| tail-beat amplitude ratio, off/on (median) | 1.03 |
| tail-beat phase shift relative to wake (median, rad) | 0.60 |
| frequency locked before / after | 1.00 / 1.00 |

### T8_convergence

Numerical convergence (Script 12).

| check | result |
|---|---|
| points (stratified) | 500 |
| max change of max|mu|, 100 to 400 steps | 2.6e-04 |
| median change of max|mu| | 1.7e-06 |
| stability flips | 0 |
| sign flips: det(I - M), det(I - M_int), a_L, index | 0, 0, 0, 0 |
| branch-type changes | 1 |
| exact vs finite-difference multipliers, step 1e-05 (max) | 7.0e-05 |
| exact vs finite-difference multipliers, step 1e-06 (max) | 7.0e-06 |
| exact vs finite-difference multipliers, step 1e-07 (max) | 7.1e-07 |
| multiplier change, Newton 15 vs 40 iterations (max) | 2.9e-13 |

