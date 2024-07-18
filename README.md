# Muda-based Solid Simulatoion Tutorial
This is a tutorial for the Solid Simulation using Muda (a CUDA programming paradigm https://github.com/MuGdxy/muda).

The basic architecture of the simulators mimics Minchen Li's Solid-Sim-Tutorial(https://github.com/phys-sim-book/solid-sim-tutorial). 

The tutorial (which is also written by a beginner) may provide some help for the beginners to learn how to write simple CUDA codes for implicit solid simulation.
## Usage
1. Clone the repository
```bash
git clone https://github.com/Roushelfy/solid-sim-muda
cd solid-sim-muda
git submodule update --init --recursive
```

2. build with cmake
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Requirements
Eigen3==3.4.0

CMake>=3.29

CUDA>=11.0

## Simulators

### 1. Simple Mass-Spring System
![Simple Mass-Spring System](./img/1.png)
### 2. Dirichlet Boundary Condition
![Dirichlet Boundary Condition](./img/2.png)
### 3. Contact
![Contact](./img/3.png)
### 4. Friction
![Friction](./img/4.png)
### 5. Moving Dirichlet Boundary
![Moving Dirichlet Boundary](./img/5.png)
### 6. Neohookean Model
![Neohookean Model](./img/6.png)
### 7. Neohookean Model with Self Collision
![Neohookean Model with Self Collision](./img/7.png)
### 8. Neohookean Model with Self  Friction
![Neohookean Model with Self  Friction](./img/8.png)