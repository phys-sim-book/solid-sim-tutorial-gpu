# MUDA-based Solid Simulatoion Tutorial
This is a tutorial for elastodynamic contact simulation using [MUDA](https://github.com/MuGdxy/muda) (a [CUDA](https://developer.nvidia.com/cuda-toolkit) programming paradigm).

The basic architecture of the simulators follows [@liminchen](https://github.com/liminchen)'s Numpy version [solid-sim-tutorial](https://github.com/phys-sim-book/solid-sim-tutorial).

The tutorial (written by a beginner of simulation) aims at helping beginners learn how to write simple CUDA codes for efficient solid simulations on the GPU.

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

## Examples

### 1. Simple Mass-Spring System
![Simple Mass-Spring System](./img/1.png)
### 2. Dirichlet Boundary Condition
![Dirichlet Boundary Condition](./img/2.png)
### 3. Contact
![Contact](./img/3.png)
### 4. Friction
![Friction](./img/4.png)
### 5. Moving Dirichlet Boundary Condition
![Moving Dirichlet Boundary](./img/5.png)
### 6. Neohookean Solids
![Neohookean Model](./img/6.png)
### 7. Neohookean Solids with Self-Contact
![Neohookean Model with Self Collision](./img/7.png)
### 8. Neohookean Solids with Frictional Self-Contact
![Neohookean Model with Self  Friction](./img/8.png)
