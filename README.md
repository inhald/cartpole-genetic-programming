# Cartpole Genetic Programming

This project implements a simple **genetic programming** algorithm to solve the classic **CartPole balancing problem**. The goal is to evolve programs that can control a cart and keep a pole balanced vertically for as long as possible.

---

![Cartpole Animation](cartpole_animation.gif)

*Visualization of a successful cartpole balance using an evolved controller.*

---

## Files

- **CartpoleEvolution.cpp**  
  Main source file containing the genetic programming logic and simulation loop.

- **cartCentering.h**  
  Header file containing constants or helper functions used in the simulation.

- **cartpole_animation.gif**  
  A visual demonstration of the balancing behavior.

## Getting Started

### Prerequisites

- A C++ compiler (e.g., `g++`)

### Compile

```bash
g++ -O2 CartpoleEvolution.cpp -o cartpole_sim
```

### Run
```bash
./cartpole_sim
```
