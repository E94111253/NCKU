# Ultimate Tic Tac Toe on PYNQ-Z2

This project implements an **Ultimate Tic Tac Toe** game using **Verilog** on the **PYNQ-Z2 FPGA board**.

The system uses push buttons as input and an **8×8 bi-color LED matrix** to display the game status.

## Features

- Implemented in Verilog
- Runs on PYNQ-Z2
- Button-controlled cursor movement and position selection
- LED matrix output
- Game state and win-condition logic
- Verilog testbench for simulation

## Project Structure

```text
Ultimate-Tic-Tac-Toe/
├── code/
│   ├── Top.v
│   ├── Main.v
│   └── MB.v
├── testbench/
│   └── TTTTB.v
├── Ultimate Tic Tac Toe report.pdf
└── README.md