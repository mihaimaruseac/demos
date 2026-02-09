# Squares and Hexagons puzzle

How well can LLMs reason about 2D grids? Suppose we place the numbers on a
spiral on a grid (square or hexagonal) and then ask the LLMs to locate 2026.
To make this easier to check, and slightly more complex, we actually ask the
LLM to identify the neighbors of 2026 and compute their sum.

This problem is my first 2026 AI benchmark, introduced in
[this blog article](https://mihai.page/ai-2026-0/), with these
[initial results](https://mihai.page/ai-2026-1/). More experiments are needed,
and I will update this with more data.

## Contents

- The main harness is in `run.py`
- The prompts sent to the LLM are in `prompts/`
- The collected results are in `results/`
- A report of all the collected runs is in `report.csv`
