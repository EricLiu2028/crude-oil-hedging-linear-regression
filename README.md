# Commodity Hedging Strategy

A dashboard-based crude oil futures hedging project using linear regression to estimate hedge ratios, compare hedge candidates, and rank strategies by variance reduction per unit trading cost.

## Project Overview
This project develops an interactive dashboard for crude oil futures hedging. It combines pairwise linear regression, portfolio variance analysis, and transaction-cost-aware ranking to generate hedge suggestions.

## Main Features
- estimate hedge ratios using pairwise linear regression
- calculate portfolio variance before and after hedging
- rank hedge candidates by variance reduction per unit trading cost
- interactive dashboard with inventory table, hedge suggestion table, heatmap, and regression plot

## Methodology
- linear regression for hedge ratio estimation
- portfolio variance comparison before and after hedging
- cost-aware hedge ranking based on score

## Dashboard Components
- trade entry
- inventory table
- hedge suggestion table
- hedging heatmap
- regression plot

## Project Report
[View the project report](docs/crude_oil_hedging_presentation.pdf)

## How to Run

1. Install dependencies:
```bash
poetry install
```

2. Run the app:
```bash
poetry run python app.py
```