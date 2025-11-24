# Diabetic Patient Consumable Acquisition Simulation

This project simulates the process of diabetic patients acquiring their consumables from different locations in Stockholm.

## Project Structure

```
.
├── model/
│   ├── patient.py      # Patient personas and Patient class
│   └── model.py        # Main simulation model
├── result/             # Output folder for CSV results and plots
├── venv/               # Virtual environment
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Setup

1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the simulation:
```bash
python run_simulation.py
# or
python model/model.py

```

This will:
- Simulate 6 patient personas with different scenarios
- Generate a CSV file with timestamp in the `result/` folder
- Create visualization plots showing the time taken for each patient

## Patient Scenarios

The simulation includes 6 patient personas:

1. **Anna**: Wants just insulin, in stock
2. **Björn**: Wants just insulin, not in stock
3. **Cecilia**: Wants just pump, in stock
4. **David**: Wants just pump, not in stock
5. **Eva**: Wants both insulin and pump, both in stock
6. **Fredrik**: Wants both insulin and pump, not in stock

## Consumables

Patients need 5 consumables:

**From Pharmacy:**
- Insulin
- Pump

**From 1177 Platform:**
- CGM (Continuous Glucose Monitor)
- Blood stripes
- Ketone stripes

## Model Logic

### Pharmacy Process:
1. Patient travels to pharmacy (10-30 mins random)
2. Queue waiting time (~5 mins)
3. Pharmacy checks stock
4. If in stock: Handover delay (~5 mins)
5. If not in stock: Order wait (1-3 days random), then return travel and handover

### 1177 Platform Process:
1. Online ordering time (5-20 mins random)
2. Delivery wait (2-7 days random)
3. Travel to pickup point (10-30 mins random)

## Output

Results are saved as CSV files in the `result/` folder with format:
- `results_YYYYMMDD_HHMMSS.csv`

Plots are saved as PNG files:
- `simulation_plot_YYYYMMDD_HHMMSS.png`

