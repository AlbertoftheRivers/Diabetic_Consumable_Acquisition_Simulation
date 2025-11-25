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
- Generate a 100-patient population following the requested demand mix
- Generate a CSV file with timestamp in the `result/` folder
- Create visualization plots showing the time taken for each patient
  plus the aggregated KPIs requested for decision making.

## Patient population & arrivals

- 100 patients are created per run.
- 50% request only insulin, 30% request only pumps, and the remaining 20% request both.
- **Arrival Schedule:** The simulation now runs a cyclical weekly schedule until all 100 patients are served.
  - Monday: 7 patients
  - Tuesday: 6 patients
  - Wednesday: 5 patients
  - Thursday: 4 patients
  - Friday: 3 patients
  - Saturday: 2 patients
  - Sunday: 1 patient
  - (Repeats the next Monday if needed)
- Working hours are constrained to 07:00–22:00. Patients are distributed evenly within these hours.
- Each patient orders the 1177 consumables before leaving for the pharmacy, meaning the online and physical processes can overlap.
- The per-patient result now includes a `return_pharmacy` boolean flag indicating whether a second visit was required due to stock-outs.

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

### Pharmacy Process

- The pharmacy starts each run with 100 units of insulin and 30 pumps.
- Patients requesting insulin consume 10 units per visit. Pump patients take 5 units.
- If an item's inventory drops to 20% of its maximum (20 insulin units or 6 pumps) a replenishment order is created automatically.
- Restocking takes 12–24 hours. Patients arriving before the truck comes may have to leave empty handed and are tagged with `return_pharmacy=True`. They must travel, queue and be served again after replenishment lands.
- All pharmacy timings (travel, queue, service, return trip) are sampled independently for every patient, and the total pharmacy time already counts eventual waiting for stock.

### 1177 Platform Process

- Patients submit the 1177 order before going to the pharmacy.
- Ordering takes 5–20 minutes, delivery waits 2–7 days, and pickup travel is 10–30 minutes.
- Because the platform order is launched first, the overall time to obtain all five consumables is the **max** of the 1177 timeline and the pharmacy timeline (they proceed in parallel).

## KPIs

Every run calculates the KPIs you requested:

1. `total_time_minutes` / `total_time_days` — Total time to get all five consumables.
2. `pharmacy_total_time_minutes` / `pharmacy_total_time_days` — Pharmacy-only timeline, including restock waits and return visits.
3. `1177_total_time_minutes` / `1177_total_time_days` — Time to get the online consumables.
4. `return_pharmacy` — Boolean per patient; aggregate return rate is shown in the console, CSV, and plots.

## Output

Results are saved as CSV files in the `result/` folder with format:
- `results_YYYYMMDD_HHMMSS.csv`

Plots are saved as PNG files:
- `simulation_plot_YYYYMMDD_HHMMSS.png`

The plot dashboard now includes:
1. **Total Pharmacy Time per Patient:** A bar chart where each patient is represented by a bar colored by their need (Insulin=Skyblue, Pump=Orange, Both=Green).
2. **Total 1177 Time per Patient:** A bar chart showing the online platform time for each patient (all Blue).
3. **Total Acquisition Time per Patient:** A bar chart of the overall max time, colored by need.
4. **Pharmacy Stock Status Overview:** A scatter plot showing the total pharmacy time for patients over the chronological sequence. Patients who had to return due to stock-outs are marked with a red 'X', while those with stock are grey dots. Vertical lines indicate weekly boundaries.

The console summary (and `run_simulation.py`) now also prints the average total time, pharmacy time, 1177 time, and the percentage of patients who needed to return to the pharmacy.

