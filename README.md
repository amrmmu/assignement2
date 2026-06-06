# Vaccination Centre Assignment — Python Benchmarking Project

**Assignment 2 · MMU**

This project assigns 10,000 vaccinees to the nearest vaccination centre
using three different algorithms, then benchmarks execution time across
two Docker containers (simulating two computers with different hardware
resources) and across the three methods.

---

## Project Structure

```
.
├── data/
│   ├── people.csv          # 10,000 vaccinee locations (auto-generated)
│   └── centre.csv          # 100 vaccination centre locations (auto-generated)
├── src/
│   ├── utils.py            # Path helpers, directory creation
│   ├── system_info.py      # Container / hardware spec collection
│   ├── data_loader.py      # CSV loading with auto column detection
│   ├── distance.py         # Haversine distance functions (3 variants)
│   ├── assignment_methods.py  # Three assignment methods
│   ├── generate_data.py    # Synthetic dataset generator
│   ├── benchmark_q1.py     # Q1: benchmarks one method per container
│   ├── benchmark_q2.py     # Q2: benchmarks all three methods
│   ├── analyze_q1.py       # Q1: statistics, plots, tables
│   └── analyze_q2.py       # Q2: statistics, plots, tables
├── results/
│   ├── q1/                 # Q1 output files (generated at runtime)
│   └── q2/                 # Q2 output files (generated at runtime)
├── report/
│   └── experiment_report.md
├── Dockerfile
├── docker-compose.yml 
└── requirements.txt
```

---

## Dataset Files

The datasets are **auto-generated** during `docker compose build` if they
are not already present.  To regenerate them manually:

```bash
python src/generate_data.py --overwrite
```

If you already have your own `people.csv` and `centre.csv`, place them in
the `data/` folder.  The loader auto-detects common column names:

| Column | Accepted names |
|--------|----------------|
| Latitude | `lat`, `latitude`, `Lat`, `Latitude` |
| Longitude | `lon`, `lng`, `longitude`, `Lon`, `Long`, `Longitude` |
| Person ID | `person_id`, `id`, `person`, `people_id`, `vaccinee_id` |
| Centre ID | `centre_id`, `center_id`, `id`, `centre`, `center` |

If no ID column is found, IDs are generated automatically
(`Person_1 … Person_10000`, `Centre_1 … Centre_100`).

---

## Assignment Methods

| Method | Description |
|--------|-------------|
| **Method 1** | Pure Python nested loops using the `math` module — no NumPy |
| **Method 2** | Vectorised NumPy broadcasting — computes the full (10 000 × 100) distance matrix |
| **Method 3** | `scipy.spatial.cKDTree` spatial index — converts lat/lon to 3-D Cartesian, builds a KD-tree, queries in O(log n) |

---

## Docker Setup

### Requirements
- Docker Engine ≥ 20.10
- Docker Compose plugin (`docker compose`) ≥ 2.0

### Container specifications

| Service | CPU limit | RAM limit | Purpose |
|---------|-----------|-----------|---------|
| `computer1` | 4 CPUs | 4 GB | Q1 benchmark — stronger container |
| `computer2` | 1 CPU  | 512 MB | Q1 benchmark — weaker container |
| `q2_methods` | 4 CPUs | 4 GB | Q2 benchmark — all three methods |
| `analysis` | (host default) | (host default) | Post-processing only |

---

## Running the Benchmarks
 
### Step by step

```bash
# 1. Build the Docker image
docker compose build

# 2. Run Q1 benchmark on Computer 1 (stronger container)
docker compose run --rm computer1

# 3. Run Q1 benchmark on Computer 2 (weaker container)
docker compose run --rm computer2

# 4. Run Q2 benchmark (all three methods, stronger container)
docker compose run --rm q2_methods

# 5. Generate plots, tables, and statistical tests
docker compose run --rm analysis
```

> Run **computer1** and **computer2** sequentially (not in parallel) so that
> the CPU limits are fully isolated and the timing results are not
> contaminated by shared load.

---

## Running Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Generate datasets
python src/generate_data.py

# Q1 benchmark (run twice — once per "computer")
python src/benchmark_q1.py --computer computer1
python src/benchmark_q1.py --computer computer2

# Q2 benchmark
python src/benchmark_q2.py

# Analysis
python src/analyze_q1.py
python src/analyze_q2.py
```

---

## Output Files

### Question 1 (`results/q1/`)

| File | Description |
|------|-------------|
| `computer1_times.csv` | 50 execution times for Computer 1 |
| `computer2_times.csv` | 50 execution times for Computer 2 |
| `table_1_execution_time_comparison.csv` | Side-by-side table (No / C1 / C2) |
| `q1_summary_statistics.csv` | Mean, SD, median, IQR for each computer |
| `q1_statistical_tests.csv` | Shapiro-Wilk + t-test or Mann-Whitney results |
| `q1_density_plot.png` | Shaded KDE density plot |
| `q1_boxplot.png` | Boxplot with jitter |
| `q1_assignment_sample.csv` | First 100 assignment rows (last run) |
| `q1_system_specs.csv` | Hardware / container specifications |

### Question 2 (`results/q2/`)

| File | Description |
|------|-------------|
| `method1_times.csv` | 50 execution times for Method 1 |
| `method2_times.csv` | 50 execution times for Method 2 |
| `method3_times.csv` | 50 execution times for Method 3 |
| `table_2_execution_time_comparison.csv` | Side-by-side table (No / M1 / M2 / M3) |
| `q2_summary_statistics.csv` | Summary stats per method |
| `q2_statistical_tests.csv` | Shapiro-Wilk + ANOVA or Kruskal-Wallis results |
| `q2_posthoc_tests.csv` | Pairwise post-hoc comparisons (if significant) |
| `q2_density_plot.png` | Shaded KDE density plot (linear scale) |
| `q2_density_plot_log.png` | Shaded KDE density plot (log10 scale) |
| `q2_boxplot.png` | Boxplot with jitter |
| `q2_assignment_sample.csv` | First 100 assignment rows (last method, last run) |

--- 

## Dependencies

```
numpy       pandas      scipy
matplotlib  seaborn     psutil
scikit-learn
```

Python 3.11 · Docker image: `python:3.11-slim`
