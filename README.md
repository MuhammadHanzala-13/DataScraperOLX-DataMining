# Pakistan Used Cars Data Mining Pipeline 🚗

A high-performance web scraping and data engineering pipeline to collect and prepare real-world vehicle data for Artificial Neural Network (ANN) modeling. 

This project bypasses common anti-bot restrictions by extracting structured HTML directly from **PakWheels.com** (Pakistan's largest automotive portal). It processes thousands of records into a clean, feature-engineered matrix ready for machine learning.

---

## ⚡ The Project Flow

This pipeline is split into two hardcore, no-fluff steps:

### Phase 1: High-Speed Web Scraper (`pakwheels_scraper.py`)
Instead of slow browser automation, this script uses high-concurrency HTML parsing. 
- **What it does:** Crawls the search pages of PakWheels.
- **Impact:** Gathers deep technical details (Title, Price, Year, Mileage, Engine CC, Fuel Type, Transmission, Location) at a speed of ~30 vehicles per second. It ensures a massive, reliable dataset without getting IP blocked by Cloudflare.
- **Output:** Saves the raw, unprocessed text strings into `data/pakwheels_cars_raw.csv`.

### Phase 2: Data Engineering Engine (`pakwheels_data_engineering.py`)
Raw data is messy. Text like "PKR 55.5 lacs" or "10,000 km" mathematically breaks neural networks.
- **What it does:** Applies strict, programmatic cleaning. 
  - Converts human-readable currencies (Lacs/Crores) into pure integers.
  - Strips text (`km`, `cc`) from numerical columns.
  - Prunes statistical outliers (e.g., cars below 100k or unrealistic mileages).
  - **Feature Engineering:** Calculates `car_age`, a logarithmic distribution of the price (`price_log`), and the wear-and-tear metric `mileage_per_year`.
  - **Categorical Encoding:** Prepares String columns (Fuel, Transmission) into model-ready integers.
- **Impact:** Converts scraped text noise into a mathematically perfect matrix. This dramatically increases the predictive accuracy and $R^2$ score of your ANN models.
- **Output:** Generates `data/pakwheels_cars_processed.csv`.

---

## 🛠️ How to Run the Pipeline

### Prerequisites
Make sure your terminal is inside the `DataMining` folder and your virtual environment is active. Install the required libraries:
```bash
pip install -r requirements.txt
```

### Step 1: Execute the Scraper
Run the scraper to build your raw data foundation.
```bash
python pakwheels_scraper.py
```
*Note: By default, it is configured to scrape 100 pages (~3000 cars). To change this, modify `total_pages=100` at the very bottom of the script.*

### Step 2: Execute the Data Engineering Pipeline
Once the raw CSV is generated, clean it and engineer the features:
```bash
python pakwheels_data_engineering.py
```
This will instantly clean types, remove missing values, prune outliers, and output the final `data/pakwheels_cars_processed.csv`.

---

## 📊 Dataset Structure (Processed Output)

The final dataset contains the following key columns prepared for Machine Learning:

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `price` | Integer | The exact target variable (Target prediction). |
| `price_log` | Float | Natural log of price, normalizing right-skewed data. |
| `year` | Integer | Manufacturing year of the vehicle. |
| `car_age` | Integer | Calculated age from the current year (More linear than 'year'). |
| `mileage_km` | Integer | Total distance driven in kilometers. |
| `engine_cc` | Integer | Engine capacity in CC. |
| `mileage_per_year` | Float | Engineered feature indicating vehicle usage intensity. |
| `fuel_type_encoded` | Integer | Label encoded mapping for Petrol/Diesel/Hybrid. |

---
*Built for ANN Modeling & Advanced Data Science Analysis.*
