# Dashboard

Interactive Streamlit dashboard for exploring UK household expenditure patterns across 5,144 households from the ONS Living Costs and Food Survey 2013.

## Features

- 🔍 Filter by demographic groups (occupational class, tenure type, number of adults, number of children)
- 📊 Compare spending distributions across categories
- 📈 View ANOVA statistical test results and effect sizes
- 💾 Export custom analyses

## Why It's Not Deployed

The dataset (ONS Living Costs and Food Survey 2013) is not included in this repository due to licensing restrictions. A live deployment would require the dataset to be hosted publicly, which is not permitted under the UK Data Service terms of use.

## Running Locally

1. Download the teaching dataset from the UK Data Service:
   - Register at [ukdataservice.ac.uk](https://ukdataservice.ac.uk/) (free for academic users)
   - Search for "Living Costs and Food Survey 2013"
   - Direct link: [SN 7472](https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=7472)

2. Place `LCF_cleaned.csv` in the `data/` folder

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the dashboard:
   ```
   streamlit run dashboard/app.py
   ```

## Data Access

See the main [README](../README.md) and [data/README.md](../data/README.md) for full data access instructions and variable descriptions.
