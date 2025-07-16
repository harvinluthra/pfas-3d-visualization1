# PFAS 3D Visualization Tool

An interactive 3D visualization tool for analyzing PFAS (Per- and polyfluoroalkyl substances) data. This Streamlit application creates dynamic 3D bar charts to visualize PFAS class distributions, carbon numbers, and concentration levels.

## Features

- 📊 **Interactive 3D Visualization**: Rotate, zoom, and explore your PFAS data in 3D space
- 📁 **CSV Data Upload**: Easy drag-and-drop file upload for your PFAS datasets
- 🎨 **Multi-Color Gradient**: Visual concentration mapping from white (zero) to red (highest)
- 📥 **Export Options**: Download interactive HTML charts and processed CSV data
- 🔬 **Scientific Focus**: Designed specifically for PFAS research and analysis

## Data Format

Your CSV file should contain these columns:
- `PFAS_Class`: PFAS classification (FASEs, FASAAs, etc.)
- `PFAS_Compd`: Compound name
- `Carbon_Number`: Number of carbon atoms (numeric)
- `Concentration`: Concentration values (numeric, can include zeros)

## Live Demo

🚀 **[Try the app here](https://pfas-3d-visualization.streamlit.app)** *(link will be available after deployment)*

## How to Use

1. Upload your CSV file containing PFAS data
2. View the interactive 3D bar chart
3. Rotate and zoom to explore your data
4. Download the chart as an interactive HTML file
5. Export processed data as CSV

## Technical Details

- Built with Streamlit and Plotly
- Supports raw concentration data (no log transformation)
- Handles zero values appropriately
- Ordered PFAS classes for consistent visualization
- Professional styling with enhanced readability

## Local Development

```bash
pip install streamlit pandas numpy plotly
streamlit run app.py
```

## About

This tool was developed for PFAS research to provide an intuitive way to visualize complex chemical data in three dimensions, helping researchers identify patterns and relationships in their datasets.