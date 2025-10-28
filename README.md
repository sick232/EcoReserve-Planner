# EcoReserve Planner

EcoReserve Planner is a web-based application for advanced vegetation analysis and ecological reserve planning. It helps environmental scientists, conservation planners, and researchers analyze vegetation patterns and plan protected areas using satellite data and advanced algorithms.

## 🌟 Features

- **CSV Data Analysis**
  - Upload and process grid-based vegetation data
  - Configure threshold values and square sizes
  - Analyze multiple datasets simultaneously

- **Live NDVI Data**
  - Fetch real-time Normalized Difference Vegetation Index (NDVI) data
  - Pre-configured locations including:
    - Bandipur, India
    - Sundarbans, India
    - Amazon, Brazil
    - Yosemite, USA
  - Customizable grid size and analysis parameters

- **Advanced Features**
  - **Matrix Chain Analysis**: Optimize sensor data fusion sequences
  - **Secure Zones**: Calculate optimal surveillance coverage
  - Interactive visualization of results
  - Advanced spatial algorithms for conservation planning

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ecoreserve.git
cd ecoreserve
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# For Windows:
.venv\Scripts\activate
# For Unix/MacOS:
source .venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the Flask server:
```bash
python eco_reserve_web.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## 📊 Usage

### CSV Data Analysis

1. Prepare your CSV file with vegetation data
2. Upload the file through the web interface
3. Set parameters:
   - Number of zones
   - Zone names
   - Threshold values
4. Process the data to generate analysis results

### NDVI Analysis

1. Select a predefined location or enter custom coordinates
2. Configure analysis parameters:
   - Grid size
   - Threshold values
   - Maximum square size
3. Generate NDVI visualization and analysis

### Advanced Features

#### Matrix Chain Analysis
- Upload sensor data
- Set grid size parameters
- Optimize data fusion sequence

#### Secure Zones
- Define map bounds
- Calculate optimal coverage
- Generate protection zones

## 🌐 Project Structure

```
ecoreserve/
├── eco_reserve_web.py      # Main Flask application
├── requirements.txt        # Python dependencies
├── templates/             # HTML templates
│   ├── index.html
│   ├── results.html
│   ├── matrix_chain.html
│   └── secure_zones.html
└── uploads/              # Uploaded file storage
    └── sample_data/     # Sample datasets
```

## 📋 File Formats

### Input CSV Format
```csv
latitude,longitude,vegetation_value
11.7,76.6,0.85
11.71,76.61,0.76
...
```

### Location Parameters Format
```
base_lat base_lon lat_step lon_step
```
Example: `11.7 76.6 0.008 0.01`

## 🤝 Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Environmental data providers
- Open-source mapping libraries
- Conservation research community

## 📞 Support

For support, please open an issue in the GitHub repository or contact the development team.