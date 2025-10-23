# Data Analytics for Pandemic Management using MapReduce and Apriori Algorithm

A scalable big data analytics framework for COVID-19 pandemic management that combines MapReduce parallel processing with Apriori association rule mining to optimize medical supply demand prediction and resource allocation.

[![Published in Elsevier](https://img.shields.io/badge/Published-Elsevier%20Procedia-blue)](https://www.sciencedirect.com)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.procs.2023.12.101-orange)](https://doi.org/10.1016/j.procs.2023.12.101)
[![Conference](https://img.shields.io/badge/ICECMSN-2023-green)](https://www.sciencedirect.com)

## 📄 Publication

**Title**: Data Analytics for Pandemic Management using MapReduce and Apriori Algorithm  
**Authors**: Shashwat Kumar, Anannya Chuli, Aditi Jain, Narayanan Prasanth  
**Published in**: *Procedia Computer Science*, Volume 230, 2023, Pages 455-466  
**Conference**: 3rd International Conference on Evolutionary Computing and Mobile Sustainable Networks (ICECMSN 2023)  
**Publisher**: Elsevier B.V.  
**DOI**: [10.1016/j.procs.2023.12.101](https://doi.org/10.1016/j.procs.2023.12.101)  
**License**: CC BY-NC-ND 4.0

## 🎯 Overview

This research addresses critical challenges in pandemic management by leveraging big data analytics to predict medical supply demand and optimize resource allocation. The framework achieved a **50% average speedup** through parallel processing and uncovered actionable insights for healthcare decision-makers.

### Key Findings

- **22.29% support rate** for N95 masks, indicating high demand
- **Perfect 1.00 support rate** between N95 masks and chloroquine co-occurrence
- **0.220 confidence** in joint purchases of butyl rubber gloves and surgical masks
- **50% performance improvement** through MapReduce parallelization

## 🔬 Methodology

### Architecture

Our multi-tiered framework integrates:

1. **Data Source Tier**: Aggregates COVID-19 data from health agencies, hospitals, and research institutions
2. **Data Preprocessing Tier**: Cleans and transforms data using Hadoop/Spark
3. **MapReduce Processing Tier**: Parallel processing for pattern discovery
4. **Apriori Algorithm Tier**: Association rule mining for demand prediction
5. **Machine Learning Tier**: Model training and validation
6. **Visualization Tier**: Interactive dashboards and reports

### Algorithms

**MapReduce Implementation**
- Parallel processing of transactional data
- Efficient item occurrence counting
- Cluster-based disease tracking and risk factor identification

**Apriori Algorithm Implementation**
- Frequent itemset mining (min support: 5%)
- Association rule generation using Support, Confidence, and Lift metrics
- Top 50 item analysis from COVID Optimization Dataset

## 📊 Dataset

**Source**: COVID Optimization Dataset (Kaggle)  
**Type**: Transactional data from pharmacy store sales  
**Content**: Pharmaceutical products and COVID-19 safety equipment purchases  
**Format**: CSV with transaction-based records

## 🚀 Installation & Usage

### Prerequisites

```bash
Python 3.7+
Apache Hadoop
Apache Spark (optional)
```

### Dependencies

```bash
pip install pandas numpy matplotlib seaborn
pip install mlxtend  # For Apriori algorithm
pip install multiprocessing  # Built-in Python module
```

### Running the Analysis

```bash
# Data visualization and preprocessing
python DV.py

# Parallel data computation
python pdc.py

# View dashboard
# Open Dashboard.png
```

## 📈 Performance Results

### Parallel Processing Speedup

| Tier | Serial Time | Parallel Time | Speedup |
|------|-------------|---------------|---------|
| Estimator | 221 ms | 144 ms | **25.4%** |
| Visualization | 250 ms | 151 ms | **34.5%** |
| Data Encoding | 990 ms | 225 ms | **75.2%** |
| Apriori | 90.1 ms | 57.7 ms | **26.7%** |

**Average Speedup: 50%**

### Association Rules Discovered

Top associations include:
- N95 masks ↔ Chloroquine (1.00 support)
- Butyl rubber gloves → Surgical masks (0.220 confidence)
- Hand sanitizer ↔ Isolation gowns
- Booster vaccines ↔ Disinfectant

## 📁 Repository Structure

```
.
├── COVID_Optimisation.csv       # Dataset
├── DV.py                         # Data visualization script
├── Dashboard.png                 # Analysis dashboard
├── pdc.py                        # Parallel data computation
├── worldometer_data.csv          # Additional COVID-19 data
└── README.md
```

## 💡 Applications

### Healthcare Management
- **Demand Prediction**: Forecast medical supply requirements
- **Inventory Optimization**: Efficient resource allocation
- **Supply Chain Management**: Prevent shortages during crises

### Public Health Policy
- **Risk Factor Identification**: Cluster analysis of vulnerable populations
- **Disease Tracking**: Monitor spread patterns over time
- **Intervention Evaluation**: Assess the effectiveness of public health measures

### Business Intelligence
- **Personalized Marketing**: Targeted product recommendations
- **Customer Behavior Analysis**: Purchase pattern insights
- **Stock Management**: Optimize inventory based on co-purchase patterns

## 🔮 Future Scope

- Integration with real-time data streams for live monitoring
- Implementation of additional ML algorithms (Decision Trees, SVM)
- Expansion to other pandemic-related datasets
- Development of predictive models for future outbreak scenarios
- Mobile application for healthcare workers
- Integration with IoT sensors for automated data collection

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{kumar2023data,
  title={Data Analytics for Pandemic Management using MapReduce and Apriori Algorithm},
  author={Kumar, Shashwat and Chuli, Anannya and Jain, Aditi and Prasanth, Narayanan},
  journal={Procedia Computer Science},
  volume={230},
  pages={455--466},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.procs.2023.12.101}
}
```



---

**Published in Procedia Computer Science | Elsevier B.V. | 2023**
