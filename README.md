# 🧠 AgenticSensor: Interpretable Wearable Anomaly Diagnosis via Multi-Agent Reasoning

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/Architecture-Multi--Agent-10B981?style=for-the-badge&logo=openai&logoColor=white" alt="Multi-Agent"></a>
  <a href="#"><img src="https://img.shields.io/badge/Evaluation-Synthetic%20Benchmark-3B82F6?style=for-the-badge&logo=pytest&logoColor=white" alt="Synthetic Benchmark"></a>
  <a href="https://archive.ics.uci.edu/ml/datasets/mhealth+dataset"><img src="https://img.shields.io/badge/Dataset-MHEALTH%20%7C%20PAMAP2-F97316?style=for-the-badge&logo=database&logoColor=white" alt="Dataset"></a>
</p>

---

## 📐 Framework Architecture & Perception Pipeline

This repository implements the **Perception Layer (Layer 2)** within the 4-layer System Architecture Framework:

<div align="center">
  <img src="utils/framework_architecture.png" alt="System Architecture Framework" width="95%"/>
</div>

> **System Framework**: The Perception Layer transforms raw multi-modal sensor streams (accelerometer, gyroscope, ECG) into structured diagnostic insights through feature extraction, statistical signal processing, and multi-modal visual plot encoding.

### 🔄 Multi-Agent 2-Thread Pipeline

```text
Raw Sensor Data (Parquet) & Fused Image Plots
                      │
                      ▼
         [Thread 1: Vision & Evidence Pipeline]
         - Encodes multi-sensor plots into visual representations
         - Extracts baseline evidence & visual anomaly tags
                      │
                      ▼
         [Thread 2: Specialist Reasoning Engine]
         ┌──────────────────┬──────────────────┬──────────────────┐
         │  Impact Agent    │   Health Agent   │ Sensor Fault Agt │
         │ (Falls/Impacts)  │(Heart Rate/Exert)│ (Dropout/Drift)  │
         └──────────────────┴──────────────────┴──────────────────┘
                      │
                      ▼
         [Thread 2 Final Reasoning Agent]
         - Aggregates specialist reasoning & numerical evidence
         - Generates unified root-cause diagnostic report
```

---

## 🔍 Anomaly Types & Capabilities

| Anomaly Category | Target Events & Signal Characteristics |
| :--- | :--- |
| 💥 **Physical Impact** | Sudden acceleration spikes, freefall deceleration windows, and physical collision patterns. |
| 🩺 **Health Event** | Physiological stress, abnormal heart rate fluctuations, and activity state transitions. |
| 🛠️ **Sensor Hardware Fault** | Signal dropouts (zero-variance flatlines), Gaussian noise injection, value clipping, and sensor drift. |
| 🖥️ **Interactive Inspection** | PyQt visual interface for step-by-step sensor signal plot analysis and agent debugging. |

---

## 📊 Experimental Evaluation (Synthetic Benchmark, n = 299)

The framework is evaluated on a **synthetic wearable anomaly benchmark dataset** consisting of **299 generated scenarios** constructed from real-world MHEALTH and PAMAP2 sensor recordings.

### 1. Detector-Level Performance (MHEALTH & PAMAP2)

| Model | MHEALTH F1 | MHEALTH Acc | PAMAP2 F1 | PAMAP2 Acc |
| :--- | :---: | :---: | :---: | :---: |
| MOMENT | 0.3076 | 0.4991 | 0.7320 | 0.6390 |
| GRU-AE | 0.3145 | 0.5009 | 0.5524 | 0.6606 |
| Linear-AE | 0.4174 | 0.5016 | 0.5373 | 0.6679 |
| TranAD | 0.7321 | 0.6964 | 0.4255 | 0.6169 |
| **AgenticSensor (Ours - CSCAD)** | **`0.9105`** | **`0.9103`** | **`0.9330`** | **`0.9330`** |

### 2. End-to-End Multimodal Anomaly Diagnosis ($n = 299$)

| System | State (TSA) | SemSim | Temporal (TIoU) | Event (EFA) | Event (TLC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Codex | 0.3067 | 0.0940 | 0.1210 | 0.1700 | 0.2840 |
| Copilot | 0.4370 | 0.1090 | 0.3100 | 0.1510 | 0.3670 |
| SAGE | 0.7070 | 0.5907 | 0.4730 | 0.4530 | 0.4160 |
| TSAD-style | 0.1100 | 0.5080 | 0.2790 | 0.4110 | 0.5960 |
| **AgenticSensor (Vision Only)** | 0.1130 | 0.6670 | 0.6430 | 0.6830 | 0.7350 |
| **AgenticSensor (Ours)** | **`0.7880`** *(+11.5%)* | **`0.7640`** *(+29.3%)* | **`0.8300`** *(+75.5%)* | **`0.7230`** *(+59.6%)* | **`0.7670`** *(+28.7%)* |

> 📌 **Benchmark Context**: By fusing visual plot representations with numerical verification across synthetic anomaly scenarios, **AgenticSensor** demonstrates substantial relative gains in Temporal IoU (**+75.5%**) and Event Frame Accuracy (**+59.6%**) compared to external baseline configurations.

---

## 🛠️ Quick Installation & Setup

1. **Clone repository:**
   ```bash
   git clone https://github.com/your-username/agentic-sensor-perception.git
   cd agentic-sensor-perception
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv env
   # On Windows: env\Scripts\activate
   # On Linux/macOS: source env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚡ Execution Commands

- **Batch Scenario Runner:**
  ```bash
  python perception_layer/main.py
  ```

- **Single Scenario Execution:**
  ```bash
  python perception_layer/main_2.py <scenario_id>
  ```

- **Interactive GUI Visualizer:**
  ```bash
  python perception_layer/gui.py
  ```
