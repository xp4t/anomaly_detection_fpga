# IntelliWatch: Real-Time Anomaly Detection Using FPGA-Powered CNN

## Overview

**IntelliWatch** is a real-time surveillance system designed to detect anomalies in video footage such as thefts, fights, and medical emergencies. The system uses a Convolutional Neural Network (CNN) deployed on an FPGA (Xilinx ZCU102) to deliver high-speed, low-latency detection, ensuring prompt alert generation in critical scenarios. The project combines the efficiency of FPGA hardware with the learning capabilities of deep neural networks to enhance public safety.

## Features

- **FPGA-based acceleration** for real-time performance
- Trained on the **UCF-Crime dataset**
- CPU-based reference model for comparison and testing
- Uses **Inception V3** CNN architecture
- Optimized using Vitis AI tools (quantization, pruning)
- Deployment-ready `.xmodel` for FPGA and `.h5` for CPU

## File Structure

```
IntelliWatch/
├── code.py                        # Code for running the model on FPGA (Vitis AI)
├── anomaly_detection.py           # CPU-based anomaly detection pipeline (TensorFlow)
├── anomaly_detection_model.h5     # Trained Keras model for CPU execution
├── anomaly_detection_model.xmodel # Compiled model for FPGA inference
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Requirements

### CPU Execution

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

Install dependencies using:
```bash
pip install -r requirements.txt
```

### FPGA Execution

- Xilinx ZCU102 FPGA board
- Vitis AI 3.5
- Vitis AI Boot Image
- Compiled `.xmodel` file

## Usage

Will Update in Detail as Time Permits

## Performance

| Platform | Execution Time | Power Consumption | Notes |
|----------|----------------|-------------------|-------|
| CPU      | <20 ms         | ~95W             | Development/debug |
| GPU      | ~15.9 ms       | ~150W            | High throughput |
| FPGA     | ~9.03 ms       | ~10W             | Real-time, power-efficient |

## Authors

- **Rithwik Vallabhan TV** (ASI21EC072)
- **Noel Martin** (ASI21EC066)
- **Gautham VG** (ASI21EC049)
- **Gouthamkrishna R** (ASI21EC052)

**Project Guide:** Mrs. Remya Ramesh  
**Affiliation:** Department of ECE, Adi Shankara Institute of Engineering and Technology

## Citation

If you use this work or base any project on it, please cite:

```
IntelliWatch: Real-Time Anomaly Detection Using FPGA-Powered CNN, ASIET, 2025.
```

## License

This project is developed as part of academic coursework and is open for research and educational use.
