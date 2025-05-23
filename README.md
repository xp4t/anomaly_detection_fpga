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
- Vitis AI 1.4+
- Vitis AI Runtime (VART)
- Compiled `.xmodel` file

## Usage

### CPU-Based Detection

Run anomaly detection on a video file using:

```bash
python anomaly_detection.py --video path/to/video.mp4 --threshold 0.5
```

Output includes prediction, confidence, and visualizations saved in a timestamped folder.

### FPGA-Based Detection

1.
Ensure that the ZCU102 SDCard has been flashed with the correct version of the image file (``xilinx-zcu102-dpu-v2022.2-v3.0.0.img.gz``) and boots correctly before proceeding:

- Vitis AI 3.5 uses [the same ZCU102 image files adopted already in Vitis AI 3.0](https://xilinx.github.io/Vitis-AI/3.0/html/docs/quickstart/mpsoc.html), build with Vitis/Petalinux 2022.2 release, in case of MPSoC (ZCU102, ZCU104, KV260) boards;

- once the board is on, follow the instructions reported in [Run The Vitis-AI (3.0) Examples](https://xilinx.github.io/Vitis-AI/3.0/html/docs/quickstart/mpsoc.html#run-the-vitis-ai-examples) to complete the Vitis-AI setup on the board.
  
3. Deploy using `code.py` (adapted to your platform's Python API for VART)

```bash
python3 code.py
```

Ensure your `.xmodel` and preprocessed video data are available on the target platform.

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
IntelliWatch: Real-Time Anomaly Detection Using FPGA-Powered CNN, 
BTech Project Report, ASIET, 2025.
```

## License

This project is developed as part of academic coursework and is open for research and educational use.
