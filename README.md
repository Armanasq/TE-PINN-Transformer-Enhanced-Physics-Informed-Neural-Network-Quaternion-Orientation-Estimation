# TE-PINN: Quaternion-Based Orientation Estimation 🚀

[![arXiv](https://img.shields.io/badge/arXiv-2409.16214-b31b1b.svg)](https://arxiv.org/abs/2409.16214)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

<div align="center">
  <img src="/images/tepinn_architecture.svg" alt="TE-PINN Architecture">
</div>

## 📊 Performance Highlights

| Metric | TE-PINN | Deep Learning Model | Improvement |
|--------|---------|-------------------|-------------|
| Mean Euler Error | 0.0195 | 0.0216 | -36.84% |
| Dynamic Error | 0.1677 | 0.1242 | -35.04% |
| Uncertainty Correlation | 0.0147 | 0.0553 | +73.44% |

## 🔍 Overview

TE-PINN is a Transformer-Enhanced Physics-Informed Neural Network for quaternion-based orientation estimation in high-dynamic environments. Our approach combines:

- 🧬 Transformer architecture for temporal dependencies
- 📐 Physics-informed constraints
- 🎯 Quaternion kinematics
- 📈 Uncertainty quantification

## 🏗️ Architecture

### Quaternion-Based Transformer Encoder

```python
x(tᵢ) = [ω(tᵢ)ᵀ, a(tᵢ)ᵀ]ᵀ
```

Positional Encoding:
```
P(i,2k) = sin(tᵢ/10000^(2k/dmodel))
P(i,2k+1) = cos(tᵢ/10000^(2k/dmodel))
```

### Physics-Informed Components

Rigid Body Dynamics:
```
I dω/dt + ω × (Iω) = τ
```

Quaternion Integration (RK4):
```
qw = cos(ϕ/2)cos(θ/2)
qx = sin(ϕ/2)cos(θ/2)
qy = cos(ϕ/2)sin(θ/2)
qz = sin(ϕ/2)sin(θ/2)
```

## 📈 Results

<div align="center">
  <img src="/images/performance_comparison.svg" alt="Performance Comparison">
</div>

### Error Metrics

```python
L_total = L_data + L_phys

where:
L_phys = λacc*L_acc + λgyro*L_gyro + λdynamics*L_dynamics
```

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/tepinn
cd tepinn
pip install -r requirements.txt
```

## 📊 Usage

```python
import tepinn

# Initialize model
model = tepinn.TEPINN(
    transformer_layers=6,
    heads=8,
    d_model=256
)

# Train model
model.train(imu_data, quaternion_gt)

# Inference
q_pred = model.predict(imu_sequence)
```

## 🔬 Experimental Results

### Ablation Study

| Component | Mean Error | Dynamic Error |
|-----------|------------|---------------|
| Base Model | 0.0216 | 0.1242 |
| + Transformer | 0.0205 | 0.1198 |
| + Physics | 0.0195 | 0.1677 |

### Parameter Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| Layers | 6 | Transformer encoder layers |
| Heads | 8 | Number of attention heads |
| d_model | 256 | Model dimension |
| λacc | 1.0 | Accelerometer loss weight |
| λgyro | 0.5 | Gyroscope loss weight |
| λdynamics | 0.1 | Dynamics loss weight |

## 🎯 Key Features

1. **Multi-Head Attention**
   - Sequential IMU processing
   - Temporal dependency capture
   - Adaptive weighting

2. **Physics-Informed Learning**
   - Quaternion kinematics integration
   - Rigid body dynamics
   - RK4 numerical integration

3. **Uncertainty Quantification**
   - Evidential deep learning
   - Confidence estimation
   - Error calibration

## 🔗 Citation

```bibtex
@article{asgharpoor2024tepinn,
  title={TE-PINN: Quaternion-Based Orientation Estimation using Transformer-Enhanced Physics-Informed Neural Networks},
  author={Asgharpoor Golroudbari, Arman},
  journal={arXiv preprint arXiv:2409.16214},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
