# TrajectoryNet++: Real-Time Small Object Trajectory Tracking with Multi-scale Priors and Cross-Knowledge Attention

TrajectoryNet++ is a real-time deep learning framework for **high-speed small object trajectory tracking** in sports videos, such as tennis and badminton.  
Built upon the TrackNetV2 backbone, TrajectoryNet++ explicitly integrates **multi-scale geometric priors** and **cross-level spatial–channel attention** to achieve robust and stable tracking under **motion blur, occlusion, and complex backgrounds**.

The framework introduces two key modules:

- **MSPEM**: Multi-scale Surround Prior Extraction Module  
- **CKAM**: Cross Knowledge Attention Module  

Extensive experiments demonstrate that TrajectoryNet++ significantly outperforms existing TrackNet-based methods in terms of **Accuracy, Recall, F1-score**, while maintaining **real-time inference speed**.

---

### Key Features

- 🚀 **Real-time performance** (≈46 FPS)
- 🎯 Robust tracking of **fast-moving tiny objects**
- 🧠 Explicit **multi-directional geometric priors**
- 🔄 Cross-layer **spatial–channel attention fusion**
- 🏟️ Designed for **sports video analysis scenarios**

---

### Evaluation Metrics
- **Detection Metrics**: Accuracy, Precision, Recall, F1-score  
- **Localization Criterion**: Correct if Euclidean distance < 4 pixels  
- **Efficiency**: Frames Per Second (FPS)  
- **Robustness Analysis**: Motion blur, occlusion, fast direction changes  

### Dataset Organization
TrajectoryNet++ follows the standard TrackNet-style dataset format:
```
dataset_root/
├── train/
│ ├── frames/ # Training video frames
│ └── labels/ # Ground-truth ball coordinates
├── test/
│ ├── frames/ # Testing video frames
│ └── labels/
├── split/
│ ├── game_level.txt # Game-level split
│ └── clip_level.txt # Clip-level split
└── dataset.yaml # Dataset configuration
```

Supported datasets:
- Tennis Tracking Dataset
- Badminton Tracking Dataset

---

## Training

### Basic Training
```bash
python train.py \
  --data /path/to/dataset.yaml \
  --epochs 100 \
  --batch-size 16 \
  --lr 0.001 \
  --model trajectorynetpp \
  --output-dir ./runs/train_exp1
```

### Advanced Training Options
```bash
python train.py \
  --data /path/to/dataset.yaml \
  --epochs 150 \
  --batch-size 32 \
  --lr 0.001 \
  --optimizer adamw \
  --scheduler cosine \
  --use-amp \
  --use-ema \
  --num-workers 8 \
  --enable-mspep \
  --enable-ckam \
  --output-dir ./runs/train_exp2

```

## Evaluation

### Standard Evaluation
```bash
python test.py \
  --data /path/to/dataset.yaml \
  --weights runs/train_exp1/checkpoints/best.pt \
  --batch-size 16 \
  --distance-threshold 4 \
  --save-results \
  --save-visualizations

```
Evaluation outputs include:

·Quantitative metrics (Accuracy / Precision / Recall / F1)

·Trajectory continuity statistics

·Visualization of predicted trajectories

### Legacy Evaluation
```bash
python test.py \
  --weights runs/train/exp/weights/best.pt \
  --data data/tennis.yaml \
  --task val \
  --distance-threshold 4 \
  --device 0
```



## Citation

If you use this code in your research, please cite:

```bibtex
@article{trajectorynetpp2025,
  title={TrajectoryNet++: Real-Time Small Object Trajectory Tracking with Multi-scale Surround Priors and Cross Knowledge Attention},
  author={Du, Zirui and Tong, Wei and Zhao, Li},
  journal={},
  year={2025}
}
```

## License

This project is released under the MIT License.

