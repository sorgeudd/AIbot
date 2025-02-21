# AIBot - The Gatherer AI

An enhanced version of The Gatherer with AI-powered automation and easy model training capabilities.

## Features

- Real-time resource detection using computer vision
- AI-powered automation with OpenAI integration
- Custom model training interface with GUI
- Headless mode support for automated operation
- Integrated visual feedback and monitoring
- Extensible resource detection system

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster detection)
- OpenAI API key (for AI analysis features)

## Dependencies

The project uses the following main libraries:
- PyTorch + torchvision (for computer vision)
- OpenCV (for image processing)
- CustomTkinter (for GUI)
- OpenAI (for AI analysis)
- MSS (for screen capture)
- NumPy (for numerical operations)
- Pillow (for image handling)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AIbot.git
cd AIbot
```

2. Set up your environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key

3. Download the model:
   Due to file size limitations, the trained model is not included in the repository.
   You can either:
   - Train your own model using the GUI (recommended)
   - Download our pre-trained model from [releases page](https://github.com/yourusername/AIbot/releases)
   and place it in the `models/` directory as `resource_detector.pth`

## Usage

### GUI Mode
```bash
python src/main.py
```

### Headless Mode
```bash
python src/main.py --headless
```

## Project Structure

```
├── src/
│   ├── ai_model/         # AI and ML components
│   ├── gui/             # GUI interfaces
│   └── utils/           # Utility functions
├── models/             # Trained model storage
└── data/              # Training data storage
```

## Training Custom Models

1. Launch the application in GUI mode
2. Click "Train New Model"
3. Load your training images
4. Follow the AI-guided training process
5. The trained model will be saved in the `models/` directory

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

Special thanks to the following open-source projects:
- PyTorch
- OpenCV
- CustomTkinter
- OpenAI API