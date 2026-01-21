# 🎭 Facial Expression Analyzer

**Real-Time Facial Emotion Detection using Computer Vision & Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-red?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)

---

## ✨ Project Overview

The **Facial Expression Analyzer** is an advanced AI-powered system that detects human emotions in **real-time** using a **Convolutional Neural Network (CNN)** and **OpenCV**.  
It intelligently processes live webcam feed, images, or videos to accurately classify emotions and display them visually with bounding boxes.

🎯 **Use Cases:**  
- 🤖 AI chatbots & virtual assistants  
- 📚 Classroom engagement & student attention analysis  
- 📊 Marketing & UX research  
- 🏥 Mental health monitoring & therapy sessions  
- 🎤 Interview & presentation feedback systems  
- 🎮 Gaming & interactive entertainment  
- 🛡️ Security & surveillance systems  

---

## 🚀 Technologies & Requirements

**Programming Language:**  
- Python 3.x

**Core Libraries:**  
- `OpenCV` - For webcam access, image processing & face detection  
- `TensorFlow/Keras` - Deep learning framework for CNN model  
- `NumPy` - Numerical operations & array processing  

**Installation:**  
```bash
pip install opencv-python tensorflow numpy
```

---


## 🧩 Model Details

- **Model Type:** Convolutional Neural Network (CNN)  
- **Input Image Size:** 64 × 64 pixels (grayscale)  
- **Output:** Probability of each emotion  
  - 😄 Happy  
  - 😢 Sad  
  - 😠 Angry  
  - 😲 Surprised  
  - 😐 Neutral  
  - 😮 Fear  
  - 😏 Disgust  

---

## 🗂 Folder Structure

Facial_Expression_Analyzer-main/  
│  
├── model/                               # Trained CNN model files  
├── haarcascade_frontalface_default.xml  # Pre-trained Haar Cascade for face detection  
├── main.py                              # Main script for detection  
├── test_webcam.py                       # Test webcam feed  
├── Facial_Expression_Analyzer_Project.txt  # Project documentation/notes  
├── README.md                             # This file  
└── .gitattributes                        # Git attributes file  

---


## 🎬 Usage

Run on Webcam:  
python main.py  

Test Webcam:  
python test_webcam.py  

Run on Image or Video:  
python main.py --image path_to_image.jpg  
python main.py --video path_to_video.mp4  

---

## 🧠 How It Works

**Step-by-Step Process:**

1. **Capture Frame:** OpenCV captures real-time frames from webcam or video file.  
2. **Convert to Grayscale:** Frame is converted to grayscale for efficient processing.  
3. **Face Detection:** Haar Cascade Classifier detects human faces in the frame.  
4. **Preprocessing:** 
   - Extract detected face region
   - Resize face image to 64×64 pixels
   - Normalize pixel values (0-1 range)
   - Reshape for CNN input format
5. **Emotion Prediction:** Pre-trained CNN model predicts emotion with probability scores.  
6. **Display Results:** 
   - Draw bounding box around detected face
   - Show predicted emotion label with confidence
   - Update display in real-time

**Technical Workflow:**
```
Webcam → Frame Capture → Grayscale Conversion → Face Detection → 
Preprocessing → CNN Model → Emotion Prediction → Visual Output
```

---

## 🎨 Features

- ✅ **Real-time Processing:** Instant emotion detection from live webcam feed  
- ✅ **Multi-format Support:** Works with images, videos, and live streams  
- ✅ **High Accuracy:** Pre-trained CNN model with optimized architecture  
- ✅ **Lightweight & Fast:** Efficient processing with minimal latency  
- ✅ **Visual Feedback:** Clear bounding boxes & emotion labels  
- ✅ **7 Emotion Classes:** Detects Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral  
- ✅ **Easy Integration:** Simple API for integration into other projects  
- ✅ **Cross-platform:** Works on Windows, Linux, and macOS  
- ✅ **Open-source:** Fully customizable and extensible  
- ✅ **Professional Documentation:** Comprehensive guides and examples

---

## 📊 Model Architecture

**CNN Configuration:**
- Input Layer: 64×64 grayscale images
- Convolutional Layers: Multiple conv2D layers with ReLU activation
- Pooling Layers: MaxPooling for feature extraction
- Dropout Layers: Prevent overfitting
- Dense Layers: Fully connected layers
- Output Layer: 7 neurons with softmax activation

**Performance:**
- Training Dataset: FER-2013 or similar emotion datasets
- Validation Accuracy: ~65-70%
- Real-time FPS: 20-30 FPS on standard hardware

---

## 🤝 Contributing

1. Fork the repository  
2. Create a branch: `git checkout -b feature-name`  
3. Commit your changes: `git commit -m 'Add feature'`  
4. Push to branch: `git push origin feature-name`  
5. Open a Pull Request  

We welcome contributions! Feel free to submit issues, feature requests, or pull requests.

---

## 📝 License

This project is open-source and available for educational and research purposes.

---

## 👨‍💻 Author

**Noman Shakir (CodeWithZavi)**

- 🌐 GitHub: [@CodeWithZavi](https://github.com/CodeWithZavi)
- 💼 LinkedIn: [codewithzavii](https://www.linkedin.com/in/codewithzavii)
- 📂 Project Repository: [Facial_Expression_Analyzer](https://github.com/CodeWithZavi/Facial_Expression_Analyzer-main)

---

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- TensorFlow/Keras for deep learning framework
- FER-2013 dataset contributors
- Open-source AI/ML community

---

## 📞 Support

If you find this project helpful, please give it a ⭐ on GitHub!  
For questions or issues, feel free to open an issue or reach out via LinkedIn.

---

**Made with ❤️ by CodeWithZavi using OpenCV & TensorFlow**
