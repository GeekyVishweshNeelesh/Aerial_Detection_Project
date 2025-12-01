"""
Aerial Object Classification & Detection - Streamlit App
3 Models: Custom CNN, ResNet50, YOLOv8
"""

import streamlit as st
import pickle
import numpy as np
from PIL import Image
import cv2
import time
import plotly.graph_objects as go
import os
import io
import base64

# Try importing deep learning libraries
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.warning("⚠️ YOLOv8 not available. Install ultralytics to use detection features.")

# Page configuration
st.set_page_config(
    page_title="Aerial Detection System",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stTitle {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
    }
    .model-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    .result-bird {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2ecc71;
        margin: 1rem 0;
    }
    .result-drone {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    .info-box {
        background: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
MODELS_DIR = "models"
MODEL_FILES = {
    'Custom CNN': 'custom_cnn_model.pkl',
    'ResNet50 Transfer Learning': 'resnet50_transfer_learning_model.pkl',
    'YOLOv8 Detection': 'best_model_yolov8.pt',
}

CLASS_NAMES = ['Bird', 'Drone']

# Model information
MODEL_INFO = {
    'Custom CNN': {
        'type': 'Deep Learning',
        'accuracy': '100%',
        'speed': 'Medium (45 FPS)',
        'description': 'Custom CNN architecture with Conv2D, pooling, and dense layers',
        'best_for': 'Balanced speed and accuracy',
        'detection': False
    },
    'ResNet50 Transfer Learning': {
        'type': 'Transfer Learning',
        'accuracy': '99.55%',
        'speed': 'Slow (30 FPS)',
        'description': 'Pre-trained ResNet50 fine-tuned on aerial dataset',
        'best_for': 'Highest classification accuracy',
        'detection': False
    },
    'YOLOv8 Detection': {
        'type': 'Object Detection',
        'accuracy': '71.3% mAP@50',
        'speed': 'Fast (78 FPS)',
        'description': 'Real-time object detection with bounding boxes',
        'best_for': 'Object localization and detection',
        'detection': True
    }
}

@st.cache_resource(show_spinner=False)
def load_models():
    """Load all available models"""
    models = {}
    model_status = {}

    for model_name, model_file in MODEL_FILES.items():
        model_path = os.path.join(MODELS_DIR, model_file)

        try:
            if os.path.exists(model_path):
                if model_file.endswith('.pt'):
                    # YOLOv8 model
                    if YOLO_AVAILABLE:
                        models[model_name] = YOLO(model_path)
                        model_status[model_name] = "✅ Loaded"
                    else:
                        model_status[model_name] = "❌ ultralytics not installed"
                else:
                    # PKL model
                    with open(model_path, 'rb') as f:
                        models[model_name] = pickle.load(f)
                    model_status[model_name] = "✅ Loaded"
            else:
                model_status[model_name] = "❌ File not found"
        except Exception as e:
            model_status[model_name] = f"❌ Error: {str(e)[:30]}"

    return models, model_status

def preprocess_for_classification(image, target_size=(224, 224)):
    """Preprocess image for classification models"""
    # Resize image
    img_resized = image.resize(target_size)

    # Convert to numpy array and normalize
    img_array = np.array(img_resized) / 255.0

    # Add batch dimension
    img_batch = np.expand_dims(img_array, axis=0)

    return img_batch

def classify_image(model, image, model_name):
    """Classify image using CNN or ResNet50"""
    start_time = time.time()

    # Preprocess
    img_preprocessed = preprocess_for_classification(image)

    try:
        # Make prediction
        prediction = model.predict(img_preprocessed, verbose=0)

        # Handle different output formats
        if len(prediction.shape) > 1 and prediction.shape[1] > 1:
            # Multi-class output [batch, classes]
            class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][class_idx])
        else:
            # Binary output [batch, 1]
            prob = float(prediction[0][0])
            class_idx = 1 if prob > 0.5 else 0
            confidence = prob if class_idx == 1 else 1 - prob

        class_name = CLASS_NAMES[class_idx]
        inference_time = time.time() - start_time

        return model_name, class_name, confidence, inference_time

    except Exception as e:
        st.error(f"Error in classification: {str(e)}")
        return model_name, "Error", 0.0, 0.0

def detect_objects(model, image, conf_threshold=0.25, iou_threshold=0.45):
    """Detect objects using YOLOv8"""
    start_time = time.time()

    try:
        # Convert PIL to numpy
        img_array = np.array(image)

        # Run detection
        results = model.predict(
            img_array,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        inference_time = time.time() - start_time

        # Process results
        detections = []
        annotated_image = img_array.copy()

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                # Get box coordinates
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)

                # Get class and confidence
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"

                # Draw bounding box
                color = (0, 255, 0) if class_name == "Bird" else (255, 0, 0)  # Green for bird, red for drone
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"{class_name}: {confidence:.2%}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Store detection
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2)
                })

        # Convert back to PIL
        annotated_pil = Image.fromarray(annotated_image)

        return annotated_pil, detections, inference_time

    except Exception as e:
        st.error(f"Error in detection: {str(e)}")
        return image, [], 0.0

def create_confidence_gauge(confidence, class_name):
    """Create a gauge chart for confidence"""
    color = "#2ecc71" if class_name == "Bird" else "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence", 'font': {'size': 20}},
        delta={'reference': 50, 'increasing': {'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 75], 'color': '#fff9c4'},
                {'range': [75, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def get_download_link(img, filename="detection_result.jpg"):
    """Generate download link for image"""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}" style="text-decoration:none;"><button style="background:#667eea;color:white;padding:10px 20px;border:none;border-radius:5px;cursor:pointer;font-size:16px;">📥 Download Result</button></a>'
    return href

# Load models
with st.spinner("🔄 Loading models..."):
    models, model_status = load_models()

# Main title
st.markdown('<h1 class="stTitle">🚁 Aerial Object Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;font-size:1.2rem;color:#666;">Deep Learning for Bird vs Drone Classification & Detection</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Model selection
    st.markdown("### Select Model")

    # Get available models
    available_models = [name for name in MODEL_FILES.keys() if model_status.get(name, "").startswith("✅")]

    if available_models:
        selected_model = st.selectbox(
            "Choose a model:",
            available_models,
            help="Select which model to use for prediction"
        )
    else:
        st.error("❌ No models available. Please check model files.")
        selected_model = None

    # Show model info
    if selected_model:
        st.markdown("### 📊 Model Information")
        info = MODEL_INFO[selected_model]
        st.markdown(f"""
        **Type:** {info['type']}
        **Accuracy:** {info['accuracy']}
        **Speed:** {info['speed']}
        **Best for:** {info['best_for']}
        """)

    # Detection parameters (only for YOLOv8)
    if selected_model == 'YOLOv8 Detection':
        st.markdown("### 🎯 Detection Parameters")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence for detection"
        )
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Intersection over Union threshold for NMS"
        )

    st.markdown("---")

    # Model status
    st.markdown("### 📦 Loaded Models")
    for model_name, status in model_status.items():
        st.markdown(f"**{model_name}**  \n{status}")

    st.markdown("---")

    # Dataset info
    st.markdown("### 📊 Dataset Information")
    st.markdown("""
    **Total Images:** 3,319
    **Classes:** Bird, Drone
    **Training:** 2,662 images
    **Validation:** 442 images
    **Testing:** 215 images
    """)

    st.markdown("---")

    # GitHub link
    st.markdown("### 🔗 Links")
    st.markdown("[📁 GitHub Repository](https://github.com/GeekyVishweshNeelesh/Aerial_Detection_Project)")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🖼️ Predict", "📊 Model Comparison", "📖 Documentation"])

# Tab 1: Home
with tab1:
    st.markdown("## 🎯 Project Overview")

    st.markdown("""
    <div class="info-box">
    This deep learning system classifies and detects aerial objects to distinguish between <b>Birds</b> and <b>Drones</b>.
    The project implements three different approaches to provide comprehensive solutions for various use cases.
    </div>
    """, unsafe_allow_html=True)

    # Model showcase
    st.markdown("## 🤖 Available Models")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="model-card">
            <h3>🧠 Custom CNN</h3>
            <p><b>Type:</b> Deep Learning</p>
            <p><b>Accuracy:</b> 100%</p>
            <p><b>Speed:</b> Medium</p>
            <p>Custom architecture optimized for bird vs drone classification.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="model-card">
            <h3>🎯 ResNet50</h3>
            <p><b>Type:</b> Transfer Learning</p>
            <p><b>Accuracy:</b> 99.55%</p>
            <p><b>Speed:</b> Slow</p>
            <p>Pre-trained ResNet50 fine-tuned for aerial classification.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="model-card">
            <h3>🎪 YOLOv8</h3>
            <p><b>Type:</b> Object Detection</p>
            <p><b>mAP@50:</b> 71.3%</p>
            <p><b>Speed:</b> Fast</p>
            <p>Real-time detection with bounding boxes and object localization.</p>
        </div>
        """, unsafe_allow_html=True)

    # Use cases
    st.markdown("## 🌍 Real-World Applications")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="model-card">
            <h4>🌿 Wildlife Protection</h4>
            <p>Monitor bird populations near wind farms to prevent collisions and protect endangered species.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="model-card">
            <h4>✈️ Airport Safety</h4>
            <p>Detect birds near runways to prevent bird strikes and ensure aircraft safety during takeoff and landing.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="model-card">
            <h4>🛡️ Security & Defense</h4>
            <p>Identify unauthorized drones in restricted airspace for security surveillance and threat detection.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="model-card">
            <h4>🔬 Environmental Research</h4>
            <p>Track and analyze bird migration patterns without misclassification from drone activity.</p>
        </div>
        """, unsafe_allow_html=True)

# Tab 2: Predict
with tab2:
    st.markdown("## 🔮 Make Predictions")

    if not selected_model:
        st.error("❌ Please ensure models are loaded correctly.")
    else:
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload an aerial image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image of a bird or drone"
        )

        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file).convert('RGB')

            # Display image and results
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### 📸 Original Image")
                st.image(image, use_container_width=True)

            with col2:
                st.markdown("### 🎯 Prediction Results")

                # Get model
                if selected_model in models:
                    model = models[selected_model]

                    # Run prediction
                    with st.spinner('🔄 Analyzing image...'):
                        if MODEL_INFO[selected_model]['detection']:
                            # Object Detection (YOLOv8)
                            annotated_img, detections, inf_time = detect_objects(
                                model, image, conf_threshold, iou_threshold
                            )

                            # Show metrics
                            met_col1, met_col2, met_col3, met_col4 = st.columns(4)

                            with met_col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{len(detections)}</h3>
                                    <p>Total Detected</p>
                                </div>
                                """, unsafe_allow_html=True)

                            birds = sum(1 for d in detections if d['class'] == 'Bird')
                            drones = sum(1 for d in detections if d['class'] == 'Drone')

                            with met_col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{birds}</h3>
                                    <p>Birds</p>
                                </div>
                                """, unsafe_allow_html=True)

                            with met_col3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{drones}</h3>
                                    <p>Drones</p>
                                </div>
                                """, unsafe_allow_html=True)

                            with met_col4:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{inf_time:.2f}s</h3>
                                    <p>Inference Time</p>
                                </div>
                                """, unsafe_allow_html=True)

                            # Show annotated image
                            st.markdown("### 🎨 Detection Results")
                            st.image(annotated_img, use_container_width=True)

                            # Download button
                            st.markdown(get_download_link(annotated_img), unsafe_allow_html=True)

                            # Detection details
                            if detections:
                                st.markdown("### 📋 Detection Details")
                                for i, det in enumerate(detections, 1):
                                    st.markdown(f"""
                                    **Detection {i}:**
                                    Class: {det['class']}
                                    Confidence: {det['confidence']:.2%}
                                    BBox: {det['bbox']}
                                    """)
                            else:
                                st.info("ℹ️ No objects detected. Try adjusting the confidence threshold.")

                        else:
                            # Classification (CNN/ResNet50)
                            model_name, class_name, confidence, inf_time = classify_image(
                                model, image, selected_model
                            )

                            # Show result card
                            if class_name == "Bird":
                                st.markdown(f"""
                                <div class="result-bird">
                                    <h2>🦅 Bird Detected!</h2>
                                    <h3>Confidence: {confidence:.2%}</h3>
                                    <p>Model: {model_name}</p>
                                    <p>Inference Time: {inf_time:.3f} seconds</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="result-drone">
                                    <h2>🚁 Drone Detected!</h2>
                                    <h3>Confidence: {confidence:.2%}</h3>
                                    <p>Model: {model_name}</p>
                                    <p>Inference Time: {inf_time:.3f} seconds</p>
                                </div>
                                """, unsafe_allow_html=True)

                            # Confidence gauge
                            st.plotly_chart(
                                create_confidence_gauge(confidence, class_name),
                                use_container_width=True
                            )
                else:
                    st.error(f"❌ Model {selected_model} not loaded")

# Tab 3: Model Comparison
with tab3:
    st.markdown("## 📊 Model Performance Comparison")

    # Comparison table
    import pandas as pd

    comparison_data = {
        'Model': ['Custom CNN', 'ResNet50', 'YOLOv8'],
        'Type': ['Deep Learning', 'Transfer Learning', 'Object Detection'],
        'Format': ['PKL', 'PKL', 'PT'],
        'Accuracy/mAP': ['100%', '99.55%', '71.3% mAP@50'],
        'Speed': ['Medium (45 FPS)', 'Slow (30 FPS)', 'Fast (78 FPS)'],
        'Bounding Boxes': ['❌', '❌', '✅'],
        'Best For': ['Balanced performance', 'Highest accuracy', 'Object localization']
    }

    df = pd.DataFrame(comparison_data)

    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # Model recommendations
    st.markdown("## 💡 Model Selection Guide")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="model-card">
            <h4>⚡ Need Speed?</h4>
            <p><b>Use: YOLOv8</b></p>
            <p>78 FPS for real-time detection with bounding boxes.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="model-card">
            <h4>🎯 Need Accuracy?</h4>
            <p><b>Use: Custom CNN</b></p>
            <p>100% accuracy for precise classification.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="model-card">
            <h4>🏆 Best Overall?</h4>
            <p><b>Use: ResNet50</b></p>
            <p>99.55% accuracy with state-of-the-art architecture.</p>
        </div>
        """, unsafe_allow_html=True)

    # Loaded models status
    st.markdown("## 📦 Loaded Models Status")

    status_data = {
        'Model': list(model_status.keys()),
        'Status': list(model_status.values()),
        'Type': [MODEL_INFO[m]['type'] for m in model_status.keys()],
        'File': [MODEL_FILES[m] for m in model_status.keys()]
    }

    status_df = pd.DataFrame(status_data)
    st.dataframe(status_df, use_container_width=True, hide_index=True)

# Tab 4: Documentation
with tab4:
    st.markdown("## 📖 User Guide")

    st.markdown("""
    ### 🚀 Quick Start

    1. **Select a Model** from the sidebar
    2. **Upload an Image** in the Predict tab
    3. **View Results** instantly
    4. **Download** detection results (for YOLOv8)

    ### 📋 Model Files Required

    Place these files in the `models/` directory:

    - `custom_cnn_model.pkl` (~157 MB)
    - `resnet50_transfer_learning_model.pkl` (~189 MB)
    - `best_model_yolov8.pt` (~6 MB)

    **Total:** ~352 MB

    ### 🎯 Model Selection Guide

    **Use Custom CNN when:**
    - You need perfect accuracy (100%)
    - Processing time is not critical
    - Classification only (no bounding boxes needed)

    **Use ResNet50 when:**
    - You want state-of-the-art architecture
    - Accuracy is top priority
    - You have computational resources

    **Use YOLOv8 when:**
    - You need real-time detection
    - Bounding boxes are required
    - Object localization is important
    - Speed is critical (78 FPS)

    ### 💡 Tips for Best Results

    - Use clear, well-lit images
    - Ensure object is visible in the frame
    - Avoid heavily occluded objects
    - For YOLOv8, adjust confidence threshold if needed

    ### ⚙️ Detection Parameters (YOLOv8)

    **Confidence Threshold:**
    - Lower (0.1-0.3): More detections, may include false positives
    - Medium (0.25-0.5): Balanced results (recommended)
    - Higher (0.5-0.9): Fewer detections, higher confidence

    **IoU Threshold:**
    - Controls overlap for Non-Maximum Suppression
    - Default: 0.45 (recommended)
    - Lower: More strict (removes more overlaps)
    - Higher: More lenient (keeps more boxes)

    ### 📊 Performance Metrics

    | Metric | Custom CNN | ResNet50 | YOLOv8 |
    |--------|------------|----------|--------|
    | Accuracy | 100% | 99.55% | 71.3% mAP@50 |
    | Speed | 45 FPS | 30 FPS | 78 FPS |
    | Size | 157 MB | 189 MB | 6 MB |
    | Detection | No | No | Yes |

    ### 🐛 Troubleshooting

    **Model not loading:**
    - Check if model files are in `models/` directory
    - Verify file names match exactly
    - Ensure sufficient RAM (2GB+ recommended)

    **Poor predictions:**
    - Try a different model
    - Ensure image quality is good
    - For YOLOv8, adjust thresholds

    **Slow performance:**
    - Use YOLOv8 for faster inference
    - Reduce image size before upload
    - Close other applications

    ### 📞 Support

    For issues or questions:
    - Check the GitHub repository
    - Review model comparison tab
    - Ensure all dependencies are installed
    """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:2rem;">
        <p style="font-size:1.2rem;">Made with ❤️ using Streamlit</p>
        <p>Deep Learning for Aerial Object Detection</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#666;padding:1rem;">
    <p>🚁 Aerial Object Detection System | 3 Models | Bird vs Drone Classification & Detection</p>
    <p>Custom CNN • ResNet50 • YOLOv8</p>
</div>
""", unsafe_allow_html=True)
