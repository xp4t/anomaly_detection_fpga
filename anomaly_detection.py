import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime

def preprocess_video(video_path, resize=(224, 224), max_frames=50):
    """
    Process a single video file and extract frames for anomaly detection.
    Modified from your original code with better error handling.
    """
    video_path = video_path.decode("utf-8") if isinstance(video_path, bytes) else video_path
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    # If video is too short, read all frames
    if total_frames <= max_frames:
        interval = 1
    else:
        interval = max(int(total_frames / max_frames), 1)  # Distribute frames evenly
        
    print(f"Reading every {interval} frame(s)")
    
    frames = []
    display_frames = []
    
    frame_count = 0
    while len(frames) < max_frames and frame_count < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame at position {frame_count}")
            break
        
        original_frame = frame.copy()
        frame_display = cv2.resize(original_frame, resize)
        
        # Process frame for model input
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        frame = frame.astype(np.float32) / 255.0
        
        frames.append(frame)
        display_frames.append(frame_display)
        
        frame_count += interval
    
    cap.release()
    
    # If we couldn't extract enough frames, duplicate the last ones
    if len(frames) < max_frames:
        print(f"Video has {len(frames)} frames but {max_frames} are needed. Duplicating frames to complete the sequence.")
        while len(frames) < max_frames:
            frames.append(frames[-1])
            display_frames.append(display_frames[-1])
    
    # Reshape for model input
    frames_for_prediction = np.array([frames], dtype=np.float32)
    
    return frames_for_prediction, np.array(display_frames)

def analyze_video(video_path, anomaly_threshold=0.5, output_dir=None):
    """
    Analyze a video file for anomalies using the loaded model.
    
    Args:
        video_path: Path to the video file
        anomaly_threshold: Threshold for anomaly detection (lower value means more likely to be anomaly)
        output_dir: Directory to save output frames and visualization
        
    Returns:
        Tuple of (prediction_text, anomaly_score)
    """
    print(f"Loading model...")
    
    try:
        # Load only the anomaly detection model
        anomaly_model = tf.keras.models.load_model('anomaly_detection_model.h5', 
                                                   custom_objects={'KerasLayer': hub.KerasLayer})
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
    print(f"Processing video: {video_path}")
    
    # Preprocess video
    frames_for_prediction, frames_for_display = preprocess_video(video_path)
    
    if frames_for_prediction is None or frames_for_display is None:
        print("Failed to process video.")
        return None, None
    
    print(f"Shape of frames for prediction: {frames_for_prediction.shape}")
    
    # Run anomaly detection
    anomaly_prediction = anomaly_model.predict(frames_for_prediction)[0][0]
    print(f"Anomaly Prediction: {anomaly_prediction:.4f}")
    
    # Process prediction
    if anomaly_prediction < anomaly_threshold:
        prediction_text = f'ANOMALY DETECTED\nConfidence: {(1-anomaly_prediction)*100:.2f}%'
        anomaly_type = "Anomaly"
    else:
        prediction_text = f'NO ANOMALY DETECTED\nNormal Activity Confidence: {anomaly_prediction*100:.2f}%'
        anomaly_type = "Normal"

    # Create visualization of the results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        create_visualization(frames_for_display, anomaly_prediction, anomaly_threshold, anomaly_type, output_dir)
    
    return prediction_text, anomaly_prediction

def create_visualization(frames, anomaly_score, threshold, anomaly_type, output_dir):
    """
    Create and save visualization of the results.
    
    Args:
        frames: Processed video frames
        anomaly_score: Anomaly detection score
        threshold: Anomaly detection threshold
        anomaly_type: Detected anomaly type
        output_dir: Directory to save output
    """
    # Save individual frames
    for i, frame in enumerate(frames):
        output_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        cv2.imwrite(output_path, frame)
    
    # Create composite visualization of frames
    num_frames = min(len(frames), 25)  # Show up to 25 frames in a 5x5 grid
    rows = int(np.ceil(np.sqrt(num_frames)))
    cols = int(np.ceil(num_frames / rows))
    
    plt.figure(figsize=(15, 15))
    for i in range(num_frames):
        plt.subplot(rows, cols, i+1)
        plt.imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "frame_grid.png"))
    
    # Create anomaly visualization
    plt.figure(figsize=(10, 5))
    plt.text(0.5, 0.5, 
             f"Result: {anomaly_type}\n" +
             f"Anomaly Score: {anomaly_score:.4f}",
             ha='center', va='center', fontsize=20, 
             bbox=dict(facecolor='red' if anomaly_score < threshold else 'green', alpha=0.5))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "anomaly_result.png"))
    
    # Create an overview image with result and sample frames
    plt.figure(figsize=(15, 10))
    
    # Add result text
    plt.subplot(2, 3, 1)
    color = 'red' if anomaly_score < threshold else 'green'
    plt.text(0.5, 0.5, 
             f"RESULT: {anomaly_type}\n" +
             f"Anomaly Score: {anomaly_score:.4f}",
             ha='center', va='center', fontsize=12, 
             bbox=dict(facecolor=color, alpha=0.5))
    plt.axis('off')
    
    # Add sample frames
    sample_indices = np.linspace(0, len(frames)-1, 5, dtype=int)
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 3, i+2)
        plt.imshow(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB))
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary.png"))
    
    print(f"Visualization saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze video for anomalies')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--anomaly_threshold', type=float, default=0.5, help='Threshold for anomaly detection')
    parser.add_argument('--output_dir', help='Directory to save output frames and visualization')
    
    args = parser.parse_args()
    
    # Create output directory with timestamp if not specified
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"anomaly_results_{timestamp}"
    
    # Run analysis
    prediction, anomaly_score = analyze_video(
        args.video_path, 
        args.anomaly_threshold,
        args.output_dir
    )
    
    if prediction:
        print("\n" + "="*50)
        print("ANALYSIS RESULT:")
        print("="*50)
        print(prediction)
        print("="*50)
        print(f"Results saved to: {args.output_dir}")
    else:
        print("Analysis failed")

if __name__ == "__main__":
    main()
