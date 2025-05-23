import cv2
import numpy as np
import vart
import os
import xir
import threading
import time
import sys
import argparse
import subprocess
from datetime import datetime
import glob

divider = '------------------------------------'

def extract_frames_with_ffmpeg(video_path, output_dir, fps=10):
    """
    Extract frames from a video using ffmpeg command line tool.
    This provides a fallback when OpenCV/GStreamer has issues.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        fps: Number of frames per second to extract
        
    Returns:
        True if extraction successful, False otherwise
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct ffmpeg command
    cmd = [
        'ffmpeg', 
        '-i', video_path, 
        '-vf', f'fps={fps}', 
        '-q:v', '2',  # Quality level
        os.path.join(output_dir, 'frame_%04d.jpg')
    ]
    
    try:
        # Run ffmpeg command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if frames were actually extracted
        frames = glob.glob(os.path.join(output_dir, '*.jpg'))
        if not frames:
            print("Warning: ffmpeg ran successfully but no frames were extracted")
            return False
            
        print(f"Successfully extracted {len(frames)} frames to {output_dir}")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames with ffmpeg: {e}")
        print(f"stderr: {e.stderr.decode('utf-8')}")
        return False
    except Exception as e:
        print(f"Error running ffmpeg: {e}")
        return False


def try_extract_frames_with_opencv(video_path, output_dir, fps=None):
    """
    Try to extract frames using OpenCV's VideoCapture.
    Falls back to reporting failure if OpenCV can't open the video.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (None = extract all frames)
        
    Returns:
        True if extraction successful, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"OpenCV couldn't open video: {video_path}")
            return False
            
        os.makedirs(output_dir, exist_ok=True)
        
        frame_count = 0
        saved_count = 0
        
        # Calculate frame interval if fps specified
        if fps is not None:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                print("Warning: Could not determine video FPS, extracting all frames")
                interval = 1
            else:
                interval = max(1, int(video_fps / fps))
        else:
            interval = 1
            
        print(f"Extracting frames with interval of {interval}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % interval == 0:
                output_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(output_path, frame)
                saved_count += 1
                
            frame_count += 1
            
        cap.release()
        
        print(f"Successfully extracted {saved_count} frames to {output_dir}")
        return saved_count > 0
        
    except Exception as e:
        print(f"Error extracting frames with OpenCV: {e}")
        return False


def preprocess_frames_from_directory(frames_dir, resize=(224, 224), max_frames=50):
    """
    Process pre-extracted frames for anomaly detection.
    
    Args:
        frames_dir: Directory containing frame images
        resize: Target size for processed frames
        max_frames: Maximum number of frames to use
        
    Returns:
        Tuple of (preprocessed frames array, display frames array)
    """
    # Get all image files from directory
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    
    if not frame_files:
        print(f"Error: No image files found in {frames_dir}.")
        return None, None
    
    print(f"Found {len(frame_files)} frames")
    
    # Sample frames if there are too many
    if len(frame_files) > max_frames:
        indices = np.linspace(0, len(frame_files)-1, max_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]
    
    frames = []
    display_frames = []
    
    for frame_file in frame_files:
        # Read frame
        frame = cv2.imread(frame_file)
        if frame is None:
            print(f"Warning: Could not read frame {frame_file}")
            continue
            
        original_frame = frame.copy()
        frame_display = cv2.resize(original_frame, resize)
        
        # Process frame for model input
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        # Normalize frame
        frame = frame.astype(np.float32) / 255.0
        frame = frame - 0.5
        frame = frame * 2.0
        
        frames.append(frame)
        display_frames.append(frame_display)
    
    # If we couldn't extract enough frames, duplicate the last ones
    if len(frames) < max_frames and len(frames) > 0:
        print(f"Only found {len(frames)} frames but {max_frames} are needed. Duplicating frames to complete the sequence.")
        while len(frames) < max_frames:
            frames.append(frames[-1])
            display_frames.append(display_frames[-1])
    elif len(frames) == 0:
        print("Error: No valid frames could be loaded.")
        return None, None
    
    return np.array(frames), np.array(display_frames)


def preprocess_video(video_path, resize=(224, 224), max_frames=50, temp_dir=None):
    """
    Process a video for anomaly detection with fallback methods.
    
    Args:
        video_path: Path to the video file
        resize: Target size for processed frames
        max_frames: Maximum number of frames to use
        temp_dir: Directory to store extracted frames (None = auto-generate)
        
    Returns:
        Tuple of (preprocessed frames array, display frames array)
    """
    # Create temp directory for extracted frames if needed
    if temp_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = f"temp_frames_{timestamp}"
    
    os.makedirs(temp_dir, exist_ok=True)
    
    # Try extracting with OpenCV first
    print("Attempting to extract frames with OpenCV...")
    if try_extract_frames_with_opencv(video_path, temp_dir, fps=10):
        print("Successfully extracted frames with OpenCV")
    else:
        # Fall back to ffmpeg
        print("OpenCV extraction failed. Attempting with ffmpeg...")
        if not extract_frames_with_ffmpeg(video_path, temp_dir, fps=10):
            print("All frame extraction methods failed")
            return None, None
    
    # Process the extracted frames
    return preprocess_frames_from_directory(temp_dir, resize, max_frames)


def quantize_frames(frames, input_scale):
    """
    Quantize preprocessed frames according to the DPU requirements
    
    Args:
        frames: Normalized floating-point frames array
        input_scale: Scaling factor for quantization
        
    Returns:
        Int8 quantized frames ready for DPU
    """
    # Apply quantization scaling for DPU
    scaled_frames = frames * input_scale
    # Convert to int8 for DPU processing
    return scaled_frames.astype(np.int8)


def get_child_subgraph_dpu(graph):
    """
    Get DPU subgraph from the compiled model file
    """
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def runDPU(id, start, dpu, frames):
    """
    Run DPU inference on a batch of frames with enhanced error handling
    
    Args:
        id: Thread ID
        start: Starting index in the global output queue
        dpu: DPU runner instance
        frames: Quantized frames to process
    """
    global out_q
    
    try:
        # Get tensors
        inputTensors = dpu.get_input_tensors()
        outputTensors = dpu.get_output_tensors()
        input_ndim = tuple(inputTensors[0].dims)
        output_ndim = tuple(outputTensors[0].dims)
        
        print(f"Thread {id}: Input shape {input_ndim}, Output shape {output_ndim}")
        
        # Prepare for batch processing
        batchSize = input_ndim[0]
        n_of_frames = len(frames)
        count = 0
        write_index = start
        ids = []
        ids_max = 50  # Maximum number of async jobs to track
        outputData = []
        
        for i in range(ids_max):
            outputData.append([np.empty(output_ndim, dtype=np.int8, order="C")])
        
        # Check for shape mismatch and resize if needed
        frame_shape = frames[0].shape
        expected_shape = tuple(input_ndim[1:])
        print(f"Thread {id}: Frame shape {frame_shape}, expected input shape {expected_shape}")
        
        # Calculate total elements to verify dimensions
        expected_elements = np.prod(expected_shape)
        actual_elements = np.prod(frame_shape)
        
        if expected_elements != actual_elements:
            print(f"Thread {id}: Shape mismatch - frame has {actual_elements} elements but model expects {expected_elements}")
            # Reshape frames to match model input shape
            reshaped_frames = []
            for frame in frames:
                # Get original dimensions for proper reshaping
                if len(frame_shape) == 3:
                    h, w, c = frame_shape
                    orig_frame = frame.reshape((h, w, c))
                else:
                    orig_frame = frame
                
                # Resize to match expected input dimensions
                resized = cv2.resize(orig_frame, (expected_shape[1], expected_shape[0]))
                # Add channel dimension if needed
                if len(expected_shape) == 3:
                    resized = resized.reshape(expected_shape)
                reshaped_frames.append(resized)
            frames = np.array(reshaped_frames)
            print(f"Thread {id}: Reshaped frames to {frames[0].shape}")
        
        while count < n_of_frames:
            if (count + batchSize <= n_of_frames):
                runSize = batchSize
            else:
                runSize = n_of_frames - count
                
            # Prepare batch input
            inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
            
            # Initialize input buffer with frame data
            for j in range(runSize):
                frameRun = inputData[0]
                frame_idx = (count + j) % n_of_frames
                
                try:
                    # Copy frame data to input tensor
                    frameRun[j, ...] = frames[frame_idx].reshape(expected_shape)
                except Exception as e:
                    print(f"Thread {id}: Error preparing frame {frame_idx}: {e}")
                    print(f"Frame shape: {frames[frame_idx].shape}, trying to reshape to {expected_shape}")
                    raise
                    
            # Execute async on DPU
            job_id = dpu.execute_async(inputData, outputData[len(ids)])
            ids.append((job_id, runSize, count))
            count = count + runSize
            
            if count < n_of_frames:
                if len(ids) < ids_max - 1:
                    continue
                    
            # Process completed jobs
            for index in range(len(ids)):
                dpu.wait(ids[index][0])
                write_index = ids[index][2] + start
                
                # Store output vectors
                for j in range(ids[index][1]):
                    if write_index < len(out_q):
                        out_q[write_index] = outputData[index][0][j].copy()  # Make a copy to ensure data persistence
                        write_index += 1
                    else:
                        print(f"Thread {id}: Warning - write_index {write_index} out of bounds for out_q (len={len(out_q)})")
                    
            ids = []
        
        print(f"Thread {id}: Successfully processed {count} frames")
        
    except Exception as e:
        print(f"Thread {id}: Exception in runDPU: {e}")
        import traceback
        traceback.print_exc()


def process_data_simple(input_path, model_path, threads=1, output_dir=None, anomaly_threshold=0.5, 
                      resize=(64, 64), max_frames=50, is_directory=False, temp_dir=None):
    """
    Non-threaded implementation to process video/frames for anomaly detection
    
    Args:
        input_path: Path to video file or directory of frames
        model_path: Path to the compiled xmodel file
        threads: Not used in this implementation but kept for API compatibility
        output_dir: Directory to save results and visualizations
        anomaly_threshold: Threshold for determining anomalies
        resize: Frame resize dimensions
        max_frames: Maximum number of frames to process
        is_directory: If True, input_path is a directory of frames
        temp_dir: Directory for temporary frame extraction
        
    Returns:
        Tuple of (prediction text, anomaly score)
    """
    # Process input to extract frames
    if is_directory:
        print(f"Processing frames from directory: {input_path}")
        frames, display_frames = preprocess_frames_from_directory(input_path, resize, max_frames)
    else:
        print(f"Processing video: {input_path}")
        frames, display_frames = preprocess_video(input_path, resize, max_frames, temp_dir)
    
    if frames is None or display_frames is None:
        print("Failed to process input.")
        return None, None
    
    print(f"Processed {len(frames)} frames, shape: {frames.shape}")
    
    # Load model
    g = xir.Graph.deserialize(model_path)
    subgraphs = get_child_subgraph_dpu(g)
    
    if not subgraphs:
        print("Error: No DPU subgraph found in the provided model.")
        return None, None
    
    # Create one DPU runner
    dpu_runner = vart.Runner.create_runner(subgraphs[0], "run")
    
    # Get input/output information
    inputTensors = dpu_runner.get_input_tensors()
    outputTensors = dpu_runner.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)
    
    print(f"Model input shape: {input_ndim}")
    print(f"Model output shape: {output_ndim}")
    
    # Get scaling factors
    input_fixpos = inputTensors[0].get_attr("fix_point")
    input_scale = 2**input_fixpos
    
    if outputTensors[0].has_attr("fix_point"):
        output_fixpos = outputTensors[0].get_attr("fix_point")
        output_scale = 1 / (2**output_fixpos)
    else:
        output_scale = 1.0
        
    # Prepare for processing
    batch_size = input_ndim[0]
    frame_count = len(frames)
    
    # Check if resize needed
    expected_shape = input_ndim[1:]
    if frames[0].shape != expected_shape:
        print(f"Resizing frames from {frames[0].shape} to {expected_shape}")
        resized_frames = []
        for frame in frames:
            # Resize to match expected input dimensions
            resized = cv2.resize(frame, (expected_shape[1], expected_shape[0]))
            resized_frames.append(resized)
        frames = np.array(resized_frames)
        
    # Quantize frames
    quantized_frames = frames * input_scale
    quantized_frames = quantized_frames.astype(np.int8)
    
    # Process frames and get output
    anomaly_scores = []
    time1 = time.time()
    
    for i in range(0, frame_count, batch_size):
        end = min(i + batch_size, frame_count)
        current_batch_size = end - i
        
        # Prepare input and output tensors
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
        outputData = [np.empty(output_ndim, dtype=np.int8, order="C")]
        
        # Fill input tensor with frame data
        for j in range(current_batch_size):
            try:
                inputData[0][j] = quantized_frames[i+j]
            except Exception as e:
                print(f"Error adding frame {i+j} to input tensor: {e}")
                print(f"Frame shape: {quantized_frames[i+j].shape}, expected: {expected_shape}")
                continue
        
        # Execute synchronously
        try:
            job_id = dpu_runner.execute_async(inputData, outputData)
            dpu_runner.wait(job_id)
            
            # Process outputs - FIXED to handle multi-dimensional output
            for j in range(current_batch_size):
                try:
                    # Get the first element of the output for this frame
                    # Check output shape and handle accordingly
                    output_shape = outputData[0].shape
                    
                    # Different approaches based on output tensor shape
                    if len(output_shape) == 4:  # Shape like (batch, 1, 1, features)
                        # If output is 4D, get the value and apply scale
                        output_value = outputData[0][j][0][0][0]  # Take first element
                        if isinstance(output_value, np.ndarray):
                            score = float(output_value[0]) * output_scale
                        else:
                            score = float(output_value) * output_scale
                    elif len(output_shape) == 2:  # Shape like (batch, features)
                        # If output is 2D, get the value and apply scale
                        score = float(outputData[0][j][0]) * output_scale
                    else:
                        # For any other shape, flatten and take first element
                        flat_output = outputData[0][j].flatten()
                        score = float(flat_output[0]) * output_scale
                    
                    # Map to [0,1] range
                    score = max(0, min(1, (score + 1) / 2))
                    anomaly_scores.append(score)
                    
                except Exception as e:
                    print(f"Error processing output for frame {i+j}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
        except Exception as e:
            print(f"Error during DPU execution: {e}")
            import traceback
            traceback.print_exc()
            
    time2 = time.time()
    timetotal = time2 - time1
    fps = float(frame_count / timetotal) if timetotal > 0 else 0
    
    print(divider)
    print(f"Throughput={fps:.2f} fps, total frames = {frame_count}, time={timetotal:.4f} seconds")
    
    if not anomaly_scores:
        print("Error: No valid scores could be calculated")
        return None, None
    
    # Calculate anomaly score and make prediction
    anomaly_score = np.mean(anomaly_scores)
    print(f"Anomaly Score: {anomaly_score:.4f}")
    
    # Determine if anomaly exists
    if anomaly_score < anomaly_threshold:
        prediction_text = f'ANOMALY DETECTED\nConfidence: {(1-anomaly_score)*100:.2f}%'
        anomaly_type = "Anomaly"
    else:
        prediction_text = f'NO ANOMALY DETECTED\nNormal Activity Confidence: {anomaly_score*100:.2f}%'
        anomaly_type = "Normal"
    
    print(prediction_text)
    
    # Create visualization if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Save individual frames only, without matplotlib visualizations
        for i, frame in enumerate(display_frames):
            output_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(output_path, frame)
        
        # Create a simple text file with results
        with open(os.path.join(output_dir, "result.txt"), "w") as f:
            f.write(f"Result: {anomaly_type}\n")
            f.write(f"Anomaly Score: {anomaly_score:.4f}\n")
            if anomaly_score < anomaly_threshold:
                f.write(f"Confidence: {(1-anomaly_score)*100:.2f}%\n")
            else:
                f.write(f"Normal Activity Confidence: {anomaly_score*100:.2f}%\n")
    
    return prediction_text, anomaly_score


def process_data(input_path, model_path, threads=1, output_dir=None, anomaly_threshold=0.5, 
                 resize=(224, 224), max_frames=50, is_directory=False, temp_dir=None):
    """
    Main function to process video/frames for anomaly detection using FPGA acceleration
    
    Args:
        input_path: Path to video file or directory of frames
        model_path: Path to the compiled xmodel file
        threads: Number of threads for parallel processing
        output_dir: Directory to save results and visualizations
        anomaly_threshold: Threshold for determining anomalies
        resize: Frame resize dimensions
        max_frames: Maximum number of frames to process
        is_directory: If True, input_path is a directory of frames
        temp_dir: Directory for temporary frame extraction
        
    Returns:
        Tuple of (prediction text, anomaly score)
    """
    # For stability, use the non-threaded implementation instead
    # of the original threaded one that had issues
    return process_data_simple(input_path, model_path, threads, output_dir, 
                             anomaly_threshold, resize, max_frames, 
                             is_directory, temp_dir)


def create_visualization(frames, anomaly_score, threshold, anomaly_type, output_dir):
    """
    Create and save visualization of the results.
    Uses only OpenCV to save frames, no matplotlib visualization.
    """
    try:
        # Save individual frames
        for i, frame in enumerate(frames):
            output_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(output_path, frame)
        
        # Create a text summary file
        with open(os.path.join(output_dir, "summary.txt"), "w") as f:
            f.write(f"RESULT: {anomaly_type}\n")
            f.write(f"Anomaly Score: {anomaly_score:.4f}\n")
            if anomaly_score < threshold:
                f.write(f"Confidence in anomaly: {(1-anomaly_score)*100:.2f}%\n")
            else:
                f.write(f"Confidence in normal activity: {anomaly_score*100:.2f}%\n")
        
        print(f"Frames saved to {output_dir}")
    except Exception as e:
        print(f"Warning: Could not save all visualization files: {e}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze video/frames for anomalies using FPGA acceleration')
    parser.add_argument('input_path', help='Path to video file or directory of frames')
    parser.add_argument('-m', '--model', type=str, default='anomaly_detection_model.xmodel', 
                        help='Path to the compiled xmodel file')
    parser.add_argument('-t', '--threads', type=int, default=1, 
                        help='Number of threads for parallel processing')
    parser.add_argument('--anomaly_threshold', type=float, default=0.5, 
                        help='Threshold for anomaly detection')
    parser.add_argument('--output_dir', help='Directory to save output frames and visualization')
    parser.add_argument('--max_frames', type=int, default=50, 
                        help='Maximum number of frames to extract/process')
    parser.add_argument('--resize', type=int, nargs=2, default=[224, 224], 
                        help='Resize dimensions for input frames (width height)')
    parser.add_argument('--is_directory', action='store_true',
                        help='Treat input_path as directory of frame images')
    parser.add_argument('--temp_dir', type=str, default=None,
                        help='Directory for temporary frame extraction')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    
    args = parser.parse_args()
    
    print(divider)
    print('Command line options:')
    print(' --input_path : ', args.input_path)
    print(' --model      : ', args.model)
    print(' --threads    : ', args.threads)
    print(' --anomaly_threshold: ', args.anomaly_threshold)
    print(' --max_frames : ', args.max_frames)
    print(' --resize     : ', args.resize)
    print(' --is_directory: ', args.is_directory)
    print(divider)
    
    # Create output directory with timestamp if not specified
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"anomaly_results_{timestamp}"
    
    # Run processing with FPGA acceleration
    prediction, anomaly_score = process_data(
        args.input_path,
        args.model,
        args.threads,
        args.output_dir,
        args.anomaly_threshold,
        tuple(args.resize),
        args.max_frames,
        args.is_directory,
        args.temp_dir
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