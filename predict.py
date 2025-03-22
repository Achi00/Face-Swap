import os

from fastapi import UploadFile
os.environ["ORT_DISABLE_CPU_FALLBACK"] = "1"
from cog import BasePredictor, Input, Path
import cv2
import onnxruntime as ort
import numpy as np
import base64
from gfpgan import GFPGANer, torch
import insightface
from insightface.app import FaceAnalysis
import requests
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from io import BytesIO
import logging
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
import threading
import time
from concurrent.futures import ThreadPoolExecutor


@dataclass
class FaceDetectionResult:
    success: bool
    faces: List[any]
    closest_face: Optional[any]
    error_message: Optional[str] = None

class Predictor(BasePredictor):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

    def _gpu_monitor(self, stop_event):
        """GPU monitoring function to run in a thread"""
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        try:
            while not stop_event.is_set():
                util = nvmlDeviceGetUtilizationRates(handle)
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                self.logger.info(
                    f"GPU Utilization: {util.gpu}% | "
                    f"GPU Memory: {mem_info.used/mem_info.total*100:.1f}%"
                )
                time.sleep(1)
        finally:
            nvmlShutdown()

    def setup(self):
        self.logger.info("Starting model setup")
        models_dir = "model_store"
        self.inswapper_path = os.path.join(models_dir, 'inswapper_128.onnx')
        self.gfpgan_path = os.path.join(models_dir, 'GFPGANv1.4.pth')

        self.logger.info("Initializing face analysis model...")
        # Optimize ONNX Runtime session options
        session_options = ort.SessionOptions()
        session_options.enable_cpu_mem_arena = False
        session_options.enable_mem_pattern = False
        session_options.intra_op_num_threads = 2  # Match your VPS CPU cores
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Initialize FaceAnalysis using only DirectML.
        # (If the underlying library adds CPU fallback, you might not be able to see the session details.)
        self.face_analysis = FaceAnalysis(
            name='buffalo_l', 
            root=models_dir,
            session_options=session_options
            # providers=['CUDAExecutionProvider']
        )
        self.face_analysis.prepare(ctx_id=-1, det_size=(320, 320))

        # Try to print providers if available (this might not be accessible via the FaceAnalysis API)

        self.logger.info("Initializing GFPGAN model...")
        self.gfpgan = GFPGANer(
            model_path=self.gfpgan_path,
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            # device=torch.device('cuda')
        )

        self.logger.info("Loading face swapper model...")
        self.swapper = insightface.model_zoo.get_model(
            self.inswapper_path,
            download=False,
            download_zip=False,
            session_options=session_options
            # providers=['CUDAExecutionProvider']
        )

        self.logger.info("All models loaded successfully")
        
    def detect_faces_in_image(self, img: np.ndarray) -> FaceDetectionResult:
        """
        Detect faces in image and find the closest/largest face if multiple exist.
        
        Args:
            img: numpy array of the image
            
        Returns:
            FaceDetectionResult containing detection results and any error messages
        """
        try:
            faces = self.face_analysis.get(img)
            
            if len(faces) == 0:
                return FaceDetectionResult(
                    success=False,
                    faces=[],
                    closest_face=None,
                    error_message="No face detected"
                )

            if len(faces) == 1:
                return FaceDetectionResult(
                    success=True,
                    faces=faces,
                    closest_face=faces[0]
                )

            # Multiple faces - find the closest (largest) one
            closest_face = max(faces, key=lambda face: 
                (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
            
            return FaceDetectionResult(
                success=True,
                faces=faces,
                closest_face=closest_face
            )

        except Exception as e:
            self.logger.error(f"Face detection error: {str(e)}")
            return FaceDetectionResult(
                success=False,
                faces=[],
                closest_face=None,
                error_message=f"Face detection error: {str(e)}"
            )

    def load_image(self, image_input: Union[Path, str]) -> np.ndarray:
        """
        Load image from either a file path or URL.
        
        Args:
            image_input: Either a Path object or URL string
            
        Returns:
            numpy array of the loaded image
        """
        try:
        # Load image based on input type
            if isinstance(image_input, str) and (image_input.startswith('http://') or   image_input.startswith('https://')):
                # Handle URL input
                response = requests.get(image_input)
                img_array = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            else:
                # Handle file path input
                img = cv2.imread(str(image_input))

            if img is None:
                raise ValueError("Failed to load image")

            return img
        
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")

    def predict(
        self,
        operation: str = Input(description="Operation to perform: 'swap-face' or 'enhance-face'"),
        source_image: Union[str, UploadFile] = Input(description="Background image where face will be placed"),
        target_image: Union[Path, str] = Input(
            description="Image containing the face to transplant (file upload or URL)",
            default=None
        ),
        enhance_result: bool = Input(description="Apply GFPGAN enhancement to result", default=True)
    ) -> Dict[str, Any]:
        """
        Process images based on the requested operation.
        
        Args:
            operation: Type of operation to perform
            source_image: Source image input
            target_image: Target image input (for face swap)
            enhance_result: Whether to apply GFPGAN enhancement
            
        Returns:
            Dictionary containing processed image and operation details
        """
        # stop_event = threading.Event()
        # monitor_thread = None
        try:
            # monitor_thread = threading.Thread(
            #     target=self._gpu_monitor, 
            #     args=(stop_event,)
            # )
            # monitor_thread.start()
            # self.logger.info("Started GPU monitoring")        
            # Load source image
            source_img = self.load_image(source_image)
            target_img = self.load_image(target_image)
            
            result = None
            operation_details = {}
            
            if operation == "swap-face":
                if not target_image:
                    raise ValueError("Target image is required for face swapping")
                
                # Load target image
                target_img = self.load_image(target_image)
                
                # Detect faces in both images
                source_result = self.detect_faces_in_image(source_img)
                target_result = self.detect_faces_in_image(target_img)
                
                if not source_result.success or not target_result.success:
                    raise ValueError(f"Source: {source_result.error_message}, Target: {target_result.error_message}")
                
                # Perform face swap
                result = self.swapper.get(
                    source_img,
                    source_result.closest_face,
                    target_result.closest_face,
                    paste_back=True
                )
                
                # Add operation details
                operation_details = {
                    'source_faces': len(source_result.faces),
                    'target_faces': len(target_result.faces),
                    'using_closest_faces': len(source_result.faces) > 1 or len(target_result.faces) > 1
                }
                
            elif operation == "enhance-face":
                result = source_img
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            # Enhance with GFPGAN if requested
            if enhance_result:
                _, _, result = self.gfpgan.enhance(
                    result,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )
            
            # Save result
            # output_path = Path("output.jpg")
            # cv2.imwrite(str(output_path), result)
            
            # Convert to base64 for response
            _, buffer = cv2.imencode('.jpg', result)
            result_base64 = base64.b64encode(buffer).decode()

            
            return {
                'success': True,
                'result_image': result_base64,
                'operation': operation,
                'operation_details': operation_details,
                'enhanced': enhance_result
            }
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

        # finally:
        #     # Stop and clean up monitoring thread
        #     if monitor_thread:
        #         stop_event.set()
        #         monitor_thread.join()
        #         self.logger.info("Stopped GPU monitoring")