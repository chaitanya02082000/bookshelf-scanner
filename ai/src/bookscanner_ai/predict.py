import os
import cv2
import numpy as np
import logging
import asyncio
from ultralytics import YOLO
from ultralytics.engine.results import Masks, Boxes
from PIL import Image, ImageEnhance
from typing import Generator, AsyncGenerator, TYPE_CHECKING
from torch import Tensor
from .utils import scale_image, image_to_base64, remove_files

if TYPE_CHECKING:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler


class BookPredictor:
    """
    A class to predict the title and author of books in an image.
    """

    yolo_model: YOLO
    llm: "Llama"
    chat_handler: "Llava15ChatHandler"
    prompt: str
    output_dir = os.path.abspath("output")
    yolo_initialized = False
    llm_initialized = False
    logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        # Ensure the output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/segmentation", exist_ok=True)

        self.prompt = """Recognize the title and author of this book in the format 'Title by Author'.
If there is no author, just the title is fine.
If there's no book in the image, please type 'No book'."""

    def load_models(self) -> None:
        """
        Initialize and load the YOLO and Qwen2.5-VL models.
        """
        self._init_yolo()
        self._init_llm()

    def predict(self, image_path: str) -> tuple[str, Generator[str, None, None]] | None:
        """
        Predict the title and author of books in the image.
        Args:
            image_path (str): The path to the image file.
        Returns:
            tuple[str, Generator[str, None, None]] | None:
            A tuple containing the segmented image (base64 string) and a generator of results.
            If no books are detected, returns None.
        """
        segmentation_result = self._segment_and_prepare_books(image_path)

        if segmentation_result is None:
            self.logger.info("No books detected.")
            return None

        yolo_output, cropped_books = segmentation_result

        def result_generator() -> Generator[str, None, None]:
            for i, image_file in enumerate(cropped_books):
                try:
                    response = self._recognize_book(image_file)
                    output = f"Book {i + 1}: {response}"
                    self.logger.info(output)
                    yield output
                except Exception as e:
                    error_message = f"Error processing book {image_file}: {str(e)}"
                    self.logger.error(error_message)
                    yield error_message

        return yolo_output, result_generator()

    async def predict_async(
        self, image_path: str
    ) -> tuple[str, AsyncGenerator[str, None]] | None:
        """
        Asynchronously predict the title and author of books in the image.
        Args:
            image_path (str): The path to the image file.
        Returns:
            tuple[str, AsyncGenerator[str, None]] | None:
            A tuple containing the segmented image (base64 string) and an async generator of results.
            If no books are detected, returns None.
        """
        self.logger.info(f"Async processing image: {image_path}")

        segmentation_result = await asyncio.to_thread(
            self._segment_and_prepare_books, image_path
        )

        if segmentation_result is None:
            self.logger.info("No books detected.")
            return None

        yolo_output, cropped_books = segmentation_result

        async def result_generator() -> AsyncGenerator[str, None]:
            for i, image_file in enumerate(cropped_books):
                try:
                    response = await asyncio.to_thread(self._recognize_book, image_file)
                    output = f"Book {i + 1}: {response}"
                    self.logger.info(output)
                    yield output
                except Exception as e:
                    error_message = f"Error processing book {image_file}: {str(e)}"
                    self.logger.error(error_message)
                    yield error_message

        return yolo_output, result_generator()

    def cleanup(self) -> None:
        """
        Remove all files in the output directory.
        """
        remove_files(self.output_dir)
        remove_files(f"{self.output_dir}/segmentation")
        self.logger.info("Output directory cleaned up.")

    def _init_yolo(self) -> None:
        """
        Initialize the YOLO segmentation model.
        """
        if self.yolo_initialized:
            return

        # Use a segmentation-capable checkpoint.
        # Default to yolo26n-seg.pt; fall back to a public checkpoint if missing.
        weights = os.getenv("YOLO_SEG_WEIGHTS", "yolo26n-seg.pt")
        if weights != "yolo26n-seg.pt" and not os.path.exists(weights):
            raise FileNotFoundError(
                f"YOLO weights not found at '{weights}'. Set YOLO_SEG_WEIGHTS to a valid path."
            )
        if weights == "yolo26n-seg.pt" and not os.path.exists(weights):
            weights = "yolov8n-seg.pt"
        self.yolo_model = YOLO(weights)
        self.yolo_initialized = True
        self.logger.info("YOLO segmentation model initialized.")

    def _init_llm(self) -> None:
        """
        Initialize Qwen2.5-VL (GGUF) with llama-cpp-python.
        """
        if self.llm_initialized:
            return

        if os.getenv("BOOKSCANNER_DISABLE_LLM", "0").lower() in {"1", "true", "yes"}:
            self.logger.info("LLM disabled via BOOKSCANNER_DISABLE_LLM.")
            return

        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler

        # Vision projector for VL model
        self.chat_handler = Llava15ChatHandler.from_pretrained(
            repo_id="unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            filename="mmproj-F16.gguf",
            local_dir="models/cache/qwen2.5",
            verbose=False,
        )

        # Text model GGUF (balanced quality/speed)
        self.llm = Llama.from_pretrained(
            repo_id="unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            filename="Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
            local_dir="models/cache/qwen2.5",
            chat_handler=self.chat_handler,
            n_ctx=2048,
            n_gpu_layers=0,  # CPU default; set -1 for full GPU offload if CUDA build is available
            verbose=False,
        )

        self.llm_initialized = True
        self.logger.info("Qwen2.5-VL model initialized.")

    def _segment_and_prepare_books(
        self, image_path: str
    ) -> tuple[str, list[str]] | None:
        """
        Segments the image using YOLO, processes each detected book, saves segmented output,
        and returns segmented image as base64 + cropped book file paths.
        """
        original_image = Image.open(image_path).convert("RGB")

        # Scale image if too large
        if original_image.size[0] > 2560 or original_image.size[1] > 2560:
            scaled_image = scale_image(original_image, (2560, 2560))
        else:
            scaled_image = original_image
            # Rotate if landscape
            if scaled_image.width > scaled_image.height:
                scaled_image = scaled_image.rotate(-90, expand=True)

        enhanced_image = self._enhance_image(scaled_image)
        enhanced_image = self._reduce_noise(enhanced_image)

        image_filename = os.path.basename(image_path)
        segmentation_mask_data = self._segment_books(enhanced_image, image_filename)

        if segmentation_mask_data is None:
            return None

        masks, boxes = segmentation_mask_data
        cropped_books: list[str] = []

        # Loop over each detected mask/box
        for i, (mask, box) in enumerate(zip(masks.data, boxes)):  # type: ignore
            mask = mask  # Tensor
            box = box  # Boxes row item

            cropped_image = self._mask_and_crop(enhanced_image, mask, box)
            cropped_image = self._rotate_if_spine(cropped_image, box)

            cropped_image_path = os.path.abspath(f"{self.output_dir}/book_{i + 1}.png")
            cropped_image.save(cropped_image_path)
            cropped_books.append(cropped_image_path)

        segmented_image_path = os.path.join(
            self.output_dir, "segmentation", image_filename
        )
        self.logger.info(f"Segmented image saved to {segmented_image_path}")
        segmented_image_encoded = image_to_base64(segmented_image_path)

        return segmented_image_encoded, cropped_books

    def _segment_books(
        self, image: Image.Image, image_filename: str
    ) -> tuple[Masks, Boxes] | None:
        """
        Segment books in the image using YOLO.
        Returns masks and boxes if detections exist, otherwise None.
        """
        if not self.yolo_initialized:
            raise ValueError(
                "YOLO model is not initialized, please call load_models() first."
            )

        results = self.yolo_model.predict(
            image,
            imgsz=max(image.size[0], image.size[1]),
            half=False,  # safer across CPU/GPU
            classes=[73],  # keep your existing behavior; remove for debugging if needed
            retina_masks=True,
            conf=0.25,
            verbose=False,
        )

        r = results[0]
        masks = r.masks
        boxes = r.boxes

        # Save visualized segmentation result
        r.save(filename=f"{self.output_dir}/segmentation/{image_filename}")

        if masks is None or boxes is None or len(boxes) == 0:
            return None

        return masks, boxes  # type: ignore[return-value]

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Enhance contrast and brightness for better OCR/VLM results.
        """
        image = ImageEnhance.Contrast(image).enhance(1.5)
        image = ImageEnhance.Brightness(image).enhance(1.1)
        return image

    def _reduce_noise(self, image: Image.Image) -> Image.Image:
        """
        Apply noise reduction using OpenCV.
        """
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image_cv = cv2.fastNlMeansDenoisingColored(image_cv, None, 10, 10, 7, 21)
        return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    def _mask_and_crop(
        self, image: Image.Image, mask_data: Tensor, box_data
    ) -> Image.Image:
        """
        Mask and crop the image using the segmentation mask and bounding box.
        """
        mask = Image.fromarray(mask_data.cpu().numpy().astype("uint8") * 255)

        masked_image = Image.new("RGB", image.size)
        masked_image.paste(image, mask=mask)

        x1, y1, x2, y2 = map(int, box_data.xyxy[0])
        cropped_image = masked_image.crop((x1, y1, x2, y2))
        return cropped_image

    def _recognize_book(self, image_path: str) -> str:
        """
        Recognize the title and author from a cropped book image.
        """
        if not self.llm_initialized:
            raise RuntimeError(
                "LLM model is not initialized. Set BOOKSCANNER_DISABLE_LLM=0 to enable."
            )

        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_base64(image_path)},
                        },
                    ],
                }
            ]
        )

        message_content: str = response["choices"][0]["message"]["content"]  # type: ignore
        return message_content.strip()

    def _rotate_if_spine(
        self, image: Image.Image, box_data, threshold: float = 2.0
    ) -> Image.Image:
        """
        Rotate image if it is likely a vertical spine based on aspect ratio.
        """
        x1, y1, x2, y2 = box_data.xyxy[0]
        width, height = (x2 - x1), (y2 - y1)

        # Avoid division by zero in degenerate boxes
        if float(width) <= 0:
            return image

        aspect_ratio = float(height) / float(width)

        if aspect_ratio > threshold:
            return image.rotate(90, expand=True)

        return image
