import os
import cv2
import numpy as np
import logging
import asyncio
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Masks, Boxes
from PIL import Image, ImageEnhance, ImageDraw
from typing import Generator, AsyncGenerator
from torch import Tensor
from torchvision.ops import nms
from .utils import scale_image, image_to_base64, remove_files

class BookPredictor:
    """
    A class to predict the title and author of books in an image.
    """

    yolo_model: YOLO
    yolo_fallback_model: YOLO | None
    llm: object | None
    chat_handler: object | None
    llm_backend: str
    detector_model: object | None
    detector_processor: object | None
    detector_initialized: bool
    text_backend: str
    ocr_enabled: bool
    prompt: str
    output_dir = os.path.abspath("output")
    yolo_initialized = False
    llm_initialized = False
    logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        # Ensure the output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/segmentation", exist_ok=True)

        self.prompt = """Read the visible book title and author from this image.
The image may show a full front cover or a book spine.
Return only `Title by Author`.
If the author is not visible, return only the title.
If some text is unclear, return the best effort using the most legible text.
Only return `No book` if the image clearly does not contain a book cover or spine."""
        self.llm = None
        self.chat_handler = None
        self.llm_backend = "disabled"
        self.yolo_fallback_model = None
        self.detector_model = None
        self.detector_processor = None
        self.detector_initialized = False
        self.text_backend = "none"
        self.ocr_enabled = False

    def load_models(self) -> None:
        """
        Initialize and load the YOLO and Qwen2.5-VL models.
        """
        self.logger.info(
            "Loading models with env: YOLO_SEG_WEIGHTS=%s YOLO_DET_WEIGHTS=%s BOOKSCANNER_DETECTOR=%s BOOKSCANNER_DETECTOR_MODEL=%s BOOKSCANNER_LLM_BACKEND=%s",
            os.getenv("YOLO_SEG_WEIGHTS"),
            os.getenv("YOLO_DET_WEIGHTS"),
            os.getenv("BOOKSCANNER_DETECTOR"),
            os.getenv("BOOKSCANNER_DETECTOR_MODEL"),
            os.getenv("BOOKSCANNER_LLM_BACKEND"),
        )
        self._init_yolo()
        self._init_llm()
        self._init_detector()

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
        # Default to a larger model for better cover detection.
        weights = os.getenv("YOLO_SEG_WEIGHTS", "yolov8x-seg.pt")
        if weights and os.path.isabs(weights) and not os.path.exists(weights):
            raise FileNotFoundError(
                f"YOLO weights not found at '{weights}'. Set YOLO_SEG_WEIGHTS to a valid path."
            )
        if weights == "yolo26n-seg.pt" and not os.path.exists(weights):
            weights = "yolov8x-seg.pt"
        self.logger.info("Initializing YOLO segmentation weights=%s", weights)
        self.yolo_model = YOLO(weights)

        fallback_weights = os.getenv("YOLO_DET_WEIGHTS", "yolov8x.pt")
        if (
            fallback_weights
            and os.path.isabs(fallback_weights)
            and not os.path.exists(fallback_weights)
        ):
            raise FileNotFoundError(
                f"YOLO fallback weights not found at '{fallback_weights}'. Set YOLO_DET_WEIGHTS to a valid path."
            )
        if fallback_weights:
            self.logger.info(
                "Initializing YOLO detection fallback weights=%s", fallback_weights
            )
            self.yolo_fallback_model = YOLO(fallback_weights)
        self.yolo_initialized = True
        self.logger.info("YOLO segmentation model initialized with %s.", weights)
        if self.yolo_fallback_model:
            self.logger.info(
                "YOLO detection fallback initialized with %s.", fallback_weights
            )

    def _init_llm(self) -> None:
        """
        Initialize a vision-language model using transformers.
        """
        if self.llm_initialized:
            return

        if os.getenv("BOOKSCANNER_DISABLE_LLM", "0").lower() in {"1", "true", "yes"}:
            self.logger.info("LLM disabled via BOOKSCANNER_DISABLE_LLM.")
            return

        from transformers import AutoProcessor, AutoModelForVision2Seq

        model_id = os.getenv("BOOKSCANNER_LLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
        device_pref = os.getenv("BOOKSCANNER_LLM_DEVICE", "auto").lower()

        if device_pref == "cpu":
            device_map = "cpu"
            torch_dtype = torch.float32
        else:
            device_map = "auto"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.tf_model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        self.llm_backend = "transformers"
        self.llm_initialized = True
        self.logger.info("Transformers VLM initialized: %s", model_id)
        return

    def _init_detector(self) -> None:
        """
        Initialize a cover-focused detector using transformers pipeline.
        """
        if self.detector_initialized:
            return

        backend = os.getenv("BOOKSCANNER_DETECTOR", "grounding-dino").lower()
        self.logger.info("Initializing detector backend=%s", backend)
        if backend in {"none", "off", "false"}:
            self.detector_initialized = True
            return

        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        model_id = os.getenv(
            "BOOKSCANNER_DETECTOR_MODEL", "IDEA-Research/grounding-dino-base"
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info("Detector model_id=%s device=%s", model_id, device)
        try:
            self.detector_processor = AutoProcessor.from_pretrained(model_id)
            self.detector_model = (
                AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
                .to(device)
                .eval()
            )
            self.logger.info("Detector initialized: %s", model_id)
        except Exception as e:
            self.detector_processor = None
            self.detector_model = None
            self.logger.warning("Detector init failed for %s: %s", model_id, e)
        finally:
            self.detector_initialized = True


    def _segment_and_prepare_books(
        self, image_path: str
    ) -> tuple[str, list[str]] | None:
        """
        Segments the image using YOLO, processes each detected book, saves segmented output,
        and returns segmented image as base64 + cropped book file paths.
        """
        original_image = Image.open(image_path).convert("RGB")

        # Upscale tiny images for better detection
        min_dim = int(os.getenv("BOOKSCANNER_MIN_DIM", "640"))
        if original_image.size[0] < min_dim or original_image.size[1] < min_dim:
            original_image = self._upscale_min_dim(original_image, min_dim)

        # Scale image if too large
        max_dim = int(os.getenv("BOOKSCANNER_MAX_DIM", "3200"))
        if original_image.size[0] > max_dim or original_image.size[1] > max_dim:
            scaled_image = scale_image(original_image, (max_dim, max_dim))
        else:
            scaled_image = original_image

        if self._should_use_full_image_cover_mode(scaled_image):
            self.logger.info(
                "Using single-cover full-image mode for %s size=%sx%s",
                os.path.basename(image_path),
                scaled_image.size[0],
                scaled_image.size[1],
            )
            return self._full_image_result(scaled_image, os.path.basename(image_path))

        enhanced_image = self._enhance_image(scaled_image)
        enhanced_image = self._reduce_noise(enhanced_image)

        image_filename = os.path.basename(image_path)
        self.logger.info(
            "Segmenting image %s size=%sx%s",
            image_filename,
            enhanced_image.size[0],
            enhanced_image.size[1],
        )
        cover_only = os.getenv("BOOKSCANNER_COVER_ONLY", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        segmentation_mask_data = None
        source = "yolo"

        if cover_only:
            detector_result = self._detect_covers(enhanced_image, image_filename)
            if detector_result is not None:
                segmentation_mask_data = (None, detector_result)
                source = "detector"
                self.logger.info("Using detector-only boxes for %s", image_filename)

        if segmentation_mask_data is None:
            segmentation_mask_data = self._segment_books(enhanced_image, image_filename)

        if segmentation_mask_data is None:
            segmentation_mask_data = self._segment_books(scaled_image, image_filename)
            if segmentation_mask_data is not None:
                enhanced_image = scaled_image

        if segmentation_mask_data is None:
            rotated = enhanced_image.rotate(90, expand=True)
            rotated_name = f"rot_{image_filename}"
            segmentation_mask_data = self._segment_books(rotated, rotated_name)
            if segmentation_mask_data is not None:
                enhanced_image = rotated

        if segmentation_mask_data is None:
            detector_result = self._detect_covers(enhanced_image, image_filename)
            if detector_result is None:
                self.logger.info(
                    "Detector returned no boxes for %s; using full-image fallback",
                    image_filename,
                )
                return self._full_image_result(enhanced_image, image_filename)
            segmentation_mask_data = (None, detector_result)
            source = "detector"

        if segmentation_mask_data is None:
            return None

        masks, boxes = segmentation_mask_data
        detector_data = None
        if source == "detector":
            detector_data = boxes
            boxes = self._boxes_from_detector(detector_data)
            self.logger.info("Skipping box filtering for detector results")
        else:
            boxes = self._select_boxes(boxes, enhanced_image.size)

        if boxes is None or len(boxes) == 0:
            self.logger.info("No boxes after filtering for %s", image_filename)
            return self._full_image_result(enhanced_image, image_filename)
        cropped_books: list[str] = []
        min_crop = int(os.getenv("BOOKSCANNER_MIN_CROP", "96"))

        # Loop over each detected mask/box
        if masks is None or masks.data is None or masks.data.numel() == 0:
            for i, box in enumerate(boxes):  # type: ignore
                if detector_data is not None:
                    cropped_image = self._crop_box_xyxy(enhanced_image, box)
                    cropped_image = self._rotate_if_spine_xyxy(cropped_image, box)
                else:
                    cropped_image = self._crop_box(enhanced_image, box)
                    cropped_image = self._rotate_if_spine(cropped_image, box)

                if min(cropped_image.size) < min_crop:
                    self.logger.info(
                        "Skipping tiny crop %s size=%sx%s",
                        i + 1,
                        cropped_image.size[0],
                        cropped_image.size[1],
                    )
                    continue
                cropped_image_path = os.path.abspath(
                    f"{self.output_dir}/book_{i + 1}.png"
                )
                cropped_image.save(cropped_image_path)
                cropped_books.append(cropped_image_path)
        else:
            for i, (mask, box) in enumerate(zip(masks.data, boxes)):  # type: ignore
                mask = mask  # Tensor
                box = box  # Boxes row item

                cropped_image = self._mask_and_crop(enhanced_image, mask, box)
                cropped_image = self._rotate_if_spine(cropped_image, box)

                if min(cropped_image.size) < min_crop:
                    self.logger.info(
                        "Skipping tiny crop %s size=%sx%s",
                        i + 1,
                        cropped_image.size[0],
                        cropped_image.size[1],
                    )
                    continue
                cropped_image_path = os.path.abspath(
                    f"{self.output_dir}/book_{i + 1}.png"
                )
                cropped_image.save(cropped_image_path)
                cropped_books.append(cropped_image_path)

        segmented_image_path = os.path.join(
            self.output_dir, "segmentation", image_filename
        )
        self.logger.info(f"Segmented image saved to {segmented_image_path}")
        segmented_image_encoded = image_to_base64(segmented_image_path)

        if not cropped_books:
            self.logger.info(
                "All crops too small for %s; using full-image fallback", image_filename
            )
            return self._full_image_result(enhanced_image, image_filename)

        return segmented_image_encoded, cropped_books

    def _full_image_result(
        self, image: Image.Image, image_filename: str
    ) -> tuple[str, list[str]]:
        segmented_image_path = os.path.join(
            self.output_dir, "segmentation", image_filename
        )
        image.save(segmented_image_path)

        fallback_path = os.path.abspath(f"{self.output_dir}/book_1.png")
        image.save(fallback_path)

        self.logger.info(
            "Using full-image fallback crop for %s size=%sx%s",
            image_filename,
            image.size[0],
            image.size[1],
        )
        return image_to_base64(segmented_image_path), [fallback_path]

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

        classes_env = os.getenv("BOOKSCANNER_YOLO_CLASSES", "73").strip()
        classes = None
        if classes_env:
            classes = [int(value) for value in classes_env.split(",")]

        conf = float(os.getenv("BOOKSCANNER_YOLO_CONF", "0.15"))
        imgsz = int(os.getenv("BOOKSCANNER_YOLO_IMGSZ", "2560"))
        iou = float(os.getenv("BOOKSCANNER_YOLO_IOU", "0.5"))
        agnostic_nms = os.getenv("BOOKSCANNER_YOLO_AGNOSTIC", "1").lower() in {
            "1",
            "true",
            "yes",
        }
        self.logger.info(
            "YOLO predict %s imgsz=%s conf=%s iou=%s agnostic=%s classes=%s",
            image_filename,
            imgsz,
            conf,
            iou,
            agnostic_nms,
            classes,
        )

        results = self.yolo_model.predict(
            image,
            imgsz=imgsz,
            half=False,  # safer across CPU/GPU
            classes=classes,
            retina_masks=True,
            conf=conf,
            iou=iou,
            agnostic_nms=agnostic_nms,
            verbose=False,
        )

        r = results[0]
        masks = r.masks
        boxes = r.boxes

        # Save visualized segmentation result
        r.save(filename=f"{self.output_dir}/segmentation/{image_filename}")
        if os.getenv("BOOKSCANNER_DEBUG_DET", "0").lower() in {"1", "true", "yes"}:
            mask_count = 0 if masks is None else len(masks)
            box_count = 0 if boxes is None else len(boxes)
            self.logger.info(
                "YOLO detections for %s: masks=%s boxes=%s",
                image_filename,
                mask_count,
                box_count,
            )

        if masks is None or boxes is None or len(boxes) == 0:
            if classes is not None:
                self.logger.info("YOLO found no boxes; retrying without class filter")
                results = self.yolo_model.predict(
                    image,
                    imgsz=imgsz,
                    half=False,
                    classes=None,
                    retina_masks=True,
                    conf=conf,
                    iou=iou,
                    agnostic_nms=agnostic_nms,
                    verbose=False,
                )
                r = results[0]
                masks = r.masks
                boxes = r.boxes
                r.save(filename=f"{self.output_dir}/segmentation/{image_filename}")

        if masks is None or boxes is None or len(boxes) == 0:
            if self.yolo_fallback_model is None:
                self.logger.info("YOLO found no boxes for %s", image_filename)
                return None

            det_results = self.yolo_fallback_model.predict(
                image,
                imgsz=imgsz,
                half=False,
                classes=classes,
                conf=conf,
                iou=iou,
                agnostic_nms=agnostic_nms,
                verbose=False,
            )
            det = det_results[0]
            if det.boxes is None or len(det.boxes) == 0:
                self.logger.info("YOLO fallback found no boxes for %s", image_filename)
                return None

            det.save(filename=f"{self.output_dir}/segmentation/{image_filename}")
            return Masks(torch.empty((0, 1, 1))), det.boxes

        return masks, boxes  # type: ignore[return-value]

    def _detect_covers(
        self, image: Image.Image, image_filename: str
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]] | None:
        """
        Detect front-facing book covers using a zero-shot detector.
        """
        if (
            not self.detector_initialized
            or self.detector_model is None
            or self.detector_processor is None
        ):
            return None

        text = os.getenv(
            "BOOKSCANNER_DETECTOR_PROMPT",
            "book cover. front cover. book spine. paperback cover. hardcover book.",
        )
        box_threshold = float(os.getenv("BOOKSCANNER_DETECTOR_BOX", "0.2"))
        text_threshold = float(os.getenv("BOOKSCANNER_DETECTOR_TEXT", "0.2"))
        self.logger.info(
            "Detector prompt='%s' box_threshold=%s text_threshold=%s",
            text,
            box_threshold,
            text_threshold,
        )

        inputs = self.detector_processor(
            images=image, text=text, return_tensors="pt"
        ).to(self.detector_model.device)
        with torch.no_grad():
            outputs = self.detector_model(**inputs)

        results = self.detector_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )

        if not results:
            return None

        boxes = results[0]["boxes"]
        scores_t = results[0]["scores"]

        if boxes is None or len(boxes) == 0:
            self.logger.info("Detector produced zero boxes for %s", image_filename)
            return None

        # Filter for likely front covers
        width, height = image.size
        img_area = float(width * height)
        min_area = float(os.getenv("BOOKSCANNER_COVER_MIN_AREA", "0.01"))
        min_aspect = float(os.getenv("BOOKSCANNER_COVER_MIN_ASPECT", "0.2"))
        max_aspect = float(os.getenv("BOOKSCANNER_COVER_MAX_ASPECT", "8.0"))

        box_w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1)
        box_h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1)
        box_area = box_w * box_h
        aspect = box_h / box_w

        keep_mask = (box_area >= (min_area * img_area)) & (
            (aspect >= min_aspect) & (aspect <= max_aspect)
        )
        if keep_mask.sum() == 0:
            self.logger.info(
                "Detector filter too strict for %s; relaxing thresholds",
                image_filename,
            )
            keep_mask = box_area >= (min_area * img_area)
        if keep_mask.sum() == 0:
            keep_mask = torch.ones_like(box_area, dtype=torch.bool)

        boxes = boxes[keep_mask]
        scores_t = scores_t[keep_mask]

        # NMS to collapse duplicates
        iou = float(os.getenv("BOOKSCANNER_COVER_NMS_IOU", "0.5"))
        keep = nms(boxes, scores_t, iou)
        boxes = boxes[keep]
        scores_t = scores_t[keep]

        # Single-cover mode: keep only the best box
        if os.getenv("BOOKSCANNER_SINGLE_COVER", "0").lower() in {"1", "true", "yes"}:
            # Prefer largest area when many boxes remain
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = int(torch.argmax(areas).item())
            boxes = boxes[best_idx : best_idx + 1]
            scores_t = scores_t[best_idx : best_idx + 1]

        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for b in boxes:
            draw.rectangle(b.tolist(), outline="red", width=3)
        overlay.save(f"{self.output_dir}/segmentation/{image_filename}")

        orig_shape = (image.size[1], image.size[0])
        # Return raw tensors to avoid ultralytics Boxes constructor issues.
        return boxes, scores_t, orig_shape

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Enhance contrast and brightness for better OCR/VLM results.
        """
        image = ImageEnhance.Contrast(image).enhance(1.6)
        image = ImageEnhance.Color(image).enhance(1.15)
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

    def _crop_box(self, image: Image.Image, box_data) -> Image.Image:
        """
        Crop the image using only the bounding box.
        """
        x1, y1, x2, y2 = map(int, box_data.xyxy[0])
        return image.crop((x1, y1, x2, y2))

    def _recognize_book(self, image_path: str) -> str:
        """
        Recognize the title and author from a cropped book image.
        """
        if not self.llm_initialized:
            raise RuntimeError(
                "LLM model is not initialized. Set BOOKSCANNER_DISABLE_LLM=0."
            )
        if self.llm_backend != "transformers":
            raise RuntimeError("Unsupported LLM backend configuration.")

        from transformers.image_utils import load_image

        image = self._resize_for_vlm(load_image(image_path))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], images=[image], return_tensors="pt"
        ).to(self.tf_model.device)
        generated_ids = self.tf_model.generate(
            **inputs,
            max_new_tokens=int(os.getenv("BOOKSCANNER_LLM_MAX_TOKENS", "48")),
            do_sample=False,
        )
        prompt_len = inputs["input_ids"].shape[1]
        output = self.processor.batch_decode(
            generated_ids[:, prompt_len:], skip_special_tokens=True
        )[0]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output.strip()

    def _resize_for_vlm(self, image: Image.Image) -> Image.Image:
        """
        Reduce image size to avoid CUDA OOM.
        """
        max_dim = int(os.getenv("BOOKSCANNER_VLM_MAX_DIM", "1280"))
        if image.size[0] <= max_dim and image.size[1] <= max_dim:
            return image
        resized = image.copy()
        resized.thumbnail((max_dim, max_dim))
        return resized

    def _should_use_full_image_cover_mode(self, image: Image.Image) -> bool:
        if os.getenv("BOOKSCANNER_AUTO_FULL_IMAGE", "1").lower() not in {
            "1",
            "true",
            "yes",
        }:
            return False

        width, height = image.size
        if width <= 0 or height <= 0:
            return False

        aspect = height / float(width)
        area = width * height
        return 0.9 <= aspect <= 2.4 and area >= 250_000

    def _upscale_min_dim(self, image: Image.Image, min_dim: int) -> Image.Image:
        width, height = image.size
        scale = max(min_dim / float(width), min_dim / float(height))
        new_size = (int(width * scale), int(height * scale))
        return image.resize(new_size, Image.Resampling.LANCZOS)


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

    def _boxes_from_detector(
        self, detector_data: tuple[torch.Tensor, torch.Tensor, tuple[int, int]]
    ) -> list[tuple[float, float, float, float]]:
        boxes, _, _ = detector_data
        return [tuple(map(float, box.tolist())) for box in boxes]

    def _crop_box_xyxy(
        self, image: Image.Image, box: tuple[float, float, float, float]
    ) -> Image.Image:
        x1, y1, x2, y2 = [int(v) for v in box]
        return image.crop((x1, y1, x2, y2))

    def _rotate_if_spine_xyxy(
        self,
        image: Image.Image,
        box: tuple[float, float, float, float],
        threshold: float = 2.0,
    ) -> Image.Image:
        x1, y1, x2, y2 = box
        width = float(x2 - x1)
        height = float(y2 - y1)

        if width <= 0:
            return image

        aspect_ratio = height / width
        if aspect_ratio > threshold:
            return image.rotate(90, expand=True)
        return image

    def _select_boxes(self, boxes: Boxes, image_size: tuple[int, int]) -> Boxes:
        """
        Filter and prioritize boxes to improve cover detection.
        """
        if boxes is None or len(boxes) == 0:
            return boxes

        width, height = image_size
        img_area = float(width * height)

        xyxy = boxes.xyxy
        scores = boxes.conf

        # Compute areas and aspect ratios
        widths = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=1)
        heights = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=1)
        areas = widths * heights
        aspect = heights / widths

        # Filters: remove very small boxes
        min_area = float(os.getenv("BOOKSCANNER_MIN_BOX_AREA", "0.002"))
        keep = areas >= (min_area * img_area)

        # Prefer book-like boxes, including tall spines.
        max_aspect = float(os.getenv("BOOKSCANNER_MAX_ASPECT", "8.0"))
        min_aspect = float(os.getenv("BOOKSCANNER_MIN_ASPECT", "0.2"))
        keep = keep & (aspect <= max_aspect) & (aspect >= min_aspect)

        if keep.sum() == 0:
            keep = areas >= (min_area * img_area)

        kept_idx = keep.nonzero(as_tuple=False).squeeze(1)
        xyxy = xyxy[kept_idx]
        scores = scores[kept_idx]

        # NMS for cleaner boxes
        iou = float(os.getenv("BOOKSCANNER_NMS_IOU", "0.6"))
        nms_idx = nms(xyxy, scores, iou)

        final_idx = kept_idx[nms_idx]
        selected = boxes[final_idx]

        if os.getenv("BOOKSCANNER_SINGLE_COVER", "0").lower() in {"1", "true", "yes"}:
            xyxy = selected.xyxy
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            best_idx = int(torch.argmax(areas).item())
            return selected[best_idx : best_idx + 1]

        return selected
