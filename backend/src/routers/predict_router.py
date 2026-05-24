import asyncio
import logging
import os
import shutil
import time
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import StreamingResponse
from bookscanner_ai import BookPredictor
from src.models import ResultWithData

predict_router = APIRouter(prefix="/predict", tags=["predict"])
router = predict_router

# Initialize the BookPredictor
book_predictor = BookPredictor()
book_predictor.load_models()


@router.post("/")
async def predict(request: Request, file: UploadFile = File(...)) -> StreamingResponse:
    """
    Predict the title and author of the books in the uploaded image file.
    Args:
        request (Request): The request object.
        file (UploadFile): The uploaded image file.
    Returns:
        StreamingResponse: The predicted titles and authors of the books.
    """
    logger = logging.getLogger()

    # Save the uploaded file to a temporary location
    temp_image_path = (
        f"{book_predictor.output_dir}/temp_{int(time.time())}_{file.filename}"
    )

    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        prediction_result = await book_predictor.predict_async(temp_image_path)

        if not prediction_result:
            return StreamingResponse(
                iter(
                    [
                        ResultWithData.fail("No books detected.").model_dump_json(
                            by_alias=True
                        )
                        + "\n"
                    ]
                ),
                media_type="application/json",
            )

        segmented_output, result_generator = prediction_result
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        error_result = ResultWithData.fail(str(e))

        return StreamingResponse(
            iter([error_result.model_dump_json(by_alias=True) + "\n"]),
            media_type="application/json",
        )

    async def stream_generator():
        client_disconnected = False
        try:
            # First, send the segmented image
            if segmented_output:
                image_result = ResultWithData[str].succeed(segmented_output)
            else:
                image_result = ResultWithData[str].fail("No books detected.")

            logger.info("Sent segmented image")
            yield image_result.model_dump_json(by_alias=True) + "\n"

            # If no books detected, stop processing
            if not segmented_output:
                logger.info("No books detected. Stopping processing.")
                return

            # Then, send the prediction results
            if not book_predictor.llm_initialized:
                logger.warning("LLM disabled; returning segmentation only.")
                return

            async for result in result_generator:
                if await request.is_disconnected():
                    client_disconnected = True
                    break

                prediction_result = ResultWithData[str].succeed(result)
                yield prediction_result.model_dump_json(by_alias=True) + "\n"
        except Exception as e:
            error_result = ResultWithData.fail(str(e))
            logger.error(f"An error occurred during streaming: {str(e)}")
            yield error_result.model_dump_json(by_alias=True) + "\n"
        finally:
            if client_disconnected:
                logger.info("Client disconnected. Stopping processing.")

            preserve_outputs = os.getenv("BOOKSCANNER_PRESERVE_OUTPUTS", "0").lower() in {
                "1",
                "true",
                "yes",
            }
            if preserve_outputs:
                logger.info("Preserving output directory for inspection.")
                marker_path = os.path.join(book_predictor.output_dir, "_modal_last_request.txt")
                output_files: list[str] = []
                for root, _, files in os.walk(book_predictor.output_dir):
                    for file_name in files:
                        full_path = os.path.join(root, file_name)
                        output_files.append(os.path.relpath(full_path, book_predictor.output_dir))

                with open(marker_path, "w", encoding="utf-8") as marker_file:
                    marker_file.write(f"timestamp={int(time.time())}\n")
                    marker_file.write(f"output_dir={book_predictor.output_dir}\n")
                    marker_file.write("files=\n")
                    for relative_path in sorted(output_files):
                        marker_file.write(f"- {relative_path}\n")

                output_volume = getattr(request.app.state, "output_volume", None)
                if output_volume is not None:
                    output_volume.commit()
                    logger.info("Committed preserved outputs to mounted volume.")
            else:
                # Clean up output directory
                book_predictor.cleanup()

    return StreamingResponse(stream_generator(), media_type="application/json")
