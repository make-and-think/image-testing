from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from Interrogator import Interrogator, SWINV2_MODEL_DSV3_REPO

app = FastAPI()

# Initialize the Interrogator
interrogator = Interrogator()
interrogator.load_model(SWINV2_MODEL_DSV3_REPO)

@app.post("/analyze_image/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        image = io.BytesIO(contents)

        # Set thresholds
        general_thresh = 0.35
        character_thresh = 0.85

        # Predict using the Interrogator
        ratings, general_tags, character_tags = interrogator.predict(image, general_thresh, character_thresh)

        # Prepare the response
        response = {
            "ratings": [{"name": name, "score": float(score)} for name, score in ratings],
            "general_tags": [{"name": name, "score": float(score)} for name, score in general_tags],
            "character_tags": [{"name": name, "score": float(score)} for name, score in character_tags]
        }

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,port=8000)