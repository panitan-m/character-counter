from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import RedirectResponse

import torch
from main import get_char_to_index, predict_letter_occurrence


app = FastAPI(
    title="NLP Engineer Assignment",
    version="1.0.0"
)


@app.get("/", include_in_schema=False)
async def index():
    """
    Redirects to the OpenAPI Swagger UI
    """
    return RedirectResponse(url="/docs")


# TODO: Add a route to the API that accepts a text input and uses the trained
# model to predict the number of occurrences of each letter in the text up to
# that point.

model = torch.load('save_model/model.pt')
char_to_index = get_char_to_index()

class TextInput(BaseModel):
    text: str

# Define endpoint
@app.post("/process_text/")
def process_text(input_data: TextInput):
    prediction = predict_letter_occurrence(model, char_to_index, input_data.text).tolist()
    prediction_text = ''.join(str(x) for x in prediction)
    return {"prediction": prediction_text}

    
