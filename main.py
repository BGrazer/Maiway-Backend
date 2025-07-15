from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import functionalities from other scripts
from rfr import check_fare_anomaly
from chatbot_model import ChatbotModel
from crowd_analysis import analyze_route_with_reference_model

app = FastAPI()

# CORS configuration
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RFR ---
class FarePredictionRequest(BaseModel):
    vehicle_type: str
    distance_km: float
    charged_fare: float
    discounted: bool

@app.post("/predict_fare")
def predict_fare(request: FarePredictionRequest):
    return check_fare_anomaly(request.vehicle_type, request.distance_km, request.charged_fare, request.discounted)

# --- Chatbot ---
chatbot = ChatbotModel()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="No 'message' key provided in JSON body or message is empty")
    response = chatbot.get_response(request.message)
    return {"response": response}

@app.get("/dynamic_suggestions")
def get_dynamic_suggestions(query: str = ""):
    if not query:
        return {"suggestions": []}
    suggestions = chatbot.get_matching_questions(query)
    return {"suggestions": suggestions}

class FaqRequest(BaseModel):
    question: str
    answer: str

@app.post("/admin/add_faq")
def add_faq(request: FaqRequest):
    if not request.question or not request.answer:
        raise HTTPException(status_code=400, detail="Both 'question' and 'answer' are required.")
    success = chatbot.add_faq(request.question, request.answer)
    if success:
        return {"message": "FAQ added and chatbot knowledge base updated."}
    else:
        return {"message": "FAQ (or similar question) already exists."}

@app.post("/admin/reload_chatbot")
def reload_chatbot():
    chatbot.reload_data()
    return {"message": "Chatbot data reloaded successfully."}

# --- Crowd Analysis ---
@app.post("/analyze_crowd")
def analyze_crowd():
    try:
        # This will print to the console of the Railway app
        analyze_route_with_reference_model()
        return {"message": "Crowd analysis started. Check logs for details."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)