import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Initialize FastAPI application
app = FastAPI(title="IITM Finance Invoice Extractor API")

# Enable CORS so Cloudflare Workers or Graders can call the endpoint seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize the Gemini Client
# Ensure GEMINI_API_KEY environment variable is set before running
client = genai.Client()

# Define the Incoming Request Payload Schema
class InvoiceRequest(BaseModel):
    invoice_text: str

# Define the Mandatory Output Schema requested by the spec
class InvoiceResponse(BaseModel):
    invoice_no: Optional[str] = Field(default=None, description="The invoice number/reference string.")
    date: Optional[str] = Field(
        default=None, 
        description="The invoice issue date strictly formatted as an ISO string YYYY-MM-DD. Convert words like '15 March 2026' to '2026-03-15'."
    )
    vendor: Optional[str] = Field(default=None, description="The name of the vendor/issuing company.")
    amount: Optional[float] = Field(default=None, description="The subtotal before any taxes are added.")
    tax: Optional[float] = Field(default=None, description="The isolated tax or GST amount only.")
    currency: Optional[str] = Field(default=None, description="The 3-letter currency code, e.g., INR, USD.")

@app.post("/extract", response_model=InvoiceResponse)
async def extract_invoice(payload: InvoiceRequest):
    if not payload.invoice_text.strip():
        raise HTTPException(status_code=400, detail="Invoice text cannot be empty.")
    
    # Prompt explicitly enforcing structural boundaries
    prompt = f"""
    You are an expert financial parsing assistant for the IITM Finance Cell.
    Analyze the following raw invoice text and extract the required fields.
    
    Strict Rules:
    1. Parse and standardize the 'date' field into exactly 'YYYY-MM-DD'. For example, convert '15 March 2026' to '2026-03-15'.
    2. 'amount' must represent the subtotal BEFORE taxes.
    3. 'tax' must represent the tax/GST component amount only. Do not mix it with the grand total.
    4. If any field cannot be identified or is missing in the text, return null for that field.

    Raw Invoice Text:
    ---
    {payload.invoice_text}
    ---
    """
    
    try:
        # Call Gemini using Structured Output features
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=InvoiceResponse,
                temperature=0.0  # Set to 0 for deterministic, extraction-focused results
            ),
        )
        
        # The response text will automatically conform to the Pydantic schema structure
        parsed_response = InvoiceResponse.model_validate_json(response.text)
        return parsed_response

    except Exception as e:
        # Graceful error handling for API failures
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

# Fallback root route for checking deployment health
@app.get("/")
def read_root():
    return {"status": "healthy", "service": "IITM Invoice Extractor"}
