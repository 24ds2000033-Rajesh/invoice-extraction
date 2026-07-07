import os
import traceback
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# 1. Initialize FastAPI application
app = FastAPI(title="IITM Finance Invoice Extractor API")

# 2. Enable CORS globally so the Cloudflare Worker grader can access it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# 3. Initialize the GenAI Client safely
# We read the environment variable directly to provide explicit errors if it's missing
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("\n[CRITICAL WARNING]: GEMINI_API_KEY environment variable is not set!")
    print("Please set it using: export GEMINI_API_KEY='your_key'\n")

client = genai.Client()

# 4. Define Request Schema
class InvoiceRequest(BaseModel):
    invoice_text: str

# 5. Define Mandatory Output Schema (Enforcing strict key presence and types)
class InvoiceResponse(BaseModel):
    invoice_no: Optional[str] = Field(default=None, description="The alphanumeric invoice number or reference code.")
    date: Optional[str] = Field(
        default=None, 
        description="The invoice date normalized to ISO format YYYY-MM-DD. Convert textual months (e.g., '15 March 2026' to '2026-03-15')."
    )
    vendor: Optional[str] = Field(default=None, description="The name of the vendor or company issuing the invoice.")
    amount: Optional[float] = Field(default=None, description="The subtotal value of items BEFORE tax or GST is added.")
    tax: Optional[float] = Field(default=None, description="The isolated tax or GST amount only. Do not include subtotal or grand total.")
    currency: Optional[str] = Field(default=None, description="The 3-letter currency code, e.g., INR, USD, EUR.")


@app.post("/extract", response_model=InvoiceResponse)
async def extract_invoice(payload: InvoiceRequest):
    # Quick guard for empty payloads
    if not payload.invoice_text or not payload.invoice_text.strip():
        raise HTTPException(status_code=400, detail="Invoice text cannot be empty.")
    
    # Construction of explicit prompt constraints
    prompt = f"""
    You are an automated extraction assistant for the IITM Finance Cell.
    Analyze the provided invoice text and map it directly into the requested JSON schema structure.
    
    Strict Guidelines:
    1. Parse dates cleanly. Convert values like '15 March 2026' or '22/01/2026' strictly into 'YYYY-MM-DD'.
    2. Separator check: 'amount' must only capture the subtotal before tax. 'tax' must capture the exact tax/GST number.
    3. If a field cannot be derived, leave it as null. Do not hallucinate values.

    Invoice Text Content:
    ---
    {payload.invoice_text}
    ---
    """
    
    try:
        # Call Gemini using native structural constraints
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=InvoiceResponse,
                temperature=0.0  # Force deterministic output
            ),
        )
        
        # Parse the JSON string coming back from Gemini into our Pydantic model
        parsed_response = InvoiceResponse.model_validate_json(response.text)
        return parsed_response

    except Exception as e:
        # --- VERBOSE LOGGING FOR DEBUGGING 500 ERRORS ---
        print("\n" + "="*60)
        print("ERROR DETECTED DURING /extract POST REQUEST")
        print("="*60)
        print(f"Exception Type: {type(e).__name__}")
        print(f"Exception Message: {str(e)}")
        print("\nFull Execution Traceback:")
        traceback.print_exc()
        print("="*60 + "\n")
        
        # Bubble up a descriptive error string back to your caller/grader log
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Server Error: [{type(e).__name__}] {str(e)}"
        )

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "IITM Invoice Extractor"}
