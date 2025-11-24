from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.router import router as airouter
app = FastAPI(
    title="Ticket Support",
    
)

app.include_router(airouter)

app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def docs():
    return RedirectResponse("/docs")
