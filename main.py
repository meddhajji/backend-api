from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import epl, laptops, search, filter, refresh

app = FastAPI(title="Portfolio Backend API")

# Allow Frontend (Localhost:3000) to talk to Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(epl.router, prefix="/api/epl", tags=["EPL"])
app.include_router(laptops.router, prefix="/api/laptops", tags=["Laptops"])
app.include_router(search.router, prefix="/api/search", tags=["Search"])
app.include_router(filter.router)
app.include_router(refresh.router)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Backend is running. Visit /docs for Swagger UI."}


