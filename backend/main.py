from fastapi import FastAPI

app = FastAPI(
    title="Medical Report Companion API",
    description="Evidence-grounded medical report explanation API",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "Medical Report Companion API",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
    