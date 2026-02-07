# Backend API

FastAPI backend powering the Avito Laptops price predictor and EPL soccer prediction tools.

## Setup

1. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   - Copy `.env.example` to `.env`
   - Fill in your API keys

4. **Run the server**:
   ```bash
   uvicorn main:app --reload
   ```

## Project Structure

```
backend-api/
├── main.py           # FastAPI app entry point
├── db.py             # Supabase database client
├── routers/          # API route handlers
│   ├── laptops.py    # Laptop CRUD operations
│   ├── search.py     # Semantic search endpoint
│   ├── filter.py     # Filter options
│   ├── refresh.py    # Scraper trigger
│   └── epl.py        # EPL predictions
├── scraper/          # Avito scraping pipeline
│   ├── scraper.py    # Main scraper
│   ├── parser.py     # Spec extraction
│   └── ...
├── epl/              # EPL predictor module
└── scripts/          # Utility scripts
```

## API Docs

Visit `/docs` for interactive Swagger UI.
