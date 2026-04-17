# Conversational AI Platform

Full-stack implementation of a memory-aware conversational AI workspace using React (Vite + TypeScript) and Flask + PostgreSQL.

## Features Implemented

- Persistent multi-session conversations (stored in SQLite)
- Memory strategy per conversation:
  - buffer
  - summary
  - entity
  - knowledge_graph
- Persona management with custom prompt, personality, and default memory strategy
- Conversation management:
  - create
  - rename
  - search by content/title
  - pin
  - delete
  - export Markdown
  - export PDF
- Entity memory dashboard
- Knowledge graph visualization
- Strategy comparison runner (same flow, multiple memory strategies)

## Backend Setup (Flask)

1. Open terminal in `backend`.
2. Create and activate virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

4. Configure environment variables in `backend/.env`:

   DATABASE_URL=postgresql+psycopg://username:password@localhost:5432/conversational_ai
   GEMINI_API_KEY=your_google_api_key_here
   GEMINI_MODEL=gemini-1.5-flash

5. Run server:

   python run.py

Backend runs on http://localhost:5000.

## Frontend Setup (React)

1. Open terminal in `frontend`.
2. Install dependencies (if needed):

   npm install

3. Start dev server:

   npm run dev

Frontend runs on http://localhost:5173.

Vite proxy forwards `/api` requests to Flask.

## Notes

- PostgreSQL database must exist first; Flask SQLAlchemy auto-creates the tables.
- Assistant replies use Gemini API when `GEMINI_API_KEY` is set.
- If `GEMINI_API_KEY` is missing, backend falls back to deterministic memory-aware responses.
