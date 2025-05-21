# RetailSense

A modern retail analytics dashboard with video processing, heatmap generation, and visitor analytics.

---

## Project Structure

```
IT332-RetailSense/
  backend/         # Python Flask backend
  frontend/        # React frontend
  ...
```

---

## Prerequisites

- **Python 3.8+** (for backend)
- **Node.js 18+ & npm** (for frontend)
- **Git**

---

## Backend Setup (Flask)

1. **Navigate to backend:**
   ```sh
   cd IT332-RetailSense/backend
   ```
2. **Create virtual environment:**
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   - Copy `.env.example` to `.env` and fill in values (e.g., secret keys, DB path).
5. **Run the backend:**
   ```sh
   flask run
   # or
   python app.py
   ```
6. **API will be available at** `http://localhost:5000`

---

## Frontend Setup (React)

1. **Navigate to frontend:**
   ```sh
   cd IT332-RetailSense/frontend
   ```
2. **Install dependencies:**
   ```sh
   npm install
   ```
3. **Set up environment variables:**
   - Copy `.env.example` to `.env` and set the backend API URL (e.g., `VITE_API_URL=http://localhost:5000/api`)
4. **Run the frontend:**
   ```sh
   npm run dev
   ```
5. **App will be available at** `http://localhost:5173`

---

## Development Tips
- Use the provided `.gitignore` to avoid committing sensitive or build files.
- For production, set strong secrets in `.env` files and use a production-ready server.
- To reset analytics, clear `project_results/` and `project_uploads/` folders and the jobs table in your DB.

---

## License
MIT (or specify your license) 