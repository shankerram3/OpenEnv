# WaterBot FastAPI React Migration

This directory contains the modified `main.py` file and setup instructions for migrating the WaterBot application from Jinja2 templates to a React frontend.

## Overview

The WaterBot application originally used Jinja2 templates for its frontend. This migration updates the FastAPI backend to serve a React single-page application (SPA) instead, while keeping all API endpoints intact.

## Repository

Original Repository: https://github.com/S-Carradini/waterbot/tree/react-frontend

## Changes Made

### 1. Modified `application/main.py`

#### Imports Changed
```python
# Removed:
from fastapi.templating import Jinja2Templates

# Added:
from fastapi.responses import FileResponse
```

#### Template Initialization Removed
```python
# Removed:
templates = Jinja2Templates(directory='templates')

# Added:
app.mount("/assets", StaticFiles(directory="../frontend/dist/assets"), name="assets")
```

#### Routes Updated

**Before:** Multiple routes serving Jinja2 templates
```python
@app.get("/", response_class=HTMLResponse)
async def home(request: Request,):
    return templates.TemplateResponse("splashScreen.html", context)

@app.get("/waterbot", response_class=HTMLResponse)
async def home(request: Request,):
    return templates.TemplateResponse("index.html", context)

@app.get("/aboutwaterbot", response_class=HTMLResponse)
async def home(request: Request,):
    return templates.TemplateResponse("aboutWaterbot.html", context)

# ... more template routes
```

**After:** Single root route and catch-all for React Router
```python
@app.get("/")
async def serve_react_app():
    return FileResponse("../frontend/dist/index.html")

# Catch-all route at the end of file
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    if not full_path.startswith(("chat_", "session-", "transcribe", "submit_", "messages", "riverbot_", "static/", "assets/")):
        return FileResponse("../frontend/dist/index.html")
    raise HTTPException(status_code=404, detail="Not found")
```

### 2. API Endpoints (Unchanged)

All existing API endpoints remain functional:
- `POST /chat_api` - Main chat endpoint
- `POST /riverbot_chat_api` - Riverbot-specific chat
- `POST /chat_detailed_api` - Detailed responses
- `POST /chat_actionItems_api` - Action items
- `POST /chat_sources_api` - Source information
- `POST /submit_rating_api` - Rating submission
- `POST /session-transcript` - Session transcript download
- `WebSocket /transcribe` - Audio transcription
- `GET /messages` - Message history (requires authentication)

### 3. Static File Serving

The application now serves two types of static files:
- `/assets/*` - React build assets (JS, CSS, images from Vite build)
- `/static/*` - Backend static files (kept for backwards compatibility)

## Implementation Guide

### Prerequisites
1. Node.js v16+ and npm installed
2. Python 3.8+ with FastAPI and dependencies
3. Access to the waterbot repository

### Step-by-Step Implementation

1. **Clone and checkout the react-frontend branch:**
   ```bash
   git clone https://github.com/S-Carradini/waterbot.git
   cd waterbot
   git checkout react-frontend
   ```

2. **Build the React frontend:**
   ```bash
   cd frontend
   npm install
   npm run build
   ```
   This creates a `frontend/dist` folder with the compiled React app.

3. **Replace the main.py file:**
   - Copy the modified `main.py` from this directory to `application/main.py`
   - Or manually apply the changes shown above

4. **Test the application:**
   ```bash
   cd application
   python main.py
   ```
   Visit http://localhost:8000 to see the React app

### Development Workflow

**Development Mode** (with hot-reload):
```bash
# Terminal 1 - Backend
cd application
python main.py

# Terminal 2 - Frontend
cd frontend
npm run dev
```
Access frontend at http://localhost:5173

**Production Mode:**
```bash
# Build frontend
cd frontend
npm run build

# Run backend (serves the built React app)
cd ../application
python main.py
```
Access app at http://localhost:8000

## Key Benefits

1. **Modern Frontend**: React provides a better user experience with component-based architecture
2. **Client-Side Routing**: React Router handles navigation without page reloads
3. **Better Developer Experience**: Hot-reload during development, component reusability
4. **API Separation**: Clean separation between frontend and backend
5. **Easier Deployment**: Can deploy frontend and backend separately if needed

## File Structure

```
waterbot/
├── application/
│   ├── main.py (modified - serves React)
│   ├── static/ (backend static files)
│   ├── templates/ (legacy - can be removed)
│   ├── managers/
│   ├── adapters/
│   └── ...
├── frontend/
│   ├── src/
│   │   ├── App.jsx (main React component)
│   │   ├── components/
│   │   ├── services/
│   │   └── ...
│   ├── dist/ (created after npm run build)
│   │   ├── index.html
│   │   └── assets/
│   ├── package.json
│   └── vite.config.js
└── REACT_FRONTEND_SETUP.md (detailed setup guide)
```

## Troubleshooting

### Issue: "FileNotFoundError: ../frontend/dist/index.html"
**Solution:** Build the React app first:
```bash
cd frontend && npm run build
```

### Issue: "404 on /assets/*"
**Solution:** Ensure the dist folder exists and contains an assets subdirectory

### Issue: API calls return CORS errors
**Solution:** Check that the FastAPI CORS middleware is configured if running frontend separately

### Issue: Routes not working (404 errors)
**Solution:**
- Ensure the catch-all route is at the END of main.py
- Verify React Router is configured in the frontend

## Docker Deployment

Update your Dockerfile to build React before starting FastAPI:

```dockerfile
FROM node:18 as frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.9
WORKDIR /app/application
COPY application/ ./
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
```

## Additional Resources

- See `REACT_FRONTEND_SETUP.md` for detailed setup instructions
- React documentation: https://react.dev/
- FastAPI documentation: https://fastapi.tiangolo.com/
- Vite documentation: https://vitejs.dev/

## Notes

- The old Jinja2 templates in `application/templates/` can be safely removed after confirming the React app works
- All backend functionality (session management, cookies, middleware) remains unchanged
- The React app handles all routing - FastAPI just serves index.html and provides API endpoints
