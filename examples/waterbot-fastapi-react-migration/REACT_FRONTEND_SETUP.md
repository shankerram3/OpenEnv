# React Frontend Setup

This document explains how to build and run the React frontend with the FastAPI backend.

## Overview

The application has been updated to use a React frontend instead of Jinja templates. The FastAPI backend now serves the built React application and provides API endpoints for the chat functionality.

## Building the React Frontend

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn

### Build Steps

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Build the React application:
   ```bash
   npm run build
   ```

   This will create a `dist` folder inside the `frontend` directory with the compiled React application.

## Running the Application

### Development Mode

For development, you can run the React dev server and FastAPI backend separately:

1. Start the FastAPI backend (from the `application` directory):
   ```bash
   cd application
   python main.py
   ```
   The backend will run on `http://localhost:8000`

2. In a separate terminal, start the React dev server (from the `frontend` directory):
   ```bash
   cd frontend
   npm run dev
   ```
   The frontend will run on `http://localhost:5173` with hot-reload enabled

### Production Mode

For production, build the React app and run only the FastAPI server:

1. Build the React app (if not already built):
   ```bash
   cd frontend
   npm run build
   ```

2. Start the FastAPI backend:
   ```bash
   cd application
   python main.py
   ```

   The backend will serve the React app at `http://localhost:8000`

## Changes Made to main.py

The following changes were made to serve the React frontend:

1. **Removed Jinja2Templates**: No longer using template rendering
2. **Added React build mounting**: The `/assets` route now serves React's built assets
3. **Updated routes**:
   - Root route (`/`) serves the React index.html
   - Catch-all route handles React Router's client-side routing
   - All API routes remain unchanged
4. **API endpoints preserved**: All chat and backend functionality remains the same

## API Endpoints

All existing API endpoints are unchanged and accessible:

- `POST /chat_api` - Main chat endpoint
- `POST /riverbot_chat_api` - Riverbot-specific chat
- `POST /chat_detailed_api` - Detailed responses
- `POST /chat_actionItems_api` - Action items
- `POST /chat_sources_api` - Source information
- `POST /submit_rating_api` - Rating submission
- `POST /session-transcript` - Session transcript download
- `WebSocket /transcribe` - Audio transcription
- `GET /messages` - Message history (requires authentication)

## File Structure

```
waterbot/
├── application/
│   ├── main.py (modified to serve React)
│   ├── static/ (backend static files)
│   ├── managers/
│   ├── adapters/
│   └── ...
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   └── ...
│   ├── dist/ (created after build)
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## Troubleshooting

### React app not loading
- Ensure the React app is built: `cd frontend && npm run build`
- Check that the `frontend/dist` folder exists
- Verify the FastAPI server is running from the `application` directory

### API calls failing
- Check that the FastAPI backend is running on port 8000
- Verify environment variables are set correctly
- Check the browser console for error messages

### Static assets not loading
- Ensure the `dist/assets` folder exists after building
- Check the browser network tab for 404 errors
- Verify the assets are being served from `/assets/`

## Environment Variables

Make sure all required environment variables are set in your `.env` file in the `application` directory:

```
DB_HOST=your_db_host
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=your_db_name
MESSAGES_TABLE=your_messages_table
TRANSCRIPT_BUCKET_NAME=your_bucket_name
```

## Docker Deployment

If using Docker, ensure your Dockerfile builds the React app before starting the FastAPI server:

```dockerfile
# Build React app
WORKDIR /app/frontend
RUN npm install && npm run build

# Run FastAPI
WORKDIR /app/application
CMD ["python", "main.py"]
```
