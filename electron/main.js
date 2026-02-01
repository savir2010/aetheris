const { app, BrowserWindow, screen, ipcMain } = require('electron');
const path = require('path');
const express = require('express');
const { uIOhook, UiohookKey } = require('uiohook-napi');
const axios = require('axios');

const server = express();

let win;
let currentCursorX = 0;
let currentCursorY = 0;
let activeDrawings = []; // Track active drawings with timestamps
let clickAlreadySent = new Set(); // Track which drawings already triggered a click

server.use(express.json());

// GET endpoint: cursor + screen size
server.get('/cursor', (req, res) => {
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width, height } = primaryDisplay.size;

  res.status(200).json({ 
    x: currentCursorX, 
    y: currentCursorY,
    screenWidth: width,
    screenHeight: height
  });
});

// POST endpoint: draw line from cursor to given coordinate
server.post('/draw', (req, res) => {
  const { x, y, color } = req.body;
  
  if (win) {
    const drawingId = Date.now(); // Unique ID for this drawing
    
    win.webContents.send('draw-command', { 
      x1: currentCursorX, 
      y1: currentCursorY, 
      x2: x, 
      y2: y, 
      color 
    });
    
    // Add drawing to active list with timestamp and ID
    activeDrawings.push({
      id: drawingId,
      timestamp: drawingId,
      x1: currentCursorX,
      y1: currentCursorY,
      x2: x,
      y2: y,
      color
    });
    
    res.status(200).send({ status: 'Success' });
  } else {
    res.status(500).send({ status: 'Window not found' });
  }
});

server.listen(3000, () => {
  console.log('Overlay API listening on port 3000');
});

// Clean up old drawings (older than 5 seconds)
setInterval(() => {
  const now = Date.now();
  activeDrawings = activeDrawings.filter(drawing => {
    const age = now - drawing.timestamp;
    if (age >= 10000) {
      // Remove from clickAlreadySent set when drawing expires
      clickAlreadySent.delete(drawing.id);
      return false;
    }
    return true;
  });
}, 1000);

// Start global mouse listener with uiohook-napi
uIOhook.on('mousedown', async (event) => {
  console.log('Global mouse click detected at', event.x, event.y);
  
  const now = Date.now();
  
  // Check if there are any active drawings that haven't triggered a click yet
  for (const drawing of activeDrawings) {
    const age = now - drawing.timestamp;
    
    // Only send if drawing is less than 5 seconds old AND hasn't sent a click yet
    if (age < 5000 && !clickAlreadySent.has(drawing.id)) {
      try {
        await axios.post('http://localhost:3001/click', {
          x: event.x,
          y: event.y,
          timestamp: now,
          drawingAge: age,
          drawingId: drawing.id,
          button: event.button // 1=left, 2=right, 3=middle
        });
        console.log(`Click event sent to server for drawing ${drawing.id} (age: ${age}ms)`);
        
        // Mark this drawing as having triggered a click
        clickAlreadySent.add(drawing.id);
        
        // Only send once per click, so break after first match
        break;
      } catch (error) {
        console.error('Error sending click event:', error.message);
      }
    }
  }
});

function createWindow() {
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width, height } = primaryDisplay.workAreaSize;

  win = new BrowserWindow({
    width, height, x: 0, y: 0,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
    type: 'toolbar',
    resizable: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  win.loadFile('index.html');
  win.setIgnoreMouseEvents(true, { forward: true });
  
  // Receive cursor updates from renderer
  ipcMain.on('cursor-update', (event, { x, y }) => {
    currentCursorX = x;
    currentCursorY = y;
  });
}

app.whenReady().then(() => {
  createWindow();
  // Start uiohook after window is ready
  uIOhook.start();
});

app.on('window-all-closed', () => {
  uIOhook.stop();
  if (process.platform !== 'darwin') app.quit();
});

app.on('quit', () => {f
  uIOhook.stop();
});