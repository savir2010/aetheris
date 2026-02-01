const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  onDrawCommand: (callback) => ipcRenderer.on('draw-command', (event, data) => callback(data)),
  sendCursorUpdate: (x, y) => ipcRenderer.send('cursor-update', { x, y })
});