const canvas = document.getElementById('overlay-canvas');
const ctx = canvas.getContext('2d');
const text =
`Aetheris V.O initialized
Join room

Make sure to share your screen`;

const typingEl = document.getElementById("typing");
let i = 0;

function typeNext() {
    if (i < text.length) {
        typingEl.textContent += text[i++];
        setTimeout(typeNext, 35);
    }
}
typeNext();

document.getElementById("copy-btn").onclick = async () => {
    await navigator.clipboard.writeText(
        document.getElementById("link").textContent
    );
};

let drawings = []; // Store drawings with their timestamps

function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);
resize();

// Track and send cursor position to main process
window.addEventListener('mousemove', (event) => {
    window.electronAPI.sendCursorUpdate(event.clientX, event.clientY);
});

function drawLine(x1, y1, x2, y2, color = 'lime', width = 2, alpha = 1) {
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.globalAlpha = 1;
}

function drawCircle(x, y, radius, color = 'yellow', width = 2, alpha = 1) {
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = 1;
}

// Animation loop
function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const now = Date.now();
    
    drawings = drawings.filter(drawing => {
        const age = now - drawing.timestamp;
        const drawTime = 1000; // 500ms to draw the line and circle
        const fadeTime = 5000; // 5 seconds total (including draw time)
        
        if (age > fadeTime) return false; // Remove old drawings
        
        let opacity = 1;
        let progress = 1;
        
        if (age < drawTime) {
            // Drawing phase: animate from 0 to 1
            progress = age / drawTime;
            opacity = 1;
        } else {
            // Fade phase: fade from 1 to 0
            const fadeAge = age - drawTime;
            const fadeDuration = fadeTime - drawTime;
            opacity = 1 - (fadeAge / fadeDuration);
            progress = 1;
        }
        
        // Calculate current endpoint based on progress
        const currentX = drawing.x1 + (drawing.x2 - drawing.x1) * progress;
        const currentY = drawing.y1 + (drawing.y2 - drawing.y1) * progress;
        
        // Draw line (growing during draw phase, fading during fade phase)
        drawLine(drawing.x1, drawing.y1, currentX, currentY, drawing.color, 3, opacity);
        
        // Draw circle at current endpoint (growing during draw phase, fading during fade phase)
        const currentRadius = 20 * progress;
        drawCircle(currentX, currentY, currentRadius, drawing.color, 3, opacity);
        
        return true;
    });
    
    requestAnimationFrame(animate);
}

animate();

// Listen for draw commands from API
window.electronAPI.onDrawCommand((data) => {
    drawings.push({
        x1: data.x1,
        y1: data.y1,
        x2: data.x2,
        y2: data.y2,
        color: data.color,
        timestamp: Date.now()
    });
});