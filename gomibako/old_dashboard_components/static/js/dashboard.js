document.addEventListener('DOMContentLoaded', () => {
    const socket = io();

    // Telemetry data elements
    const rollSpan = document.getElementById('roll');
    const pitchSpan = document.getElementById('pitch');
    const yawSpan = document.getElementById('yaw');
    const altitudeSpan = document.getElementById('altitude');
    const auxLabels = [
        document.getElementById('aux1'),
        document.getElementById('aux2'),
        document.getElementById('aux3'),
        document.getElementById('aux4')
    ];

    // Canvas elements
    const adiCanvas = document.getElementById('adiCanvas');
    const adiCtx = adiCanvas.getContext('2d');
    const leftStickCanvas = document.getElementById('leftStickCanvas');
    const leftStickCtx = leftStickCanvas.getContext('2d');
    const rightStickCanvas = document.getElementById('rightStickCanvas');
    const rightStickCtx = rightStickCanvas.getContext('2d');

    // COM Port Control elements
    const comPortSelect = document.getElementById('comPortSelect');
    const connectComPortBtn = document.getElementById('connectComPortBtn');
    const comPortStatus = document.getElementById('comPortStatus');

    // Camera URL Control elements
    const cameraUrlInput = document.getElementById('cameraUrlInput');
    const setCameraUrlBtn = document.getElementById('setCameraUrlBtn');
    const droneCameraStream = document.getElementById('droneCameraStream');

    // --- Initialize Camera URL from Local Storage ---
    const savedCameraUrl = localStorage.getItem('droneCameraUrl');
    if (savedCameraUrl) {
        cameraUrlInput.value = savedCameraUrl;
        droneCameraStream.src = savedCameraUrl;
    }

    // --- Socket.IO Event Listener ---
    socket.on('connect', () => {
        console.log('Connected to Socket.IO server.');
        socket.emit('list_com_ports'); // Request COM port list on connect
    });

    socket.on('telemetry_update', (data) => {
        // Update numerical displays
        rollSpan.textContent = data.roll.toFixed(1);
        pitchSpan.textContent = data.pitch.toFixed(1);
        yawSpan.textContent = data.yaw.toFixed(1);
        altitudeSpan.textContent = data.altitude.toFixed(1);

        // Update AUX switches
        updateAuxSwitches([data.aux1, data.aux2, data.aux3, data.aux4]);

        // Update canvas instruments
        drawADI(adiCtx, adiCanvas.width, adiCanvas.height, data.roll, data.pitch);
        
        // Normalize stick values (assuming raw values are similar to original app.py)
        // These ranges might need tuning based on actual input
        const rudderNorm = normalizeSymmetrical(data.rudder, 830, 1148, 1500);
        const elevatorNorm = normalizeSymmetrical(data.elevator, 800, 966, 1070);
        const aileronNorm = normalizeSymmetrical(data.aileron, 560, 1164, 1750);
        const throttleNorm = normalizeValue(data.throttle, 360, 1590);

        drawStick(leftStickCtx, leftStickCanvas.width, leftStickCanvas.height, -rudderNorm, -elevatorNorm, "ラダー", "エレベーター");
        drawStick(rightStickCtx, rightStickCanvas.width, rightStickCanvas.height, aileronNorm, throttleNorm, "エルロン", "スロットル");
    });

    socket.on('com_ports_list', (ports) => {
        console.log('Received COM ports list:', ports);
        comPortSelect.innerHTML = ''; // Clear existing options
        if (ports.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No COM ports found';
            comPortSelect.appendChild(option);
            connectComPortBtn.disabled = true;
        } else {
            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = 'Select a port';
            comPortSelect.appendChild(defaultOption);
            ports.forEach(port => {
                const option = document.createElement('option');
                option.value = port.device;
                option.textContent = `${port.device} (${port.description})`;
                comPortSelect.appendChild(option);
            });
            connectComPortBtn.disabled = false;
        }
    });

    socket.on('com_port_status', (status) => {
        console.log('COM Port Status:', status);
        comPortStatus.textContent = status.message;
        if (status.success) {
            comPortStatus.style.color = 'green';
        } else {
            comPortStatus.style.color = 'red';
        }
    });

    // --- Event Listeners for COM Port Control ---
    connectComPortBtn.addEventListener('click', () => {
        const selectedPort = comPortSelect.value;
        if (selectedPort) {
            comPortStatus.textContent = 'Connecting...';
            comPortStatus.style.color = 'orange';
            socket.emit('connect_com_port', { port: selectedPort });
        } else {
            comPortStatus.textContent = 'Please select a COM port.';
            comPortStatus.style.color = 'red';
        }
    });

    // --- Event Listener for Camera URL Control ---
    setCameraUrlBtn.addEventListener('click', () => {
        const newUrl = cameraUrlInput.value;
        if (newUrl) {
            droneCameraStream.src = newUrl;
            localStorage.setItem('droneCameraUrl', newUrl); // Save to local storage
            console.log('Camera URL set to:', newUrl);
        } else {
            droneCameraStream.src = ''; // Clear the image if URL is empty
            localStorage.removeItem('droneCameraUrl');
            console.log('Camera URL cleared.');
        }
    });

    // --- Helper Functions (from original app.py, adapted for JS) ---
    function normalizeValue(value, min_in, max_in, min_out = -1.0, max_out = 1.0) {
        if (max_in === min_in) return 0;
        value = Math.max(min_in, Math.min(value, max_in));
        return min_out + (value - min_in) * (max_out - min_in) / (max_in - min_in);
    }

    function normalizeSymmetrical(value, min_in, center_in, max_in) {
        value = Math.max(min_in, Math.min(value, max_in));
        if (value >= center_in) {
            const span = max_in - center_in;
            return span > 0 ? (value - center_in) / span : 1.0;
        } else {
            const span = center_in - min_in;
            return span > 0 ? (value - center_in) / span : -1.0;
        }
    }

    function updateAuxSwitches(auxValues) {
        const onStyle = "background-color: #28a745; color: white;";
        const offStyle = "background-color: #555; color: white;";
        auxValues.forEach((value, i) => {
            const label = auxLabels[i];
            if (value > 1100) { // Threshold from original app.py
                label.textContent = `AUX${i + 1}: ON`;
                label.style = onStyle;
            } else {
                label.textContent = `AUX${i + 1}: OFF`;
                label.style = offStyle;
            }
        });
    }

    // --- Canvas Drawing Functions (adapted from original app.py) ---

    function drawADI(ctx, width, height, roll, pitch) {
        ctx.clearRect(0, 0, width, height);
        ctx.save();
        ctx.translate(width / 2, height / 2);
        const size = Math.min(width, height);
        ctx.scale(size / 200.0, size / 200.0); // Scale to original 200x200 logic

        // Background
        ctx.fillStyle = "#2b2b2b";
        ctx.fillRect(-100, -100, 200, 200);

        ctx.save();
        const pitchOffset = pitch * 2; // Scale pitch similar to original
        ctx.translate(0, pitchOffset);
        ctx.rotate(-roll * Math.PI / 180); // Convert degrees to radians

        // Sky and Ground
        ctx.fillStyle = "#3282F6"; // Sky blue
        ctx.fillRect(-300, -300, 600, 300);
        ctx.fillStyle = "#8B4513"; // Ground brown
        ctx.fillRect(-300, 0, 600, 300);

        // Horizon line
        ctx.strokeStyle = "white";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(-300, 0);
        ctx.lineTo(300, 0);
        ctx.stroke();
        ctx.restore();

        // Aircraft symbol (yellow)
        ctx.strokeStyle = "yellow";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(-50, 0);
        ctx.lineTo(-10, 0);
        ctx.moveTo(10, 0);
        ctx.lineTo(50, 0);
        ctx.moveTo(0, -5);
        ctx.lineTo(0, 5);
        ctx.stroke();
        ctx.restore();
    }

    function drawStick(ctx, width, height, x, y, xLabel, yLabel) {
        ctx.clearRect(0, 0, width, height);
        ctx.save();

        // Background
        ctx.fillStyle = "#2b2b2b";
        ctx.strokeStyle = "gray";
        ctx.lineWidth = 1;
        ctx.fillRect(0, 0, width, height);
        ctx.strokeRect(0, 0, width - 1, height - 1);

        // Center lines (dashed)
        ctx.strokeStyle = "gray";
        ctx.setLineDash([5, 5]); // Dashed line
        ctx.beginPath();
        ctx.moveTo(width / 2, 5);
        ctx.lineTo(width / 2, height - 5);
        ctx.moveTo(5, height / 2);
        ctx.lineTo(width - 5, height / 2);
        ctx.stroke();
        ctx.setLineDash([]); // Reset to solid line

        // Labels
        ctx.fillStyle = "white";
        ctx.font = "12px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(yLabel, width / 2, 5);
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(xLabel, width - 5, height / 2);

        // Stick position (cyan dot)
        const centerX = width / 2 + x * (width / 2 - 10);
        const centerY = height / 2 - y * (height / 2 - 10); // Y-axis inverted for display

        ctx.fillStyle = "cyan";
        ctx.strokeStyle = "white";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(centerX, centerY, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
    }

    // Initial draw to show empty instruments
    drawADI(adiCtx, adiCanvas.width, adiCanvas.height, 0, 0);
    drawStick(leftStickCtx, leftStickCanvas.width, leftStickCanvas.height, 0, 0, "ラダー", "エレベーター");
    drawStick(rightStickCtx, rightStickCanvas.width, rightStickCanvas.height, 0, 0, "エルロン", "スロットル");
});