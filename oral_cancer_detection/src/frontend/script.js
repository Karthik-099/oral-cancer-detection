document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('file', document.getElementById('fileInput').files[0]);
    const response = await fetch('/predict', { method: 'POST', body: formData });
    const data = await response.json();
    document.getElementById('result').innerHTML = `Result: ${data.result}<br>Confidence: ${data.confidence}`;
});
