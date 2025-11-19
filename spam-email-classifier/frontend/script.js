const emailText = document.getElementById('emailText');
const classifyBtn = document.getElementById('classifyBtn');
const clearBtn = document.getElementById('clearBtn');
const resultBox = document.getElementById('result');
const errorBox = document.getElementById('error');

function showResult(text, isSpam) {
  resultBox.classList.remove('hidden', 'ok', 'bad');
  resultBox.classList.add(isSpam ? 'bad' : 'ok');
  resultBox.textContent = `Prediction: ${isSpam ? 'SPAM' : 'HAM'}`;
}

function showError(msg) {
  errorBox.classList.remove('hidden');
  errorBox.textContent = msg;
}

function clearMessages() {
  resultBox.classList.add('hidden');
  errorBox.classList.add('hidden');
}

classifyBtn.addEventListener('click', async () => {
  clearMessages();
  const text = (emailText.value || '').trim();
  if (!text) {
    showError('Please enter some text to classify.');
    return;
  }
  classifyBtn.disabled = true;
  classifyBtn.textContent = 'Classifying...';
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    if (!data.ok) {
      showError(data.error || 'Prediction failed.');
    } else {
      showResult(text, data.prediction.toLowerCase() === 'spam');
    }
  } catch (e) {
    showError('Could not connect to backend. Make sure app.py is running.');
  } finally {
    classifyBtn.disabled = false;
    classifyBtn.textContent = 'Classify';
  }
});

clearBtn.addEventListener('click', () => {
  emailText.value = '';
  clearMessages();
});
