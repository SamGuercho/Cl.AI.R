document.getElementById('fetch-html').addEventListener('click', () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.scripting.executeScript({
            target: { tabId: tabs[0].id },
            files: ['content.js']
        });
    });
});

chrome.runtime.onMessage.addListener((message) => {
    if (message.type === 'htmlContent') {
        document.getElementById('result').textContent = message.content;
    } else {
        console.error('Unknown message type:', message);
    }
});
