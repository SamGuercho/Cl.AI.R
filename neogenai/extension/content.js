(async () => {
    const html = document.documentElement.outerHTML;

    try {
        const response = await fetch('https://neogenai-a7a8938144db.herokuapp.com/process_html', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ html: html })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        chrome.runtime.sendMessage({ type: 'htmlContent', content: result.processed_content });
    } catch (error) {
        console.error('Error:', error);
    }
})();
