import os
import subprocess
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from http.client import HTTPException
from pydantic import BaseModel
from typing import Any

from starlette.middleware.cors import CORSMiddleware

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


class Article(BaseModel):
    data: Any
    desired_position: str


@app.post("/transform_article")
def echo_json(article: Article):
    if article.desired_position not in ["left", "right", "center"]:
        raise HTTPException(status_code=400, detail="Invalid position value. Must be 'left', 'right', or 'center'.")
    return "Qui Vis Pacem Para Bellum !"


def get_article_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # The Guardian uses <div> with class 'article-body-commercial-selector' for the main content
    article_body = soup.find('div', {'class': 'article-body-commercial-selector'})
    if not article_body:
        return "Could not find the article body."

    # Extract text content from paragraphs
    paragraphs = article_body.find_all('p')
    article_text = "\n\n".join([p.get_text() for p in paragraphs])

    return article_text


@app.post('/process_html')
async def process_html(request: Request):
    data = await request.json()
    html_content = data.get('html', '')
    print(html_content[:200])  # Print to console for debugging (truncated for readability)
    article_text = get_article_text(html_content)
    return JSONResponse(content={"processed_content": article_text})

def install_large_dependency():
    # Install the large dependency at runtime
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", "llama-index-embeddings-huggingface"])


if __name__ == "__main__":
    # install_large_dependency()
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
