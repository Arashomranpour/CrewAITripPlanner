import json
import requests
import os
from dotenv import load_dotenv
import streamlit as st
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from crewai import Agent, Task
from langchain_groq import ChatGroq
from crewai import LLM

# Load environment variables
load_dotenv()


class WebsiteInput(BaseModel):
    website: str = Field(..., description="The website URL to scrape")


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    text = soup.get_text(separator="\n")
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])


class BrowserTools(BaseTool):
    name: str = "Scrape website content"
    description: str = "Useful to scrape and summarize a website content"
    args_schema: type[BaseModel] = WebsiteInput

    def _run(self, website: str) -> str:
        try:
            api_key = os.getenv("BROWSERLESS_API_KEY")
            if not api_key:
                return "Error: BROWSERLESS_API_KEY not found in environment variables."

            url = f"https://chrome.browserless.io/content?token={api_key}"
            payload = json.dumps({"url": website})
            headers = {"cache-control": "no-cache", "content-type": "application/json"}
            response = requests.post(url, headers=headers, data=payload)

            if response.status_code != 200:
                return f"Error: Failed to fetch website content. Status code: {response.status_code}"

            cleaned_text = extract_text_from_html(response.text)
            chunks = [
                cleaned_text[i : i + 8000] for i in range(0, len(cleaned_text), 8000)
            ]

            llm = LLM(model="gemini/gemini-2.0-flash")

            summaries = []
            for chunk in chunks:
                agent = Agent(
                    role="Principal Researcher",
                    goal="Do amazing researches and summaries based on the content you are working with",
                    backstory="You're a Principal Researcher at a big company and you need to do a research about a given topic.",
                    allow_delegation=False,
                    llm=llm,
                )
                task = Task(
                    description=f"Analyze and summarize the content below, include only the most relevant information in your summary:\n\nCONTENT:\n{chunk}",
                    agent=agent,
                )
                summary = task.execute()
                summaries.append(summary)

            return "\n\n".join(summaries)

        except Exception as e:
            return f"Error while processing website: {str(e)}"

    async def _arun(self, website: str) -> str:
        raise NotImplementedError("Async not implemented")
