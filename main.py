from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from prompts import build_prompt, build_prompt_session2
import os

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index_minimal.html", {"request": request})


@app.get("/survey", response_class=HTMLResponse)
async def survey(request: Request):
    return templates.TemplateResponse("survey.html", {"request": request})


@app.get("/result", response_class=HTMLResponse)
async def result(request: Request):
    return templates.TemplateResponse("result.html", {"request": request})


@app.get("/session2", response_class=HTMLResponse)
async def session2(request: Request):
    return templates.TemplateResponse("session2.html", {"request": request})


@app.get("/result2", response_class=HTMLResponse)
async def result2(request: Request):
    return templates.TemplateResponse("result2.html", {"request": request})


class TaskItem(BaseModel):
    task: str
    energy: str
    time_percent: str


class Session2Data(BaseModel):
    tasks: list[TaskItem]
    session1: dict


@app.post("/analyze2")
async def analyze2(data: Session2Data):
    prompt = build_prompt_session2(
        [t.model_dump() for t in data.tasks],
        data.session1
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    report = response.choices[0].message.content
    return {"report": report}


class SurveyData(BaseModel):
    goal: str
    goal_why: str
    strengths: str
    experience: str
    reputation: str
    productive_when: str
    not_working: str
    drains: str
    swot_strong: str
    swot_weak: str
    swot_opportunities: str
    swot_threats: str


@app.post("/analyze")
async def analyze(data: SurveyData):
    prompt = build_prompt(data.model_dump())
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    report = response.choices[0].message.content
    return {"report": report}
