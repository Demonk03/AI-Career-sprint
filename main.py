from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from prompts import build_prompt, build_prompt_session2, build_prompt_session3
import os
import httpx

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


@app.get("/session3", response_class=HTMLResponse)
async def session3(request: Request):
    return templates.TemplateResponse("session3.html", {"request": request})


@app.get("/result3", response_class=HTMLResponse)
async def result3(request: Request):
    return templates.TemplateResponse("result3.html", {"request": request})


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


class Session3Data(BaseModel):
    hours_per_week: str
    constraints: str = ""
    session1: dict
    session2_tasks: list


@app.post("/analyze3")
async def analyze3(data: Session3Data):
    prompt = build_prompt_session3(
        data.hours_per_week,
        data.constraints,
        data.session1,
        data.session2_tasks
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    report = response.choices[0].message.content
    return {"report": report}


class FeedbackData(BaseModel):
    session: int
    rating: int
    comment: str = ""


@app.post("/feedback")
async def feedback(data: FeedbackData):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    stars = "★" * data.rating + "☆" * (5 - data.rating)
    text = f"Оценка сессии {data.session}: {stars} ({data.rating}/5)"
    if data.comment:
        text += f"\n\n{data.comment}"
    async with httpx.AsyncClient() as client_http:
        await client_http.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text}
        )
    return {"ok": True}


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
