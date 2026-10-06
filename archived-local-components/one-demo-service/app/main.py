from copy import deepcopy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ONE Demo API", version="0.1.0")

class DemoLogin(BaseModel):
    email: str
    password: str

SCENARIOS = {
    "online-camera": {"camera": "online", "presence": "current"},
    "offline-camera": {"camera": "offline", "presence": "unknown"},
    "active-presence": {"camera": "online", "presence": "active"},
    "fallback-presence": {"camera": "online", "presence": "recent-fallback"},
}

_seed = {
    "user": {"id": "demo-user", "email": "demo@demo", "synthetic_demo": True},
    "home": {"id": "demo-home", "name": "Demo Home", "synthetic_demo": True},
    "scenario": "online-camera",
}

state = deepcopy(_seed)

@app.post("/api/demo/auth/login")
def login(body: DemoLogin):
    if body.email.lower() != "demo@demo" or body.password != "demo":
        return {"authenticated": False}
    return {"authenticated": True, "access_token": "demo-only-token", "token_type": "demo", "account": state["user"]}

@app.get("/api/demo/state")
def get_state():
    return {**state, "config": SCENARIOS[state["scenario"]]}

@app.post("/api/demo/scenario/{scenario}")
def set_scenario(scenario: str):
    if scenario not in SCENARIOS:
        raise HTTPException(404, "unknown scenario")
    state["scenario"] = scenario
    return get_state()

@app.post("/api/demo/reset")
def reset():
    state.clear()
    state.update(deepcopy(_seed))
    return get_state()
