import pathlib
from threading import Lock
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from canvas_chat import ChatCanvas, CanvasConfig, build_llm, load_config_from_env

# Load configuration for LLM
try:
    CONFIG = load_config_from_env()
except ValueError:
    # Fallback placeholders keep the app bootable for UI demos.
    CONFIG = CanvasConfig(
        api_key="123456",
        base_url="http://127.0.0.1:3003/v1",
        model="gemini-3-pro-preview",
        system_prompt="You are a helpful assistant on a canvas chat.",
    )

llm = build_llm(CONFIG)
canvas = ChatCanvas(llm=llm, system_prompt=CONFIG.system_prompt)
lock = Lock()

BASE_DIR = pathlib.Path(__file__).parent

app = FastAPI(title="CanvasChat")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class BranchRequest(BaseModel):
    from_node: str
    title: str | None = None


class MessageRequest(BaseModel):
    node_id: str
    message: str


class DeleteRequest(BaseModel):
    node_id: str


class LayoutItem(BaseModel):
    node_id: str
    x: float
    y: float
    width: float
    height: float
    user_sized: bool | None = False
    scroll_top: float | None = None
    collapsed: Dict[str, bool] | None = None


class LayoutRequest(BaseModel):
    layouts: List[LayoutItem]


class ImportRequest(BaseModel):
    data: dict


class RenameNodeRequest(BaseModel):
    node_id: str
    title: str


class RenameCanvasRequest(BaseModel):
    title: str


class NewCanvasRequest(BaseModel):
    title: str | None = None


def _serialize_nodes() -> List[Dict]:
    data: List[Dict] = []
    for node in canvas.nodes.values():
        messages = []
        for m in node.messages:
            role = "system"
            if m.__class__.__name__ == "HumanMessage":
                role = "user"
            elif m.__class__.__name__ == "AIMessage":
                role = "assistant"
            messages.append({"role": role, "content": m.content})
        data.append(
            {
                "id": node.id,
                "parent_id": node.parent_id,
                "title": node.title,
                "messages": messages,
                "inherited_count": node.inherited_count,
                "layout": canvas.layout.get(node.id),
            }
        )
    return data


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/nodes")
def list_nodes() -> Dict[str, object]:
    with lock:
        return {"nodes": _serialize_nodes(), "title": canvas.title}


@app.post("/api/branch")
def branch(req: BranchRequest) -> Dict[str, Dict]:
    with lock:
        try:
            node = canvas.branch_from(req.from_node, title=req.title)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"node": {"id": node.id, "title": node.title, "parent_id": node.parent_id}}


@app.post("/api/message")
def send_message(req: MessageRequest) -> Dict[str, object]:
    with lock:
        try:
            ai_message = canvas.send(req.node_id, req.message)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "reply": {"role": "assistant", "content": ai_message.content},
            "nodes": _serialize_nodes(),
        }


@app.post("/api/delete")
def delete_node(req: DeleteRequest) -> Dict[str, object]:
    with lock:
        try:
            canvas.delete_subtree(req.node_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"nodes": _serialize_nodes()}


@app.post("/api/layout")
def save_layout(req: LayoutRequest) -> Dict[str, object]:
    with lock:
        canvas.update_layout([item.model_dump() for item in req.layouts])
        return {"ok": True}


@app.get("/api/export")
def export_canvas() -> Dict[str, object]:
    with lock:
        return canvas.export_state()


@app.post("/api/import")
def import_canvas(req: ImportRequest) -> Dict[str, object]:
    with lock:
        try:
            canvas.import_state(req.data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"nodes": _serialize_nodes(), "title": canvas.title}


@app.post("/api/rename_node")
def rename_node(req: RenameNodeRequest) -> Dict[str, object]:
    with lock:
        try:
            canvas.rename_node(req.node_id, req.title.strip())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"nodes": _serialize_nodes()}


@app.post("/api/rename_canvas")
def rename_canvas(req: RenameCanvasRequest) -> Dict[str, object]:
    with lock:
        title = req.title.strip()
        if not title:
            raise HTTPException(status_code=400, detail="标题不能为空")
        canvas.rename_canvas(title)
        return {"title": canvas.title}


@app.post("/api/new_canvas")
def new_canvas(req: NewCanvasRequest) -> Dict[str, object]:
    with lock:
        title = req.title.strip() if req.title else None
        canvas.reset(title)
        return {"nodes": _serialize_nodes(), "title": canvas.title}
