import os
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


@dataclass
class CanvasNode:
    """Represents a branch on the canvas."""

    id: str
    parent_id: Optional[str]
    title: str
    messages: List[BaseMessage] = field(default_factory=list)
    inherited_count: int = 0


@dataclass
class CanvasConfig:
    """Configuration needed to create a chat model."""

    api_key: str
    base_url: str
    model: str
    system_prompt: str = "You are a helpful canvas chat assistant."


def build_llm(config: CanvasConfig) -> ChatOpenAI:
    return ChatOpenAI(api_key=config.api_key, base_url=config.base_url, model=config.model)


def load_config_from_env() -> CanvasConfig:
    load_dotenv()
    api_key = os.getenv("CANVAS_API_KEY")
    base_url = os.getenv("CANVAS_BASE_URL")
    model = os.getenv("CANVAS_MODEL")
    if not all([api_key, base_url, model]):
        raise ValueError("Please set CANVAS_API_KEY, CANVAS_BASE_URL, and CANVAS_MODEL in your environment.")
    system_prompt = os.getenv("CANVAS_SYSTEM_PROMPT", "You are a helpful canvas chat assistant.")
    return CanvasConfig(api_key=api_key, base_url=base_url, model=model, system_prompt=system_prompt)


class ChatCanvas:
    """
    Canvas-style chat manager.

    Each node carries its own message history. Creating a branch copies the
    ancestry context so conversations can diverge without losing memory.
    """

    DEFAULT_TITLE = "多分支对话画布"

    def __init__(self, llm: BaseChatModel, system_prompt: str):
        self.llm = llm
        self.system_prompt = system_prompt
        # Layout info keyed by node id: x, y, width, height, user_sized
        self.layout: Dict[str, Dict[str, float | bool]] = {}
        self.nodes: Dict[str, CanvasNode] = {}
        self.title = self.DEFAULT_TITLE
        self._init_canvas()

    def _init_canvas(self, title: Optional[str] = None) -> None:
        root = CanvasNode(
            id="root",
            parent_id=None,
            title="Root",
            messages=[SystemMessage(content=self.system_prompt)],
            inherited_count=0,
        )
        self.nodes = {"root": root}
        self.layout = {}
        self.title = title or self.DEFAULT_TITLE

    def reset(self, title: Optional[str] = None) -> CanvasNode:
        """Clear all nodes/layout and start a fresh canvas."""
        self._init_canvas(title=title)
        return self.nodes["root"]

    @staticmethod
    def _visible_message_count(node: CanvasNode) -> int:
        """Count messages that are visible to the user (skip system)."""
        return len([m for m in node.messages if not isinstance(m, SystemMessage)])

    def _require_node(self, node_id: str) -> CanvasNode:
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} does not exist.")
        return self.nodes[node_id]

    def branch_from(self, node_id: str, title: Optional[str] = None) -> CanvasNode:
        parent = self._require_node(node_id)
        branch_id = str(uuid.uuid4())
        branch_title = title or f"Branch {branch_id[:4]}"
        # Copy messages so the branch inherits memory up to this point.
        new_messages = list(parent.messages)
        node = CanvasNode(
            id=branch_id,
            parent_id=parent.id,
            title=branch_title,
            messages=new_messages,
            inherited_count=self._visible_message_count(parent),
        )
        self.nodes[branch_id] = node
        return node

    def rename_node(self, node_id: str, title: str) -> None:
        node = self._require_node(node_id)
        node.title = title

    def rename_canvas(self, title: str) -> None:
        self.title = title

    def send(self, node_id: str, user_message: str) -> AIMessage:
        node = self._require_node(node_id)
        node.messages.append(HumanMessage(content=user_message))
        ai_message = self.llm.invoke(node.messages)
        node.messages.append(ai_message)
        return ai_message

    def history(self, node_id: str) -> List[BaseMessage]:
        return list(self._require_node(node_id).messages)

    def ancestry(self, node_id: str) -> List[CanvasNode]:
        trail = []
        current = self._require_node(node_id)
        while current:
            trail.append(current)
            current = self.nodes.get(current.parent_id) if current.parent_id else None
        trail.reverse()
        return trail

    def delete_subtree(self, node_id: str) -> None:
        """
        Delete a node and all its descendants. Root cannot be deleted.
        """
        if node_id == "root":
            raise ValueError("Cannot delete root node.")

        # Collect descendants via DFS
        to_delete = []

        def collect(nid: str) -> None:
            to_delete.append(nid)
            for child in list(self.nodes.values()):
                if child.parent_id == nid:
                    collect(child.id)

        self._require_node(node_id)
        collect(node_id)
        for nid in to_delete:
            self.nodes.pop(nid, None)
            self.layout.pop(nid, None)

    def describe(self) -> str:
        """Return a simple ASCII snapshot of the canvas."""
        children: Dict[str, List[CanvasNode]] = {}
        for node in self.nodes.values():
            children.setdefault(node.parent_id or "", []).append(node)
        for siblings in children.values():
            siblings.sort(key=lambda n: n.title)

        lines: List[str] = []

        def walk(node: CanvasNode, prefix: str = "") -> None:
            label = f"{node.title} ({node.id})"
            summary = self._last_message_preview(node)
            lines.append(f"{prefix}{label} - {summary}")
            for index, child in enumerate(children.get(node.id, [])):
                connector = "└─ " if index == len(children[node.id]) - 1 else "├─ "
                walk(child, prefix + connector)

        walk(self.nodes["root"])
        return "\n".join(lines)

    @staticmethod
    def _last_message_preview(node: CanvasNode) -> str:
        for message in reversed(node.messages):
            if isinstance(message, AIMessage):
                return f"AI: {message.content[:40]}..."
            if isinstance(message, HumanMessage):
                return f"User: {message.content[:40]}..."
        return "Empty"

    def update_layout(self, layouts: List[Dict[str, object]]) -> None:
        for item in layouts:
            node_id = str(item.get("node_id"))
            if node_id not in self.nodes:
                continue
            try:
                x = float(item.get("x", 0))
                y = float(item.get("y", 0))
                width = float(item.get("width", 240))
                height = float(item.get("height", 160))
                scroll_top_raw = item.get("scroll_top", 0)
                scroll_top = float(scroll_top_raw) if scroll_top_raw is not None else 0.0
            except (TypeError, ValueError):
                continue
            collapsed_raw = item.get("collapsed", {})
            collapsed: Dict[str, bool] = {}
            if isinstance(collapsed_raw, dict):
                for key, value in collapsed_raw.items():
                    try:
                        idx = str(int(key))
                    except (TypeError, ValueError):
                        idx = str(key)
                    collapsed[idx] = bool(value)
            self.layout[node_id] = {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "user_sized": bool(item.get("user_sized", False)),
                "scroll_top": max(scroll_top, 0.0),
                "collapsed": collapsed,
            }

    def export_state(self) -> Dict[str, object]:
        return {
            "version": 1,
            "title": self.title,
            "nodes": self._export_nodes(),
            "layout": self.layout,
        }

    def _export_nodes(self) -> List[Dict[str, object]]:
        payload: List[Dict[str, object]] = []
        for node in self.nodes.values():
            messages = []
            for m in node.messages:
                role = "system"
                if isinstance(m, HumanMessage):
                    role = "user"
                elif isinstance(m, AIMessage):
                    role = "assistant"
                messages.append({"role": role, "content": m.content})
            payload.append(
                {
                    "id": node.id,
                    "parent_id": node.parent_id,
                    "title": node.title,
                    "messages": messages,
                    "inherited_count": node.inherited_count,
                }
            )
        return payload

    def import_state(self, data: Dict[str, object]) -> None:
        version = data.get("version", 1)
        try:
            version_num = int(version)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid version") from exc
        if version_num < 1:
            raise ValueError(f"Unsupported version: {version}")
        nodes_payload = data.get("nodes", [])
        if not isinstance(nodes_payload, list):
            raise ValueError("nodes must be a list")

        new_nodes: Dict[str, CanvasNode] = {}
        for raw in nodes_payload:
            try:
                node_id = str(raw["id"])
                parent_id = raw.get("parent_id")
                title = raw.get("title", "") or ""
                inherited = int(raw.get("inherited_count", 0))
                messages_data = raw.get("messages", [])
            except Exception as exc:
                raise ValueError(f"Invalid node payload: {raw}") from exc

            messages: List[BaseMessage] = []
            for msg in messages_data:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "assistant":
                    messages.append(AIMessage(content=content))
                elif role == "system":
                    messages.append(SystemMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))

            new_nodes[node_id] = CanvasNode(
                id=node_id,
                parent_id=parent_id,
                title=title,
                messages=messages,
                inherited_count=min(inherited, len([m for m in messages if not isinstance(m, SystemMessage)])),
            )

        if "root" not in new_nodes:
            raise ValueError("Import data must contain a root node.")

        self.nodes = new_nodes
        layout_payload = data.get("layout", {})
        if isinstance(layout_payload, dict):
            filtered_layout = {k: v for k, v in layout_payload.items() if k in self.nodes}
            self.layout = filtered_layout
        else:
            self.layout = {}
        title = data.get("title")
        if isinstance(title, str) and title.strip():
            self.title = title.strip()
