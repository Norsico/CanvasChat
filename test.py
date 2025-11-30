from canvas_chat import ChatCanvas, CanvasConfig, build_llm, load_config_from_env


def main() -> None:
    """
    Quick demo: run `python test.py` after setting
    CANVAS_API_KEY, CANVAS_BASE_URL, CANVAS_MODEL in your environment or .env.
    """
    # Prefer explicit config; fall back to environment-based loading.
    try:
        config = load_config_from_env()
    except ValueError:
        config = CanvasConfig(
            api_key="123456",
            base_url="http://127.0.0.1:3003/v1",
            model="gemini-3-pro-preview",
            system_prompt="You are a helpful assistant on a canvas chat.",
        )

    llm = build_llm(config)
    canvas = ChatCanvas(llm=llm, system_prompt=config.system_prompt)

    print("Root chat:")
    root_reply = canvas.send("root", "你好，简单介绍一下LangChain吧？")
    print(root_reply.content)

    branch = canvas.branch_from("root", title="深挖问题分支")
    print("\nBranched chat:")
    branch_reply = canvas.send(branch.id, "给我更详细的LangChain组件介绍。")
    print(branch_reply.content)

    print("\nCanvas snapshot:")
    print(canvas.describe())


if __name__ == "__main__":
    main()

