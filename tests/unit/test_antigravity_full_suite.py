"""Unit and Integration Tests for Antigravity App, Extensions, CLI, IDE, Plugins, Runtimes, and Setup."""
from src.antigravity_sdk import (
    AntigravityApp,
    AntigravityExtension,
    AntigravityPlugin,
    AppConfig,
    ExtensionHost,
    ExtensionManifest,
    PluginRegistry,
)
from src.antigravity_sdk.cli import main as cli_main
from scripts.setup_antigravity import run_setup


def test_antigravity_plugin_lifecycle():
    registry = PluginRegistry()

    class SamplePlugin(AntigravityPlugin):
        def on_load(self):
            self.register_tool("echo_tool", lambda text: f"echo: {text}")

    p = SamplePlugin("sample_plugin", version="2.0.0", description="Echoes text")
    registry.register(p)

    plugins = registry.list_plugins()
    assert len(plugins) == 1
    assert plugins[0]["name"] == "sample_plugin"

    tools = registry.get_all_tools()
    assert "echo_tool" in tools
    assert tools["echo_tool"]("hello") == "echo: hello"


def test_antigravity_extension_lifecycle():
    host = ExtensionHost()
    manifest = ExtensionManifest(
        id="ext.syntax",
        name="Syntax Checker",
        description="Lints syntax",
    )
    ext = AntigravityExtension(manifest)
    ext.register_command("check_syntax", lambda code: len(code) > 0)
    host.install(ext)

    assert host.get_extension("ext.syntax") is not None
    res = host.dispatch_command("check_syntax", "print('hello')")
    assert res is True


def test_antigravity_app_orchestration():
    app = AntigravityApp(AppConfig(runtime_mode="offline"))
    status = app.get_status()
    assert status["app"] == "Antigravity App & Workbench"
    assert status["runtime_mode"] == "offline"

    exec_res = app.execute_prompt("Analyze portfolio risk under stress")
    assert exec_res["status"] == "success"
    assert "deterministic" in exec_res["response"].lower() or "sandboxed" in exec_res["response"].lower()


def test_antigravity_cli_commands(capsys):
    ret = cli_main(["status"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Antigravity App & Workbench" in captured.out

    ret = cli_main(["plan", "Audit balance sheet"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Layer 0 x Layer 1" in captured.out


def test_antigravity_ide_endpoints():
    from src.antigravity_sdk.ide.server import (
        api_status,
        api_run,
        PromptRequest,
        get_ide_html,
    )

    status = api_status()
    assert status["app"] == "Antigravity App & Workbench"

    run_res = api_run(PromptRequest(prompt="Test run", runtime="offline"))
    assert run_res["status"] == "success"

    html = get_ide_html()
    assert "Antigravity Studio" in html


def test_antigravity_setup_runner():
    code = run_setup(check_only=True)
    assert code == 0
