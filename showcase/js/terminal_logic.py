import js
from pyodide.ffi import create_proxy
import sys
import asyncio
import json

# Terminal Interface
class TerminalInterface:
    def __init__(self):
        self.term = js.window.terminal
        self.buffer = ""
        self.history = []
        self.history_index = 0

        # Bind key event
        self.on_key_proxy = create_proxy(self.on_key)
        self.term.onKey(self.on_key_proxy)

        self.prompt_str = "\x1b[1;32madam@core:~$ \x1b[0m"
        self.write(f"\r\nWelcome to the Adam v23.5 Interactive Shell.\r\nType 'help' for available commands.\r\n\r\n{self.prompt_str}")

    def write(self, text):
        self.term.write(text)

    def writeln(self, text):
        self.term.writeln(text)

    def on_key(self, event):
        key = event.key
        dom_event = event.domEvent

        printable = not dom_event.altKey and not dom_event.ctrlKey and not dom_event.metaKey

        if dom_event.keyCode == 13: # Enter
            self.write("\r\n")
            asyncio.ensure_future(self.process_command(self.buffer))
            self.history.append(self.buffer)
            self.history_index = len(self.history)
            self.buffer = ""
        elif dom_event.keyCode == 8: # Backspace
            if len(self.buffer) > 0:
                self.buffer = self.buffer[:-1]
                self.write("\b \b")
        elif dom_event.keyCode == 38: # Up Arrow
            if self.history_index > 0:
                self.history_index -= 1
                self.clear_line()
                self.buffer = self.history[self.history_index]
                self.write(self.buffer)
        elif dom_event.keyCode == 40: # Down Arrow
            if self.history_index < len(self.history):
                self.history_index += 1
                self.clear_line()
                if self.history_index < len(self.history):
                    self.buffer = self.history[self.history_index]
                    self.write(self.buffer)
                else:
                    self.buffer = ""
        elif printable:
            self.buffer += key
            self.write(key)

    def clear_line(self):
        # Move to start of line, clear to end
        # This is a bit hacky, xterm.js has better ways but this works for simple cases
        # We need to calculate prompt length.
        # Simple backspace loop for now
        self.write("\r" + self.prompt_str + " " * len(self.buffer) + "\r" + self.prompt_str)

    async def process_command(self, command):
        command = command.strip()
        if not command:
            self.write(self.prompt_str)
            return

        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:]

        output = ""

        if cmd == "help":
            output = """
Available Commands:
  help                     Show this help message
  clear                    Clear the terminal
  whoami                   Display current user profile
  risk <entity> <amount>   Calculate credit exposure (MCP Tool)
  market <ticker>          Retrieve market data (MCP Tool)
  python <code>            Execute Python code (Sandboxed)
  bio                      Show professional biography
  studies                  Show case studies
  redteam <scenario>       Run Red Team attack simulation
  crisis <shock>           Run Crisis simulation
"""
        elif cmd == "clear":
            self.term.clear()
            self.write(self.prompt_str)
            return
        elif cmd == "whoami":
            output = "Adam v23.5 - Autonomous Financial Analyst"
        elif cmd == "bio":
            output = await self.mcp_call("get_profile_bio", {})
        elif cmd == "studies":
            output = await self.mcp_call("get_case_studies", {})
        elif cmd == "risk":
            if len(args) < 2:
                output = "Usage: risk <entity> <amount>"
            else:
                output = await self.mcp_call("calculate_credit_exposure", {"entity_id": args[0], "amount": float(args[1])})
        elif cmd == "market":
            if len(args) < 1:
                output = "Usage: market <ticker>"
            else:
                output = await self.mcp_call("retrieve_market_data", {"ticker": args[0]})
        elif cmd == "redteam":
            scenario = " ".join(args) if args else "Standard Stress Test"
            output = await self.mcp_call("run_red_team_attack", {"scenario": scenario})
        elif cmd == "crisis":
            shock = " ".join(args) if args else "Global Recession"
            output = await self.mcp_call("run_crisis_simulation", {"macro_shock": shock})
        elif cmd == "python":
            code = " ".join(args)
            output = await self.mcp_call("execute_python_sandbox", {"code": code})
        else:
            output = f"Command not found: {cmd}"

        self.writeln(output)
        self.write(self.prompt_str)

    async def mcp_call(self, tool_name, kwargs):
        """
        Simulates an MCP call.
        In a real scenario, this would fetch from the server.
        Here we implement the logic client-side for the 'Live Profile' experience
        since the server might not be running or accessible from GitHub Pages.
        """
        self.writeln(f"\x1b[33m[MCP] Calling {tool_name}...\x1b[0m")
        await asyncio.sleep(0.5) # Simulate latency

        # Mock Logic mirroring server.py
        if tool_name == "calculate_credit_exposure":
            entity = kwargs.get("entity_id", "UNKNOWN")
            amount = kwargs.get("amount", 0)
            risk = 0.15
            if entity.upper() in ["AMC", "GME"]: risk = 0.45
            elif entity.upper() in ["AAPL", "MSFT"]: risk = 0.05
            return json.dumps({
                "entity": entity,
                "exposure": amount * risk,
                "risk_factor": risk
            }, indent=2)

        elif tool_name == "retrieve_market_data":
            ticker = kwargs.get("ticker", "").upper()
            # Mock Data
            data = {
                "symbol": ticker,
                "currentPrice": 150.25,
                "marketCap": 2500000000000,
                "sector": "Technology"
            }
            return json.dumps(data, indent=2)

        elif tool_name == "get_profile_bio":
            return """
Adam is a high-level financial services professional with a background in credit risk management,
investment banking, and corporate ratings advisory. He has experience at institutions such as
Credit Suisse and S&P.
            """

        elif tool_name == "get_case_studies":
            return json.dumps([
                {"title": "LBO of Tech Conglomerate", "roi": "22%"},
                {"title": "Distressed Debt Restructuring", "recovery": "85 cents on dollar"}
            ], indent=2)

        elif tool_name == "run_red_team_attack":
            return json.dumps({
                "status": "ATTACK_COMPLETE",
                "vulnerabilities": ["Liquidity Crunch", "Correlation Breakdown"],
                "severity": "HIGH"
            }, indent=2)

        elif tool_name == "run_crisis_simulation":
            return json.dumps({
                "status": "SIMULATION_COMPLETE",
                "impact": {"sp500": "-15%", "gdp": "-1.2%"}
            }, indent=2)

        elif tool_name == "execute_python_sandbox":
            code = kwargs.get("code", "")
            try:
                # We can actually run this in PyScript!
                # But we need to capture output
                import io
                from contextlib import redirect_stdout
                f = io.StringIO()
                with redirect_stdout(f):
                    exec(code)
                return f.getvalue() or "[Success (No Output)]"
            except Exception as e:
                return f"Error: {e}"

        return "Tool not implemented in client-side mock."

# Initialize
terminal = TerminalInterface()
