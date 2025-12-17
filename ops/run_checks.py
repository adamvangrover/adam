import asyncio
import sys
import os
import time

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

CHECKS = {
    "syntax": ["python", "ops/checks/check_syntax.py"],
    "lint": ["python", "ops/checks/check_lint.py"],
    "security": ["python", "ops/checks/check_security.py"],
    "types": ["python", "ops/checks/check_types.py"],
    "tests": ["python", "ops/checks/check_tests.py"],
}

async def run_check(name, cmd):
    start_time = time.time()
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    end_time = time.time()
    duration = end_time - start_time

    return {
        "name": name,
        "success": process.returncode == 0,
        "stdout": stdout.decode().strip(),
        "stderr": stderr.decode().strip(),
        "duration": duration
    }

async def main():
    if HAS_RICH:
        console.print("[bold blue]Running all checks...[/bold blue]")
    else:
        print("Running all checks...")

    tasks = [run_check(name, cmd) for name, cmd in CHECKS.items()]

    results = await asyncio.gather(*tasks)

    success_count = 0

    if HAS_RICH:
        table = Table(title="Check Results")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Duration", justify="right")
        table.add_column("Details")

    for res in results:
        status = "[green]PASS[/green]" if res["success"] else "[red]FAIL[/red]"
        if res["success"]:
            success_count += 1

        details = ""
        if not res["success"]:
            details = (res["stdout"] + "\n" + res["stderr"]).replace("\n", " ")

        if HAS_RICH:
            table.add_row(res["name"], status, f"{res['duration']:.2f}s", details[:50] + "..." if details else "")
        else:
            print(f"{res['name']}: {'PASS' if res['success'] else 'FAIL'} ({res['duration']:.2f}s)")

    if HAS_RICH:
        console.print(table)

    # Print failures in detail
    failed = False
    for res in results:
        if not res["success"]:
            failed = True
            print(f"\n{'='*20} {res['name']} FAILED {'='*20}")
            print("STDOUT:", res["stdout"])
            print("STDERR:", res["stderr"])

    if not failed:
        if HAS_RICH:
            console.print("\n[bold green]All checks passed![/bold green]")
        sys.exit(0)
    else:
        if HAS_RICH:
            console.print(f"\n[bold red]{len(CHECKS) - success_count} checks failed.[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
