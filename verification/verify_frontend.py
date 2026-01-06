from playwright.sync_api import sync_playwright
import os

def verify_terminal_widget():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        content = """
        <html>
        <head>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-slate-900 text-white p-24">
            <div class="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex mb-10">
                <p class="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit lg:static lg:w-auto  lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30">
                  Adam v24.0&nbsp;
                  <code class="font-mono font-bold">Pragmatic Edition</code>
                </p>
            </div>

            <div class="w-full max-w-4xl grid grid-cols-1 gap-4">
                <div class="p-6 border border-slate-700 rounded-xl bg-slate-800/50">
                  <h2 class="text-xl font-bold mb-4">Market Operations</h2>

                  <!-- Terminal Widget -->
                  <div class="bg-black text-green-500 font-mono p-4 rounded-lg border border-green-800 h-64 overflow-y-auto">
                      <div class="mb-2 border-b border-green-800 pb-2 flex justify-between">
                        <span>ADAM-v24 TERMINAL</span>
                        <span class="animate-pulse">● ONLINE</span>
                      </div>
                      <div class="space-y-1 text-sm">
                        <p>[SYSTEM] Initializing v24 Interface...</p>
                        <p>[SYSTEM] Connected to Rust Core Engine.</p>
                        <p>[SYSTEM] Listening for market data...</p>
                        <p class="text-blue-400">[INFO] Waiting for user input...</p>
                      </div>
                      <div class="mt-4 flex">
                        <span class="mr-2">></span>
                        <input
                          type="text"
                          class="bg-transparent outline-none flex-1 text-green-500 placeholder-green-800"
                          placeholder="Enter command..."
                        />
                      </div>
                    </div>
                    <!-- End Terminal Widget -->

                </div>
            </div>
        </body>
        </html>
        """

        file_path = os.path.abspath("verification/mock_v24.html")
        with open(file_path, "w") as f:
            f.write(content)

        page.goto("file://" + file_path)
        page.screenshot(path="verification/v24_dashboard.png")
        print(f"Screenshot saved to {os.path.abspath('verification/v24_dashboard.png')}")

if __name__ == "__main__":
    verify_terminal_widget()
