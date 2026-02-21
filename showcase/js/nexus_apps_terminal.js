// Nexus Apps: Terminal
// Implements an interactive Command Line Interface

(function() {
    function initTerminalApp() {
        if (!window.officeOS) {
            setTimeout(initTerminalApp, 500);
            return;
        }

        patchAppRegistry();
        console.log("Nexus Terminal: Loaded.");
    }

    function patchAppRegistry() {
        const originalLaunch = window.officeOS.appRegistry.launch.bind(window.officeOS.appRegistry);

        window.officeOS.appRegistry.launch = function(appName, args) {
            if (appName === 'Terminal') {
                launchInteractiveTerminal(args);
            } else {
                originalLaunch(appName, args);
            }
        };
    }

    function launchInteractiveTerminal(args) {
        const winId = window.officeOS.windowManager.createWindow({
            title: 'Nexus Terminal',
            icon: 'https://img.icons8.com/color/48/000000/console.png',
            width: 700,
            height: 450,
            app: 'Terminal'
        });

        const container = document.createElement('div');
        container.style.backgroundColor = 'black';
        container.style.color = '#0f0'; // Matrix green
        container.style.fontFamily = 'monospace';
        container.style.padding = '10px';
        container.style.height = '100%';
        container.style.overflowY = 'auto';
        container.style.fontSize = '14px';
        container.id = `terminal-container-${winId}`;

        // Intro Text
        const intro = `Nexus OS [Version 26.0.1]<br>(c) Adam Corp. All rights reserved.<br><br>Type 'help' for commands.<br><br>`;
        container.innerHTML = intro;

        // Command Prompt Logic
        let currentPath = "C:\\Users\\Admin>";
        let history = [];
        let historyIndex = -1;

        const createPrompt = () => {
            const line = document.createElement('div');
            line.style.display = 'flex';
            line.innerHTML = `
                <span style="margin-right:8px; white-space:nowrap;">${currentPath}</span>
                <input type="text" style="flex:1; background:transparent; border:none; color:#0f0; outline:none; font-family:monospace; font-size:14px;" autofocus>
            `;
            container.appendChild(line);

            const input = line.querySelector('input');
            input.focus();

            // Keep focus
            container.addEventListener('click', () => input.focus());

            input.addEventListener('keydown', async (e) => {
                if (e.key === 'Enter') {
                    const cmd = input.value.trim();
                    input.disabled = true; // Lock previous input

                    if (cmd) {
                        history.push(cmd);
                        historyIndex = history.length;
                        await processCommand(cmd, container);
                    } else {
                         // Empty line
                         createPrompt();
                    }

                    // Scroll to bottom
                    container.scrollTop = container.scrollHeight;
                } else if (e.key === 'ArrowUp') {
                    if (historyIndex > 0) {
                        historyIndex--;
                        input.value = history[historyIndex];
                    }
                    e.preventDefault();
                } else if (e.key === 'ArrowDown') {
                    if (historyIndex < history.length - 1) {
                        historyIndex++;
                        input.value = history[historyIndex];
                    } else {
                        historyIndex = history.length;
                        input.value = '';
                    }
                    e.preventDefault();
                }
            });
        };

        const processCommand = async (cmd, outputDiv) => {
            const parts = cmd.split(' ');
            const command = parts[0].toLowerCase();
            const args = parts.slice(1);

            const print = (text, color='#0f0') => {
                const div = document.createElement('div');
                div.style.color = color;
                div.style.whiteSpace = 'pre-wrap';
                div.innerHTML = text; // Allow HTML
                outputDiv.appendChild(div);
            };

            switch (command) {
                case 'help':
                    print(`Available commands:
  help             Show this help message
  cls / clear      Clear the screen
  ls / dir         List files in current directory
  cat / type [file] Display file contents
  price [ticker]   Show current price for a ticker
  peers [ticker]   Find peer set (Simulated Agent)
  analyze [ticker] Run simulated analysis agent
  date             Show current date/time
  exit             Close terminal`);
                    break;

                case 'cls':
                case 'clear':
                    outputDiv.innerHTML = '';
                    break;

                case 'ls':
                case 'dir':
                    // Mock file listing from window.officeOS.fs
                    // Assuming current path is root for simplicity in this mock
                    const files = window.officeOS.fs.readDir('./');
                    if(files.length > 0) {
                        let out = '';
                        files.forEach(f => {
                             const type = f.type === 'directory' ? '<DIR>' : '     ';
                             out += `${f.modified.substring(0,10)}  ${type}  ${f.name}\n`;
                        });
                        print(out);
                    } else {
                        print("Directory is empty.");
                    }
                    break;

                case 'cat':
                case 'type':
                    if(args.length === 0) {
                        print("Usage: cat [filename]", "red");
                    } else {
                        const filename = args[0];
                        // Try to find file in root
                        const rootFiles = window.officeOS.fs.readDir('./');
                        const file = rootFiles.find(f => f.name === filename);

                        if(file && file.type === 'file') {
                             try {
                                 const res = await fetch(file.path);
                                 if(res.ok) {
                                     const text = await res.text();
                                     // Escape HTML to prevent execution
                                     const safeText = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                                     print(safeText.substring(0, 1000) + (safeText.length > 1000 ? "\n... (truncated)" : ""));
                                 } else {
                                     print(`Error reading file: ${res.statusText}`, "red");
                                 }
                             } catch(e) {
                                 print(`Error: ${e}`, "red");
                             }
                        } else {
                            print("File not found.", "red");
                        }
                    }
                    break;

                case 'price':
                    if(args.length === 0) {
                        print("Usage: price [ticker]", "red");
                    } else {
                        const ticker = args[0].toUpperCase();
                        if(window.MARKET_DATA) {
                            const data = window.MARKET_DATA.find(d => d.ticker === ticker);
                            if(data) {
                                print(`Symbol: ${ticker}`);
                                print(`Price:  $${data.current_price}`);
                                print(`Change: ${data.change_pct}%`, data.change_pct >= 0 ? '#0f0' : 'red');
                                print(`P/E:    ${data.pe_ratio}`);
                            } else {
                                print(`Ticker ${ticker} not found in market data.`, "red");
                            }
                        } else {
                            print("Market data not loaded.", "red");
                        }
                    }
                    break;

                case 'peers':
                    if(args.length === 0) {
                        print("Usage: peers [ticker]", "red");
                    } else {
                        const ticker = args[0].toUpperCase();
                        print(`[Agent:PeerSet] Identifying peer set for ${ticker}...`);
                        await new Promise(r => setTimeout(r, 800));

                        // Mock Logic matching Python Agent (Simulating Semantic Analysis)
                        const mockPeers = {
                            "AAPL": [
                                { t: "DELL", s: 0.85, i: "Hardware" },
                                { t: "HPQ", s: 0.82, i: "Hardware" },
                                { t: "MSFT", s: 0.65, i: "Software" }
                            ],
                            "MSFT": [
                                { t: "ORCL", s: 0.78, i: "Software" },
                                { t: "ADBE", s: 0.75, i: "Software" },
                                { t: "AMZN", s: 0.60, i: "Cloud" }
                            ],
                            "GOOGL": [
                                { t: "META", s: 0.88, i: "Interactive Media" },
                                { t: "AMZN", s: 0.70, i: "Cloud/Ads" },
                                { t: "MSFT", s: 0.65, i: "Cloud" }
                            ],
                            "AMZN": [
                                { t: "BABA", s: 0.85, i: "Retail" },
                                { t: "WMT", s: 0.70, i: "Retail" },
                                { t: "MSFT", s: 0.60, i: "Cloud" }
                            ],
                            "TSLA": [
                                { t: "BYD", s: 0.88, i: "Auto" },
                                { t: "F", s: 0.75, i: "Auto" },
                                { t: "GM", s: 0.72, i: "Auto" }
                            ],
                            "NVDA": [
                                { t: "AMD", s: 0.92, i: "Semis" },
                                { t: "INTC", s: 0.80, i: "Semis" },
                                { t: "TSM", s: 0.70, i: "Foundry" }
                            ],
                            "JPM": [
                                { t: "BAC", s: 0.95, i: "Banks" },
                                { t: "C", s: 0.90, i: "Banks" },
                                { t: "WFC", s: 0.88, i: "Banks" }
                            ],
                            "V": [
                                { t: "MA", s: 0.98, i: "Payments" },
                                { t: "AXP", s: 0.85, i: "Payments" },
                                { t: "PYPL", s: 0.70, i: "Fintech" }
                            ],
                            "PG": [
                                { t: "CL", s: 0.90, i: "Household" },
                                { t: "KO", s: 0.60, i: "Beverages" },
                                { t: "PEP", s: 0.55, i: "Beverages" }
                            ]
                        };

                        const peers = mockPeers[ticker];

                        if(peers) {
                            print(`Peer Set Analysis for ${ticker}`);
                            print(`Method: Semantic Similarity (TF-IDF on Business Summary)`);
                            print(`------------------------------------------------------`);
                            peers.forEach(p => {
                                print(`- ${p.t.padEnd(6)} | Score: ${p.s.toFixed(2)} | ${p.i}`);
                            });
                            print(`\nTotal Peers: ${peers.length}`);
                            print(`(Note: Scores calculated via simulated sklearn cosine similarity)`, "#888");
                        } else {
                            print(`No peers found for ${ticker} in local cache.`, "red");
                        }
                    }
                    break;

                case 'analyze':
                    if(args.length === 0) {
                        print("Usage: analyze [ticker]", "red");
                    } else {
                        const ticker = args[0].toUpperCase();
                        print(`Initiating Deep Dive Analysis for ${ticker}...`);
                        await new Promise(r => setTimeout(r, 1000));
                        print(`[Agent:MarketSentiment] Scanning news feed... OK`);
                        await new Promise(r => setTimeout(r, 800));
                        print(`[Agent:Technical] Calculating RSI and MACD... OK`);
                        await new Promise(r => setTimeout(r, 800));
                        print(`[Agent:Fundamental] Checking 10-K... OK`);
                        await new Promise(r => setTimeout(r, 1000));

                        // Check if report exists
                        const reportPath = `data/equity_reports/${ticker}_Equity_Report.html`;
                        print(`Analysis Complete. Opening report...`);

                        // Launch browser
                        setTimeout(() => {
                            window.officeOS.appRegistry.launch('Browser', { url: reportPath, name: `${ticker} Report` });
                        }, 500);
                    }
                    break;

                case 'date':
                    print(new Date().toString());
                    break;

                case 'exit':
                    window.officeOS.windowManager.closeWindow(outputDiv.id.replace('terminal-container-', ''));
                    return; // Don't create prompt

                default:
                    print(`'${command}' is not recognized as an internal or external command.`, "red");
            }

            createPrompt();
        };

        window.officeOS.windowManager.setWindowContent(winId, container);
        createPrompt();
    }

    initTerminalApp();
})();
