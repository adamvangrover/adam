/**
 * PROMPT ALPHA - Client-Side Intelligence Aggregator
 * "The Bloomberg Terminal for Prompts"
 */

const PromptAlpha = {
    // --- State ---
    state: {
        prompts: [],
        portfolio: JSON.parse(localStorage.getItem('pa_portfolio')) || [],
        shorts: JSON.parse(localStorage.getItem('pa_shorts')) || [],
        currentPrompt: null,
        mode: 'LIVE', // or 'MOCK'
        activeTab: 'briefing'
    },

    // --- Configuration ---
    config: {
        redditUrl: 'https://www.reddit.com/r/ChatGPT/top.json?t=day&limit=25',
        scoring: {
            complexityWeight: 0.6,
            viralityWeight: 0.4,
            baseScore: 50
        },
        blacklist: ["nsfw", "xxx", "porn", "nudity", "explicit", "sex", "erotic"]
    },

    // --- Initialization ---
    init: async function() {
        console.log("Initializing Prompt Alpha...");
        
        try {
            await this.fetchData();
        } catch (e) {
            console.warn("Live fetch failed, switching to Mock Data", e);
            this.state.mode = 'MOCK';
            this.loadMockData();
        }

        this.renderTicker();
        this.renderMarketMovers();
        this.switchTab('briefing'); // Initial render of right panel

        // Select first prompt if available
        if (this.state.prompts.length > 0) {
            this.loadPrompt(this.state.prompts[0]);
        }
    },

    updateDate: function() {
        // Returned date string for usage in safe rendering
        return new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }).toUpperCase();
    },

    // --- Data Fetching ---
    fetchData: async function() {
        const response = await fetch(this.config.redditUrl);
        if (!response.ok) throw new Error("Network response was not ok");
        
        const data = await response.json();
        const posts = data.data.children
            .map(child => child.data)
            .filter(post => {
                const text = (post.selftext || "").toLowerCase();
                const title = (post.title || "").toLowerCase();
                const isSafe = !post.over_18 && 
                               !this.config.blacklist.some(word => text.includes(word) || title.includes(word));
                return isSafe && post.selftext && post.selftext.length > 50;
            })
            .map(post => this.processPost(post));

        // Filter out "Shorts" (Hidden prompts)
        this.state.prompts = posts.filter(p => !this.state.shorts.includes(p.id));
        
        // Sort by Alpha Score
        this.state.prompts.sort((a, b) => b.alphaScore - a.alphaScore);
        
        if (this.state.prompts.length === 0) throw new Error("No valid prompts found");
    },

    loadMockData: function() {
        const mocks = [
            {
                id: "mock_1",
                title: "The Universal Simulator v2.1",
                author: "SystemOne",
                selftext: "Act as a high-fidelity physics and sociology engine. I will provide a starting condition (e.g., 'A meteor hits Paris'), and you will output a minute-by-minute timeline of events, focusing on infrastructure failure and emergency response logistics. Use {{event}} as the variable.",
                ups: 15400,
                created_utc: Date.now() / 1000 - 3600,
                source: "ARCHIVE"
            },
            {
                id: "mock_2",
                title: "Code Refactor: Rust Expert",
                author: "DevNull",
                selftext: "You are a Senior Systems Engineer specializing in Rust. Review the following code for memory safety, concurrency issues, and idiomatic patterns. Explain your reasoning for every change using the format: [ISSUE] -> [FIX] -> [REASON]. Code: {{code_snippet}}",
                ups: 8200,
                created_utc: Date.now() / 1000 - 7200,
                source: "GITHUB"
            },
            {
                id: "mock_3",
                title: "Chain-of-Thought Mathematics",
                author: "EulerIdentity",
                selftext: "Solve the following problem step-by-step. Do not output the answer immediately. 1. Break down the problem into variables. 2. Formulate a hypothesis. 3. Calculate intermediate steps. 4. Verify logic. 5. Output final answer. Problem: {{math_problem}}",
                ups: 5300,
                created_utc: Date.now() / 1000 - 10800,
                source: "REDDIT"
            },
            {
                id: "mock_4",
                title: "Midjourney Photorealism Generator",
                author: "ArtGenius",
                selftext: "/imagine prompt: A cinematic shot of {{subject}}, shot on 35mm film, Kodak Portra 400, f/1.8, natural lighting, highly detailed, 8k --ar 16:9 --v 5.2",
                ups: 4100,
                created_utc: Date.now() / 1000 - 15000,
                source: "DISCORD"
            },
            {
                id: "mock_5",
                title: "Socratic Tutor Mode",
                author: "EduBot",
                selftext: "Do not give me the answer. Instead, ask me guiding questions to help me solve {{topic}} on my own. Adjust the difficulty of your questions based on my responses.",
                ups: 3200,
                created_utc: Date.now() / 1000 - 20000,
                source: "ARCHIVE"
            }
        ];
        
        this.state.prompts = mocks.map(post => this.processPost(post, true));
    },

    // --- Core Logic: The Alpha Score ---
    processPost: function(post, isMock = false) {
        const text = post.selftext || "";
        
        // 1. Complexity Analysis
        const hasVariables = (text.match(/{{.*?}}/g) || []).length;
        const lengthScore = Math.min(text.length / 500, 1) * 10; // Cap at 10
        const keywords = ["Act as", "Step by step", "Chain of thought", "JSON", "Format"].filter(k => text.includes(k)).length * 5;
        
        const complexityScore = (hasVariables * 5) + lengthScore + keywords;

        // 2. Virality Analysis
        const viralityScore = Math.min(Math.log10(post.ups || 1) * 20, 100);

        // 3. Final Alpha
        let alpha = (complexityScore * this.config.scoring.complexityWeight) + 
                    (viralityScore * this.config.scoring.viralityWeight);
        
        // Normalize to 0-100ish (can go higher)
        alpha = Math.round(Math.min(alpha + this.config.scoring.baseScore, 99.9) * 10) / 10;

        return {
            id: post.id || Math.random().toString(36),
            title: post.title,
            author: post.author || "Anon",
            content: text,
            ups: post.ups || 0,
            time: this.formatTime(post.created_utc),
            alphaScore: alpha,
            source: isMock ? post.source : "REDDIT",
            tags: this.generateTags(text, alpha)
        };
    },

    generateTags: function(text, alpha) {
        const tags = [];
        if (alpha > 85) tags.push("HIGH ALPHA");
        if (text.includes("{{")) tags.push("TEMPLATE");
        if (text.includes("JSON")) tags.push("STRUCTURED");
        if (text.includes("Act as")) tags.push("ROLEPLAY");
        if (text.length > 1000) tags.push("LONG-CONTEXT");
        return tags;
    },

    // --- UI Rendering (Safe) ---
    renderMarketMovers: function() {
        const container = document.getElementById('market-movers-list');
        container.textContent = ''; // Clear safely

        this.state.prompts.forEach(p => {
            const div = document.createElement('div');
            // Determine color based on score
            const scoreColor = p.alphaScore > 80 ? 'text-cyan-400' : (p.alphaScore > 60 ? 'text-emerald-400' : 'text-slate-400');
            
            div.className = 'p-2 hover:bg-slate-800/50 cursor-pointer border-l-2 border-transparent hover:border-cyan-500 transition group';
            div.onclick = () => this.loadPrompt(p);
            
            // Safe HTML construction
            // Row 1: ID and Score
            const row1 = document.createElement('div');
            row1.className = 'flex justify-between items-center mb-1';
            
            const idSpan = document.createElement('span');
            idSpan.className = 'text-[10px] font-mono text-slate-500 truncate w-32';
            idSpan.textContent = p.id.substring(0,6) + '...';
            
            const scoreSpan = document.createElement('span');
            scoreSpan.className = `text-xs font-bold font-mono ${scoreColor}`;
            scoreSpan.textContent = p.alphaScore;
            
            row1.appendChild(idSpan);
            row1.appendChild(scoreSpan);
            
            // Row 2: Title
            const titleDiv = document.createElement('div');
            titleDiv.className = 'text-xs text-slate-300 font-bold truncate group-hover:text-white';
            titleDiv.textContent = p.title;

            // Row 3: Author and Ups
            const row3 = document.createElement('div');
            row3.className = 'flex justify-between mt-1';
            
            const authorSpan = document.createElement('span');
            authorSpan.className = 'text-[10px] text-slate-500 font-mono';
            authorSpan.textContent = p.author;
            
            const upSpan = document.createElement('span');
            upSpan.className = 'text-[10px] text-slate-500 font-mono';
            upSpan.innerHTML = `<i class="fas fa-arrow-up text-emerald-500"></i> ${this.formatNumber(p.ups)}`; // InnerHTML safe here as icon is static and ups is numeric formatted

            row3.appendChild(authorSpan);
            row3.appendChild(upSpan);

            div.appendChild(row1);
            div.appendChild(titleDiv);
            div.appendChild(row3);
            
            container.appendChild(div);
        });
    },

    renderTicker: function() {
        const container = document.getElementById('ticker-content');
        if(!this.state.prompts.length) return;
        
        container.textContent = '';

        const fragment = document.createDocumentFragment();
        
        // We create items twice for loop effect
        for (let i = 0; i < 2; i++) {
            this.state.prompts.forEach(p => {
                const change = (Math.random() * 5).toFixed(1);
                const isUp = Math.random() > 0.3;
                const color = isUp ? 'text-[#10b981]' : 'text-[#ef4444]'; // Using specific colors classes from HTML didn't work well with textContent, sticking to classes
                const sign = isUp ? '+' : '-';
                
                const span = document.createElement('span');
                span.className = 'ticker-item';
                
                const symbolSpan = document.createElement('span');
                symbolSpan.className = 'symbol';
                symbolSpan.textContent = p.title.toUpperCase().substring(0, 15);
                
                const valueSpan = document.createElement('span');
                valueSpan.className = isUp ? 'up' : 'down';
                valueSpan.textContent = ` ${p.alphaScore} (${sign}${change}%)`;

                span.appendChild(symbolSpan);
                span.appendChild(valueSpan);
                fragment.appendChild(span);
            });
        }
        
        container.appendChild(fragment);
    },

    switchTab: function(tab) {
        this.state.activeTab = tab;
        
        // Update Buttons
        const btnBriefing = document.getElementById('tab-briefing');
        const btnPortfolio = document.getElementById('tab-portfolio');
        
        if (tab === 'briefing') {
            btnBriefing.className = 'flex-1 py-3 text-xs font-bold font-mono text-cyan-400 border-b-2 border-cyan-500 bg-slate-800/50 transition';
            btnPortfolio.className = 'flex-1 py-3 text-xs font-bold font-mono text-slate-500 hover:text-slate-300 border-b-2 border-transparent transition';
            this.renderBriefing();
        } else {
            btnBriefing.className = 'flex-1 py-3 text-xs font-bold font-mono text-slate-500 hover:text-slate-300 border-b-2 border-transparent transition';
            btnPortfolio.className = 'flex-1 py-3 text-xs font-bold font-mono text-cyan-400 border-b-2 border-cyan-500 bg-slate-800/50 transition';
            this.renderPortfolio();
        }
    },

    renderBriefing: function() {
        const container = document.getElementById('right-panel-content');
        container.className = 'flex-1 p-4 overflow-y-auto custom-scrollbar bg-[#e2e8f0] text-slate-800 dot-matrix relative shadow-inner';
        container.textContent = '';

        // Overlay div
        const overlay = document.createElement('div');
        overlay.className = 'absolute inset-0 bg-gradient-to-b from-white/20 to-transparent pointer-events-none';
        container.appendChild(overlay);

        // Header
        const header = document.createElement('div');
        header.className = 'text-center border-b-2 border-slate-800 pb-2 mb-4';
        header.innerHTML = `<h3 class="text-xl font-bold tracking-tight">DAILY ALPHA</h3>
                            <div class="text-[10px] uppercase">Vol. 235 &bull; <span id="briefing-date">${this.updateDate()}</span></div>`;
        container.appendChild(header);

        // Content Wrapper
        const content = document.createElement('div');
        content.className = 'space-y-6 text-sm';
        
        if (this.state.prompts.length > 0) {
            const top = this.state.prompts[0];
            
            // Top Gainer Section
            const section1 = document.createElement('div');
            section1.innerHTML = '<h4 class="font-bold uppercase border-b border-slate-400 mb-1 text-xs">Top Gainer</h4>';
            
            const p1 = document.createElement('p');
            p1.className = 'leading-snug';
            
            const titleSpan = document.createElement('span');
            titleSpan.className = 'font-bold text-slate-900';
            titleSpan.textContent = top.title;
            
            const scoreBr = document.createElement('br');
            const scoreSpan = document.createElement('span');
            scoreSpan.className = 'text-xs text-slate-600';
            scoreSpan.textContent = `Alpha Score: ${top.alphaScore}`;
            
            p1.appendChild(titleSpan);
            p1.appendChild(scoreBr);
            p1.appendChild(scoreSpan);
            section1.appendChild(p1);
            content.appendChild(section1);
        }

        // Static Sections (Safe to use innerHTML for static content)
        const staticSections = document.createElement('div');
        staticSections.innerHTML = `
            <div class="mt-6">
                <h4 class="font-bold uppercase border-b border-slate-400 mb-1 text-xs">Sector Watch</h4>
                <p class="leading-snug">"Coding" prompts are up 15% today driven by new React patterns.</p>
            </div>
            <div class="mt-6">
                <h4 class="font-bold uppercase border-b border-slate-400 mb-1 text-xs">Sentiment</h4>
                <div class="flex items-center gap-2 mt-1">
                    <div class="flex-1 h-2 bg-slate-300 rounded-full overflow-hidden border border-slate-400">
                        <div class="h-full bg-slate-800 w-[75%]"></div>
                    </div>
                    <span class="text-xs font-bold">BULLISH</span>
                </div>
            </div>
        `;
        content.appendChild(staticSections);
        container.appendChild(content);

        // Footer
        const footer = document.createElement('div');
        footer.className = 'mt-8 text-[10px] text-center opacity-70';
        footer.textContent = '* PROMPT ALPHA IS NOT FINANCIAL ADVICE *';
        container.appendChild(footer);
    },

    renderPortfolio: function() {
        const container = document.getElementById('right-panel-content');
        // Switch styling for portfolio view (dark mode style instead of dot matrix)
        container.className = 'flex-1 p-2 overflow-y-auto custom-scrollbar bg-[#0b1221] text-slate-200 relative shadow-inner';
        container.textContent = '';

        if (this.state.portfolio.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'h-full flex flex-col items-center justify-center text-slate-500 text-xs font-mono text-center p-4';
            empty.innerHTML = '<i class="fas fa-wallet text-2xl mb-2 opacity-50"></i><p>Portfolio Empty.<br>Buy prompts to save them.</p>';
            container.appendChild(empty);
            return;
        }

        this.state.portfolio.forEach(p => {
            const div = document.createElement('div');
            div.className = 'bg-[#0f172a] border border-slate-800 p-3 rounded mb-2 hover:border-cyan-500 transition cursor-pointer group';
            div.onclick = () => this.loadPrompt(p);

            const header = document.createElement('div');
            header.className = 'flex justify-between items-start mb-2';
            
            const title = document.createElement('div');
            title.className = 'text-xs font-bold text-white truncate flex-1 mr-2';
            title.textContent = p.title;

            const score = document.createElement('div');
            score.className = 'text-[10px] font-mono text-cyan-400 border border-cyan-900 px-1 rounded';
            score.textContent = p.alphaScore;

            header.appendChild(title);
            header.appendChild(score);
            
            const meta = document.createElement('div');
            meta.className = 'text-[10px] text-slate-500 font-mono flex justify-between';
            meta.textContent = `u/${p.author}`;

            div.appendChild(header);
            div.appendChild(meta);
            container.appendChild(div);
        });
    },

    // --- Interaction ---
    loadPrompt: function(prompt) {
        this.state.currentPrompt = prompt;
        
        document.getElementById('trading-desk-empty').classList.add('hidden');
        document.getElementById('trading-desk-content').classList.remove('hidden');

        // Safe assignment
        document.getElementById('detail-title').textContent = prompt.title;
        document.getElementById('detail-author').textContent = `u/${prompt.author}`;
        document.getElementById('detail-upvotes').textContent = this.formatNumber(prompt.ups);
        document.getElementById('detail-time').textContent = prompt.time;
        document.getElementById('detail-score').textContent = prompt.alphaScore;
        document.getElementById('detail-body').textContent = prompt.content;
        
        const tagsContainer = document.getElementById('detail-tags');
        tagsContainer.textContent = '';
        prompt.tags.forEach(t => {
            const span = document.createElement('span');
            span.className = 'text-[10px] px-1.5 py-0.5 bg-cyan-900/30 text-cyan-400 border border-cyan-800 rounded';
            span.textContent = t;
            tagsContainer.appendChild(span);
        });

        document.getElementById('detail-tokens').textContent = `~${Math.ceil(prompt.content.length / 4)}`;
    },

    buy: function() {
        if (!this.state.currentPrompt) return;
        
        navigator.clipboard.writeText(this.state.currentPrompt.content).then(() => {
            alert(`LONG POSITION OPENED: Copied "${this.state.currentPrompt.title}"`);
        }).catch(err => console.error('Failed to copy', err));

        if (!this.state.portfolio.some(p => p.id === this.state.currentPrompt.id)) {
            this.state.portfolio.push(this.state.currentPrompt);
            localStorage.setItem('pa_portfolio', JSON.stringify(this.state.portfolio));
            // Update view if portfolio is active
            if (this.state.activeTab === 'portfolio') {
                this.renderPortfolio();
            }
        }
    },

    short: function() {
        if (!this.state.currentPrompt) return;
        
        const id = this.state.currentPrompt.id;
        
        if (!this.state.shorts.includes(id)) {
            this.state.shorts.push(id);
            localStorage.setItem('pa_shorts', JSON.stringify(this.state.shorts));
        }

        this.state.prompts = this.state.prompts.filter(p => p.id !== id);
        this.renderMarketMovers();

        document.getElementById('trading-desk-empty').classList.remove('hidden');
        document.getElementById('trading-desk-content').classList.add('hidden');
        
        alert("SHORT POSITION OPENED: Prompt hidden from feed.");
    },

    test: function() {
        if (!this.state.currentPrompt) return;
        
        const modal = document.getElementById('test-modal');
        const input = document.getElementById('test-prompt-input');
        
        input.value = this.state.currentPrompt.content;
        modal.classList.remove('hidden');
    },

    // --- Helpers ---
    formatNumber: function(num) {
        if (num >= 1000) return (num / 1000).toFixed(1) + 'k';
        return num;
    },

    formatTime: function(timestamp) {
        const date = new Date(timestamp * (timestamp > 1e11 ? 1 : 1000));
        const now = new Date();
        const diffHrs = Math.round((now - date) / 36e5);
        
        if (diffHrs < 24) return `${diffHrs}h ago`;
        return `${Math.round(diffHrs / 24)}d ago`;
    }
};

document.addEventListener('DOMContentLoaded', () => {
    PromptAlpha.init();
});
