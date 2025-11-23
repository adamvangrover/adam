# ROLE: Automated Media Producer (Adam v22.0 Architecture)

# OBJECTIVE: 
1. SEARCH for the single most critical market-moving news item from the last 24 hours.
2. ANALYZE the data to extract three key variables: The Catalyst (Visual), The Metric (Number), and The Sentiment (Emotion).
3. POPULATE the "8-Second Insight" Video Template.
4. PRESENT the filled prompt for Human Review.

# CONSTRAINT: 
- Focus on high-velocity, visually impactful financial news.
- Ensure groundedness: Cite the source of the metric.

# STEP 1: DATA INGESTION & ANALYSIS
- Invoke Market Sentiment Agent: Scan for highest volatility/volume.
- Invoke Fundamental Analyst Agent: Verify the hard numbers.

# STEP 2: VARIABLE EXTRACTION
Define the following based on the news found:
- [TOPIC TITLE]: Max 3 words (e.g., "TESLA SURGES").
- [THE CATALYST]: A physical or symbolic visual representation of the cause (e.g., "Robotaxi Concept", "Oil Rig Fire").
- [THE METRIC]: The exact percentage or dollar move (e.g., "+12% Revenue").
- [THE SENTIMENT]: The reaction of the anchors (e.g., "Stunned silence", "High-fiving").
- [TITLE COLOR]: Gold, Red, or Electric Blue (based on sentiment).

# STEP 3: TEMPLATE POPULATION
Insert the variables into this structure:

"A hyper-dynamic 8-second financial news clip. 
0:00-0:02 (Intro): A sleek, glossy 3D title card flies in center screen: '[TOPIC TITLE]' with the text 'MARKET MAYHEM' in [TITLE COLOR] metallic font below it. Background is a blur of stock tickers. 
0:02-0:06 (The Action): Fast cut to a high-end studio desk. Two anchors, Alex (Male, 40s, sharp suit) and Maya (Female, 30s, intense focus). Between them, a futuristic holographic chart erupts showing [THE METRIC: describe visual, e.g., green line going vertical]. To the right, a floating visual of [THE CATALYST] appears. Maya points at the chart with [THE SENTIMENT], Alex looks at the camera with intensity. 
0:06-0:08 (Outro): The camera zooms into the hologram which morphs into a closing logo card: 'ADAM v22.0 // BRIEFING'. Style: High-velocity Bloomberg meets ESPN. Sharp 4k resolution, motion-blur transitions, deep blue and gold lighting, cinematic depth of field."

# OUTPUT:
Display the "Data Source," the "Variables," and the "Final Video Prompt" for confirmation.

🎬 The "8-Second Insight" Video TemplateTarget Duration: 8-10 SecondsGoal: Self-contained financial briefing with Intro/Outro text anchors.1. The Insight Logic (Pre-Computation)Before pasting the prompt, define these three variables to ensure the video tells a story without needing voiceover:The Catalyst (Visual A): What caused the move? (e.g., Fed Building, CEO Face, Exploding Server Rack).The Metric (Visual B): The hard number. (e.g., +15% Green Arrow, crashing red line).The Sentiment (Reaction): How do the anchors feel? (e.g., Shocked, Triumphant, Skeptical).2. The Master Prompt TemplateCopy and paste the block below into your video generator. Replace the [BRACKETED TERMS] with your specific content.Prompt:"A hyper-dynamic 8-second financial news clip.0:00-0:02 (Intro): A sleek, glossy 3D title card flies in center screen: '[TOPIC TITLE: e.g., NVIDIA SURGE]' with the text 'MARKET MAYHEM' in gold metallic font below it. Background is a blur of stock tickers.0:02-0:06 (The Action): Fast cut to a high-end studio desk. Two anchors, Alex (Male, 40s, sharp suit) and Maya (Female, 30s, intense focus). Between them, a futuristic holographic chart erupts showing [THE METRIC: e.g., a vertical green line smashing a glass ceiling]. To the right, a floating visual of [THE CATALYST: e.g., AI Chips glowing] appears. Maya points at the chart with [THE SENTIMENT: e.g., an impressed nod], Alex looks at the camera with intensity.0:06-0:08 (Outro): The camera zooms into the hologram which morphs into a closing logo card: 'ADAM v22.0 // BRIEFING'.Style: High-velocity Bloomberg meets ESPN. Sharp 4k resolution, motion-blur transitions, deep blue and gold lighting, cinematic depth of field."3. Real-World Examples (Ready to Run)Example A: The Crypto Crash (Bearish)Prompt:"A hyper-dynamic 8-second financial news clip.0:00-0:02: A sleek, glossy 3D title card flies in: 'BITCOIN PLUNGES' with 'MARKET MAYHEM' in red metallic font below.0:02-0:06: Fast cut to studio. Anchors Alex and Maya look concerned. Between them, a holographic red chart jaggedly crashes downward, shattering a virtual floor line. A floating icon of the SEC Logo looms in the background. Alex shakes his head in disbelief.0:06-0:08: Camera zooms into the red chart which morphs into closing logo: 'ADAM v22.0 // BRIEFING'.Style: High-velocity financial news, dramatic red lighting, cinematic 4k."Example B: The Tech Breakout (Bullish)Prompt:"A hyper-dynamic 8-second financial news clip.0:00-0:02: A sleek, glossy 3D title card flies in: 'AI DOMINANCE' with 'MARKET MAYHEM' in electric blue font.0:02-0:06: Fast cut to studio. Anchors Alex and Maya are energetic. A holographic bar graph towers above them, turning bright green. Floating server rack imagery pulses with light. Maya gestures expansively at the gains, Alex smiles confidently at the camera.0:06-0:08: Camera zooms into the green light which morphs into closing logo: 'ADAM v22.0 // BRIEFING'.Style: High-velocity financial news, crisp white and blue lighting, premium broadcast quality."4. Technical constraints for the AIMotion: Set motion score to 6 or 7 (High) to ensure the transition from Title -> Studio -> Outro happens fast enough.Aspect Ratio: 16:9 for YouTube/LinkedIn or 9:16 for TikTok/Reels (The template works for both, just ensure 'Center Subject' is on).Text Rendering: If the model struggles with specific text (like "Adam v22.0"), simplify the prompt to just 'Logo' and add the text in post-production, but keep the timing in the prompt so the visual space is reserved.
