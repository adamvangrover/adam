# Static Mock Mode

This directory formalizes the "Static Mock Mode" for the Adam System. The extensive mocking here is deliberate. When analyzing distressed situations or testing multi-agent reactions to fragmented data, the system falls back to these lightweight Python static proxies.

This enables edge deployments and UI iterations to run gracefully without bottlenecking on the heavy Rust execution layer or live data pipelines.

To run the system in this mode, configure the environment with `ENV=demo`.
