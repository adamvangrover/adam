#!/bin/bash
# 1. Create Virtual Environment
echo "Creating virtual environment 'venv-tinker'..."
python3 -m venv venv-tinker

# 2. Activate Environment
source venv-tinker/bin/activate

# 3. Install Tinker SDK and dependencies
echo "Installing Tinker SDK and Adam dependencies..."
pip install tinker pandas matplotlib neo4j python-dotenv

# 4. Clone Tinker Cookbook (for recipes)
if [ ! -d "tinker-cookbook" ]; then
    echo "Cloning Tinker Cookbook..."
    git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
    cd tinker-cookbook
    pip install -e .
    cd ..
else
    echo "Tinker Cookbook already exists."
fi

echo "Setup complete. Don't forget to verify your API key in the .env file!"
