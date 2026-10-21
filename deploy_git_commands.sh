#!/usr/bin/env bash
set -euo pipefail

echo "1. Running Pre-Deployment Unit & Stress Test Suite..."
PYTHONPATH=. python3 -m unittest discover -s tests -v

echo "2. Initializing Git Repository & Committing Project..."
git init
git branch -M main
git add .
git commit -m "feat: deploy Autonomous Data Analyst Copilot (Layers 1-4, AST sandbox, Docker)"

echo "3. Ready to push to GitHub / Render / Hugging Face Spaces:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/autonomous-data-analyst-copilot.git"
echo "   git push -u origin main"
