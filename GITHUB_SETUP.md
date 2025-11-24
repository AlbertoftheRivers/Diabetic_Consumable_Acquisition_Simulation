# GitHub Repository Setup Guide

Follow these steps to create and push your project to GitHub:

## Step 1: Initialize Git Repository

```bash
git init
```

## Step 2: Add All Files

```bash
git add .
```

## Step 3: Create Initial Commit

```bash
git commit -m "Initial commit: Diabetic patient consumable acquisition simulation"
```

## Step 4: Create GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Name your repository (e.g., `diabetic-patient-simulation` or `healthcare-systems-logistics`)
5. Add a description: "Simulation model for diabetic patient consumable acquisition in Stockholm"
6. Choose public or private
7. **DO NOT** initialize with README, .gitignore, or license (we already have these)
8. Click "Create repository"

## Step 5: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these (replace `YOUR_USERNAME` and `YOUR_REPO_NAME`):

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

## Alternative: Using GitHub CLI

If you have GitHub CLI installed:

```bash
gh repo create YOUR_REPO_NAME --public --source=. --remote=origin --push
```

## Files Included

The following files will be committed:
- `model/` - All Python model files
- `result/` - Results folder (CSV and plots)
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies
- `run_simulation.py` - Main simulation runner
- `.gitignore` - Git ignore rules

## Files Excluded (via .gitignore)

- `venv/` - Virtual environment (not tracked)
- `__pycache__/` - Python cache files
- `*.pyc` - Compiled Python files

## Verify Setup

After pushing, verify your repository:

```bash
git status
git log --oneline
```

Your repository should now be live on GitHub!

