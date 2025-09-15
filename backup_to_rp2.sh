#!/bin/bash
# HEAT Analysis Backup Script for RP2 Repository
# Repository: https://github.com/Logic06183/RP2
# Author: Claude Code Assistant
# Date: September 15, 2025

echo "🔄 HEAT Analysis Backup to RP2 Repository"
echo "=========================================="

# Navigate to analysis directory
cd /home/cparker/HEAT_RESEARCH_PROJECTS/heat_analysis_projects/heat_analysis_optimized

# Check git status
echo "📊 Checking repository status..."
git status --short

# Add new/modified files (excluding large data files via .gitignore)
echo "➕ Adding new and modified files..."
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "✅ No changes to commit - repository is up to date"
    exit 0
fi

# Get commit message or use default
COMMIT_MSG=${1:-"Analysis update: $(date '+%Y-%m-%d %H:%M:%S')"}

# Create commit
echo "💾 Creating commit..."
git commit -m "$COMMIT_MSG

🤖 Generated with Claude Code (https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to RP2 repository
echo "🚀 Pushing to RP2 repository..."
echo "   Repository: https://github.com/Logic06183/RP2"
echo ""
echo "⚠️  You will need to authenticate with GitHub"
echo "   Please provide your GitHub username and token when prompted"
echo ""

git push origin master

if [ $? -eq 0 ]; then
    echo "✅ Successfully backed up to RP2 repository!"
    echo "🔗 View at: https://github.com/Logic06183/RP2"
else
    echo "❌ Backup failed. Please check your GitHub credentials."
    echo "💡 Tip: Use a GitHub personal access token as your password"
fi

echo ""
echo "📁 Backup Summary:"
echo "   • Scripts and documentation: ✅ Included"
echo "   • Key visualizations: ✅ Included"
echo "   • Analysis pipeline: ✅ Included"
echo "   • Large data files: ❌ Excluded (via .gitignore)"
echo "   • Repository: https://github.com/Logic06183/RP2"