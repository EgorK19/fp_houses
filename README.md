# Small Template

A minimal template designed to mirror project architecture and provide base script execution.

## Getting Started

1. Clone the repository and navigate to the project directory.
2. Initialize and sync the environment:
   ```bash
   uv sync
   ```

## Install Recommended VS Code Extensions

### Option 1: Via VS Code GUI
1. Open the project in VS Code.
2. Open the Extensions tab (`Ctrl+Shift+X`).
3. Type `@recommended` in the search bar.
4. Click **Install** under the "Workspace Recommendations" section.

### Option 2: Via Terminal

* **Windows (PowerShell):**
  ```powershell
  Get-Content .vscode\extensions.json | ConvertFrom-Json | ForEach-Object { \(_.recommendations } \vert{} ForEach-Object { code --install-extension\)_ }
  ```
* **Bash / Git Bash:**
  ```bash
  cat .vscode/extensions.json | grep -oP '"\K[^"]+(?=")' | grep -v '^recommendations\$' | xargs -L 1 code --install-extension
  ```
