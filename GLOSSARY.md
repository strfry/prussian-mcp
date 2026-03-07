# Glossary: Technical Terms

## Deployment

**Deploy** (or Deployment)
: The process of moving code changes from a development/local environment to a production/live environment. In this project, deploying means copying updated files (`chatbot.html`, `chatbot.js`, and related assets) from the git repository (`/home/strfry/prussian-dictionary`) to the web server directory (`/home/strfry/html/`), making them publicly accessible via the web.

Example workflow in this project:
1. Edit files locally (e.g., change title in `chatbot.html`)
2. Commit changes to git: `git add` and `git commit`
3. Copy files to web root: `cp chatbot.html /home/strfry/html/`
4. Verify deployment: Check that changes appear on the live website

Related terms:
- **Git commit**: Recording changes to version control
- **Web root**: The directory where the live website files are served from (`/home/strfry/html/`)
