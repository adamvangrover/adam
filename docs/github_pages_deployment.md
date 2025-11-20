# Deploying Adam Mission Control to GitHub Pages

The root `index.html` has been redesigned to serve as a static "Mission Control" dashboard for the Adam system. It is self-contained and ready for deployment on GitHub Pages.

## Deployment Steps

1.  **Enable GitHub Pages:**
    *   Go to the repository settings on GitHub.
    *   Navigate to the "Pages" section.
    *   Under "Build and deployment", select "Deploy from a branch".
    *   Select the branch you want to deploy (e.g., `main` or `master`) and ensure the folder is set to `/` (root).
    *   Click "Save".

2.  **Verification:**
    *   Once the deployment action finishes, GitHub will provide a URL (usually `https://<username>.github.io/<repo-name>/`).
    *   Visit the URL to see the Mission Control dashboard.

## How it Works

*   **Single Entry Point:** The `index.html` in the root acts as the entry point.
*   **Relative Links:** All links to documentation, code, and other artifacts are relative (e.g., `docs/getting_started.md`). This allows them to work correctly both locally (viewing the file in a browser) and on GitHub Pages.
    *   *Note:* Links to `.py` files will typically display the raw code or trigger a download, depending on the browser.
    *   *Note:* Links to Markdown files (`.md`) will display the raw Markdown on GitHub Pages unless a Jekyll theme processes them. The dashboard is designed to link to the source for developer reference.
*   **External Assets:** The dashboard uses CDN links for:
    *   Tailwind CSS (Styling)
    *   Google Fonts (Typography)
    *   Lucide Icons (Iconography)
    *   *Ensure the viewing machine has internet access to load these assets.*

## Legacy Artifacts

The `docs/ui_archive_v1/` directory contains a snapshot of all static HTML files found in the repository at the time of the migration. This serves as a historical record of previous UI experiments (mockups, newsletters, legacy tool interfaces). These can be browsed via the "UI Archive" link in the dashboard sidebar.
