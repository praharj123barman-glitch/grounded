# Deploying Grounded

The app is a FastAPI service in a Dockerfile that binds to `$PORT`. On startup it
ingests the sample corpus (when `BOOTSTRAP_ON_START=true`), so a fresh deploy comes
up answerable with no manual step. Both options below are free and connect to this
GitHub repo. The only manual part is setting your `GOOGLE_API_KEY` secret, which
cannot be committed.

## Option A — Render (blueprint included)

1. Push this repo to GitHub (done).
2. Render dashboard > New > Blueprint > connect `praharj123barman-glitch/grounded`.
3. Render reads `render.yaml`. Set the `GOOGLE_API_KEY` secret when prompted.
4. Deploy. Health check: `GET /health`. Ask: `POST /ask {"question": "..."}`.

## Option B — Hugging Face Spaces (Docker)

1. Create a new Space, SDK = Docker.
2. Push this repo to the Space (or connect the GitHub repo).
3. Copy the frontmatter from `deploy/huggingface_space_README.md` to the top of the
   Space's `README.md`.
4. In Space settings, add secrets: `GOOGLE_API_KEY` and `BOOTSTRAP_ON_START=true`.
5. The Space builds the Dockerfile and serves on the port in the frontmatter.

## Local (Docker)

```bash
docker build -t grounded .
docker run -p 8000:8000 -e GOOGLE_API_KEY=your_key -e BOOTSTRAP_ON_START=true grounded
# then: curl localhost:8000/health
```
