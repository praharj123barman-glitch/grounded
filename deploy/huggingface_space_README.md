---
title: Grounded
emoji: 📄
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# Grounded

Agentic RAG research assistant with a citation-grounded answer API. Copy the
frontmatter above to the top of your Space's README.md, then set the Space secrets
`GOOGLE_API_KEY` and `BOOTSTRAP_ON_START=true`.

Endpoints: `GET /health`, `POST /ask {"question": "..."}`.
