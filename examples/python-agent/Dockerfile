# ---- Stage 1: vendor Node 20
FROM node:20-bullseye AS node20

# ---- Stage 2: Python runtime
FROM python:3.11-slim AS runtime

# Use vendored Node 20 for all AgentPM tools
COPY --from=node20 /usr/local/bin/node /opt/node20/bin/node
RUN chmod +x /opt/node20/bin/node && /opt/node20/bin/node -v
ENV AGENTPM_NODE=/opt/node20/bin/node

# Normal app setup
RUN useradd -m app
USER app
WORKDIR /app

COPY --chown=app:app .agentpm /app/.agentpm
COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src/ ./src/
COPY --chown=app:app vendor/agentpm-*.whl /wheels/

RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir /wheels/agentpm-*.whl \
 && python -m pip install --no-cache-dir .

COPY --chown=app:app .. .

ENV AGENTPM_DEBUG=1
#ENV AGENTPM_NODE_JITLESS=1
#ENV AGENTPM_NODE_OLD_SPACE_MB=1024

CMD ["python", "run.py"]
