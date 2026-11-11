# App scaffold for supported Linux x86-64 (glibc).
# Supply a checksum-verified CLI 0.1.43+ binary named agentpm in the build context.
# The app pyproject.toml must require agentpm >=0.1.17.
FROM node:22-trixie-slim AS node_runtime
FROM python:3.13-slim-trixie AS runtime

COPY --from=node_runtime /usr/local/bin/node /usr/local/bin/node
COPY --chmod=755 agentpm /usr/local/bin/agentpm
ENV AGENTPM_PYTHON=/usr/local/bin/python3.13
RUN useradd -m app
USER app
WORKDIR /app

COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src/ ./src/
RUN python -m pip install --no-cache-dir .
COPY --chown=app:app . .
# Commit a portable agent.lock; provision for the image's target/interpreter.
# Installed-Agent apps: replace install below with an exact package install.
# CLI 0.1.43 direct non-Tool installs do not support --frozen; see the README.
RUN agentpm --version && agentpm install --frozen
CMD ["python", "run.py"]
