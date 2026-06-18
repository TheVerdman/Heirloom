FROM rust:1.91-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY Cargo.toml Cargo.lock ./
COPY heirloom-kernels/Cargo.toml heirloom-kernels/Cargo.toml
COPY heirloom-kernels/src heirloom-kernels/src
COPY src src
COPY tests tests
COPY tools tools
COPY scripts scripts
COPY README.md ARCHITECTURE.md HARD_MODE.md OPERATORS.md DISPATCH.md ALIASING.md EXTENDING.md PROGRESS.md ./

RUN cargo fetch

CMD ["./scripts/validate.sh"]
