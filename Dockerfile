FROM rust:1.95.0-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3 python3-dev shellcheck zsh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . .

RUN ./scripts/check_toolchain.sh && cargo fetch --locked

CMD ["./scripts/validate.sh"]
