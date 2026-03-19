SHELL := /bin/bash
MODEL ?= Qwen/Qwen3-0.6B
TOKENS ?= 30
CTX ?= 2048
PY := .venv/bin/python

.PHONY: build test fmt check profile profile-raw chat run clean-cache

build:
	source env-llvm21-dev.sh && cargo build --release

test:
	source env-llvm21-dev.sh && cargo test

fmt:
	source env-llvm21-dev.sh && cargo fmt

check:
	source env-llvm21-dev.sh && cargo fmt -- --check && cargo build --release

# Profile decode performance with a nice table
profile: build
	source env-llvm21-dev.sh && $(PY) scripts/profile.py $(MODEL) --tokens $(TOKENS) --ctx $(CTX)

# Profile with raw per-kernel output
profile-raw: build
	source env-llvm21-dev.sh && $(PY) scripts/profile.py $(MODEL) --tokens $(TOKENS) --ctx $(CTX) --raw

# Interactive chat
chat: build
	source env-llvm21-dev.sh && cargo run --release -- chat $(MODEL) --ctx $(CTX)

# Single prompt generation
run: build
	source env-llvm21-dev.sh && cargo run --release -- run $(MODEL) --prompt "$(PROMPT)" --max-tokens $(TOKENS) --ctx $(CTX)

clean-cache:
	source env-llvm21-dev.sh && cargo run --release -- clean-cache
