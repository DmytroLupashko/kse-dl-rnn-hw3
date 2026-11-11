# Local targets driving Colab VM through `colab` CLI. Run from repo root on own machine.

# Local secrets, e.g. WANDB_API_KEY. Optional: missing file ignored
-include .env

# CONFIG, SESSION, GPU have no defaults: pass on command line, e.g. make up SESSION=ner GPU=T4
# Seconds `colab exec` waits for remote command
TIMEOUT ?= 86400
REMOTE := /content/hw3
TMP := /tmp/hw3
# Run outputs tarball on VM
OUT := /content/out.tgz

# $(1): variable names that must be set, checked before recipe runs
require = $(foreach v,$(1),$(if $($(v)),,$(error $(v) required, pass $(v)=... on command line)))

# $(1): shell command run on VM inside $(REMOTE)
remote = printf '%s\n' "!cd $(REMOTE) && $(1)" | colab exec -s $(SESSION) --timeout $(TIMEOUT) \
	$(if $(WANDB_API_KEY),--env WANDB_API_KEY=$(WANDB_API_KEY))

# $(1): local tarball, extracted into $(REMOTE). Sent in 50m parts: Colab contents API rejects ~250MB files
push = rm -f $(1).part-* && split -b 50m $(1) $(1).part- && \
	for p in $(1).part-*; do colab upload -s $(SESSION) $$p /content/$$(basename $$p) || exit 1; done && \
	printf '%s\n' "!cat /content/$(notdir $(1)).part-* > /content/$(notdir $(1)) && rm /content/$(notdir $(1)).part-* && mkdir -p $(REMOTE) && tar xzf /content/$(notdir $(1)) -C $(REMOTE)" | colab exec -s $(SESSION)

# VM side: run outputs into $(OUT) parts, part list in $(OUT).parts. Run after `rm -f $(OUT)*`
stage = tar czf $(OUT) data/processed data/wandb && split -b 50m $(OUT) $(OUT).part- && rm $(OUT) && ls $(OUT).part-* > $(OUT).parts

# Local side: downloads staged parts, extracts into data/. Fails if nothing staged.
# `colab exec` exits 0 even when remote command fails, so missing $(OUT).parts is failure signal
fetch = rm -f $(TMP)/out.tgz* && mkdir -p $(TMP) && colab download -s $(SESSION) $(OUT).parts $(TMP)/out.tgz.parts && \
	for p in $$(cat $(TMP)/out.tgz.parts); do colab download -s $(SESSION) $$p $(TMP)/$$(basename $$p) || exit 1; done && \
	cat $(TMP)/out.tgz.part-* | tar xzf - && rm $(TMP)/out.tgz.part-*

.PHONY: up down install data sync train pack

up:
	$(call require,SESSION GPU)
	colab new -s $(SESSION) --gpu $(GPU)

down:
	$(call require,SESSION)
	colab stop -s $(SESSION)

# Colab python not 3.14, so no pinned pyproject install
install:
	$(call require,SESSION)
	colab install -s $(SESSION) transformers accelerate seqeval evaluate wandb pyyaml

# Raw datasets, once per VM
data:
	$(call require,SESSION)
	mkdir -p $(TMP) && tar czf $(TMP)/data.tgz data/ukr-synth
	$(call push,$(TMP)/data.tgz)

# Code and configs, before each run
sync:
	$(call require,SESSION)
	mkdir -p $(TMP) && tar czf $(TMP)/code.tgz --exclude=__pycache__ src scripts configs
	$(call push,$(TMP)/code.tgz)

# Needs WANDB_API_KEY in .env or environment. Outputs packed into data/ only on success
train: sync
	$(call require,SESSION CONFIG)
	$(call remote,rm -f $(OUT)* && PYTHONPATH=src python scripts/train_ner.py $(CONFIG) && $(stage))
	$(fetch)

# Fetches VM run outputs (data/processed, data/wandb) into local data/
pack:
	$(call require,SESSION)
	$(call remote,rm -f $(OUT)* && $(stage))
	$(fetch)
