# Local targets driving Colab VM through `colab` CLI. Run from repo root on own machine.

# Local secrets, e.g. WANDB_API_KEY. Optional: missing file ignored
-include .env

# CONFIG, SESSION, GPU have no defaults: pass on command line, e.g. make up SESSION=ner GPU=T4
# Seconds `colab exec` waits for remote command
TIMEOUT ?= 86400
REMOTE := /content/hw3
TMP := /tmp/hw3
# Results folder per run, named after config's run_name
RUN ?= $(shell sed -n 's/^run_name: *//p' $(CONFIG))
DRIVE_REMOTE := /content/drive/MyDrive/Colab/Deep Learning Homework 3
# Google Drive for Desktop mirror of DRIVE_REMOTE
DRIVE_LOCAL ?= $(firstword $(wildcard $(HOME)/Library/CloudStorage/GoogleDrive-*))/My Drive/Colab/Deep Learning Homework 3

# $(1): variable names that must be set, checked before recipe runs
require = $(foreach v,$(1),$(if $($(v)),,$(error $(v) required, pass $(v)=... on command line)))

# $(1): shell command run on VM inside $(REMOTE)
remote = printf '%s\n' "!cd $(REMOTE) && $(1)" | colab exec -s $(SESSION) --timeout $(TIMEOUT) \
	$(if $(WANDB_API_KEY),--env WANDB_API_KEY=$(WANDB_API_KEY))

# $(1): local tarball, extracted into $(REMOTE). Sent in 50m parts: Colab contents API rejects ~250MB files
push = rm -f $(1).part-* && split -b 50m $(1) $(1).part- && \
	for p in $(1).part-*; do colab upload -s $(SESSION) $$p /content/$$(basename $$p) || exit 1; done && \
	printf '%s\n' "!cat /content/$(notdir $(1)).part-* > /content/$(notdir $(1)) && rm /content/$(notdir $(1)).part-* && mkdir -p $(REMOTE) && tar xzf /content/$(notdir $(1)) -C $(REMOTE)" | colab exec -s $(SESSION)

.PHONY: up down install data sync train pack

up:
	$(call require,SESSION GPU)
	colab new -s $(SESSION) --gpu $(GPU)
	colab drivemount -s $(SESSION)

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
	mkdir -p $(TMP) && tar czf $(TMP)/data.tgz data/ukr-synth data/nlp-getting-started
	$(call push,$(TMP)/data.tgz)

# Code and configs, before each run
sync:
	$(call require,SESSION)
	mkdir -p $(TMP) && tar czf $(TMP)/code.tgz --exclude=__pycache__ src scripts configs
	$(call push,$(TMP)/code.tgz)

# Needs WANDB_API_KEY in .env or environment. Outputs copied to Drive only on success
train: sync
	$(call require,SESSION CONFIG)
	$(call remote,PYTHONPATH=src python scripts/train_ner.py $(CONFIG) && mkdir -p '$(DRIVE_REMOTE)/$(RUN)' && cp -r data/processed data/wandb '$(DRIVE_REMOTE)/$(RUN)/')

# Fetches run outputs from local Drive mirror into data/, no VM needed
pack:
	$(call require,CONFIG)
	rsync -a "$(DRIVE_LOCAL)/$(RUN)/" data/
