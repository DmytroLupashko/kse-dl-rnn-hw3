# Local targets driving Colab VM through `colab` CLI. Run from repo root on own machine.

# Local secrets: WANDB_API_KEY, KAGGLE_USERNAME + KAGGLE_KEY (or KAGGLE_API_TOKEN). Optional: missing file ignored
-include .env
# Secrets passed to VM commands when set
SECRETS := WANDB_API_KEY KAGGLE_USERNAME KAGGLE_KEY KAGGLE_API_TOKEN

# CONFIG, SESSION, GPU have no defaults: pass on command line, e.g. make up SESSION=ner GPU=T4

# BATCH: YAML with keys session, gpu, configs (list); sets SESSION, GPU, CONFIGS, e.g. make batch BATCH=...
ifdef BATCH
batch_get = $(shell .venv/bin/python -c "import yaml; v = yaml.safe_load(open('$(BATCH)'))['$(1)']; print(*v if isinstance(v, list) else [v])")
SESSION := $(call batch_get,session)
GPU := $(call batch_get,gpu)
CONFIGS := $(call batch_get,configs)
endif
# Seconds `colab exec` waits for remote command
TIMEOUT ?= 86400
REMOTE := /content/hw3
# Per session, so runs on several VMs at once do not share files
TMP := /tmp/hw3-$(SESSION)
# Run outputs tarball on VM
OUT := /content/out-$(SESSION).tgz

# $(1): variable names that must be set, checked before recipe runs
require = $(foreach v,$(1),$(if $($(v)),,$(error $(v) required, pass $(v)=... on command line)))

# $(1): shell command run on VM inside $(REMOTE); creates it, `data` runs before any `push`
remote = printf '%s\n' "!mkdir -p $(REMOTE) && cd $(REMOTE) && $(1)" | colab exec -s $(SESSION) --timeout $(TIMEOUT) \
	$(foreach v,$(SECRETS),$(if $($(v)),--env $(v)=$($(v))))

# $(1): local tarball, extracted into $(REMOTE). Sent in 50m parts: Colab contents API rejects ~250MB files
push = rm -f $(1).part-* && split -b 50m $(1) $(1).part- && \
	for p in $(1).part-*; do colab upload -s $(SESSION) $$p /content/$$(basename $$p) || exit 1; done && \
	printf '%s\n' "!cat /content/$(notdir $(1)).part-* > /content/$(notdir $(1)) && rm /content/$(notdir $(1)).part-* && mkdir -p $(REMOTE) && tar xzf /content/$(notdir $(1)) -C $(REMOTE)" | colab exec -s $(SESSION)

# VM side: run outputs into $(OUT) parts, part list in $(OUT).parts. Run after `rm -f $(OUT)*`
# Trainer checkpoints skipped: best model already saved in run folder
stage = mkdir -p data/processed data/wandb data/submissions && \
	tar czf $(OUT) --exclude='checkpoint-*' data/processed data/wandb data/submissions && split -b 50m $(OUT) $(OUT).part- && rm $(OUT) && ls $(OUT).part-* > $(OUT).parts

# Local side: downloads staged parts, extracts into data/. Fails if nothing staged.
# `colab exec` exits 0 even when remote command fails, so missing $(OUT).parts is failure signal
# Local part names keep VM names: $(notdir $(OUT)).part-*
fetch = rm -f $(TMP)/$(notdir $(OUT))* && mkdir -p $(TMP) && colab download -s $(SESSION) $(OUT).parts $(TMP)/$(notdir $(OUT)).parts && \
	for p in $$(cat $(TMP)/$(notdir $(OUT)).parts); do colab download -s $(SESSION) $$p $(TMP)/$$(basename $$p) || exit 1; done && \
	cat $(TMP)/$(notdir $(OUT)).part-* | tar xzf - && rm $(TMP)/$(notdir $(OUT)).part-*

COMPETITION := kse-ua-location-extraction-2026
KSE := data/$(COMPETITION)
KSE_FILES := uk_geo_dataset.csv ru_geo_dataset.csv test.csv
# $(1): competition file into $(KSE); Kaggle may serve it zipped
kaggle_get = kaggle competitions download -q -c $(COMPETITION) -f $(1) -p $(KSE) && \
	(test ! -f $(KSE)/$(1).zip || (unzip -oq $(KSE)/$(1).zip -d $(KSE) && rm $(KSE)/$(1).zip))
FASTTEXT_URL := https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.uk.300.vec.gz

.PHONY: setup up down install data fasttext sync train batch pack

# New VM ready for train: up, install, data, fasttext, in this order (also under make -j)
setup:
	$(call require,SESSION GPU)
	$(MAKE) up
	$(MAKE) install
	$(MAKE) data
	$(MAKE) fasttext

up:
	$(call require,SESSION GPU)
	colab new -s $(SESSION) --gpu $(GPU)

down:
	$(call require,SESSION)
	colab stop -s $(SESSION)

# Colab python not 3.14, so no pinned pyproject install
install:
	$(call require,SESSION)
	colab install -s $(SESSION) transformers accelerate seqeval evaluate wandb pyyaml kaggle

# Datasets downloaded on VM from Kaggle, once per VM. Needs Kaggle secrets in .env
data:
	$(call require,SESSION)
	$(call remote,mkdir -p $(KSE) && $(foreach f,$(KSE_FILES),$(call kaggle_get,$(f)) &&) true)

# fastText cc.uk.300 vectors (~1.3 GB), downloaded on VM, once per VM; FASTTEXT_PATH
fasttext:
	$(call require,SESSION)
	$(call remote,mkdir -p data/fasttext && wget -q -nc -P data/fasttext $(FASTTEXT_URL))

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

# New VM, CONFIGS trained one after another, outputs fetched once, VM stopped.
# Training failure skips later configs but stages and fetches finished ones; fetch failure keeps VM up for `make pack`.
# Fails at end if any config left no submission newer than batch start. E.g. make batch BATCH=configs/<slug>/batches/<name>.yaml
batch:
	$(call require,SESSION GPU CONFIGS)
	mkdir -p $(TMP) && touch $(TMP)/batch.start
	$(MAKE) setup
	$(MAKE) sync
	$(call remote,rm -f $(OUT)* && ($(foreach c,$(CONFIGS),PYTHONPATH=src python scripts/train_ner.py $(c) &&) true) ; $(stage))
	$(fetch)
	$(MAKE) down
	@missing=0; for c in $(CONFIGS); do n=$$(sed -n 's/^name: *//p' $$c); \
		test data/submissions/$$n.csv -nt $(TMP)/batch.start || { echo "no new submission: $$c"; missing=1; }; done; exit $$missing

# Fetches VM run outputs (data/processed, data/wandb) into local data/
pack:
	$(call require,SESSION)
	$(call remote,rm -f $(OUT)* && $(stage))
	$(fetch)
