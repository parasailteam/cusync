TRANSFORMER_RESULTS=../volta_transformer/results/
CONV_RESULTS=../volta_conv2d/

all: attention mlp resnet

gptstreamk: plotGPTStreamK.py mlp_results-gpt-3.csv attention_results-gpt-3.csv
	python3 plotGPTStreamK.py mlp_results-gpt-3.csv attention_results-gpt-3.csv $@.pdf

mlp: mlp-gpt-3 mlp-llama 

attention: attention-gpt-3 attention-llama

attention-gpt-3: plotGPT.py $(TRANSFORMER_RESULTS)/attention_results-gpt-3.csv
	python3 plotGPT.py $(TRANSFORMER_RESULTS)/attention_results-gpt-3.csv $@.pdf $*

mlp-gpt-3: plotGPT.py $(TRANSFORMER_RESULTS)/mlp_results-gpt-3.csv
	python3 plotGPT.py $(TRANSFORMER_RESULTS)/mlp_results-gpt-3.csv $@.pdf

attention-llama: plotGPT.py $(TRANSFORMER_RESULTS)/attention_results-llama.csv
	python3 plotGPT.py $(TRANSFORMER_RESULTS)/attention_results-llama.csv $@.pdf $*

mlp-llama: plotGPT.py $(TRANSFORMER_RESULTS)/mlp_results-llama.csv
	python3 plotGPT.py $(TRANSFORMER_RESULTS)/mlp_results-llama.csv $@.pdf

resnet-vgg-common: plotResnet.py $(CONV_RESULTS)/resnet_results.csv
	python3 plotResnet.py $(CONV_RESULTS)/resnet_results.csv $@.pdf

resnet: plotResnet.py $(CONV_RESULTS)/resnet_results.csv
	python3 plotResnet.py $(CONV_RESULTS)/resnet_results.csv $@.pdf

vgg: plotResnet.py $(CONV_RESULTS)/vgg_results.csv
	python3 plotResnet.py $(CONV_RESULTS)/vgg_results.csv $@.pdf

resnetEnd2End: plotResnet.py plotResnetEnd2End.py $(CONV_RESULTS)/resnet_results.csv $(CONV_RESULTS)/vgg_results.csv
	python3 plotResnetEnd2End.py $(CONV_RESULTS)/resnet_results.csv $@.pdf

gptend2End: plotGPT.py plotGPTEnd2End.py $(TRANSFORMER_RESULTS)/mlp_results-gpt-3.csv $(TRANSFORMER_RESULTS)/attention_results-gpt-3.csv $(TRANSFORMER_RESULTS)/mlp_results-llama.csv $(TRANSFORMER_RESULTS)/attention_results-llama.csv
	python3 plotGPTEnd2End.py $(TRANSFORMER_RESULTS)/mlp_results-gpt-3.csv $(TRANSFORMER_RESULTS)/attention_results-gpt-3.csv $(TRANSFORMER_RESULTS)/mlp_results-llama.csv $(TRANSFORMER_RESULTS)/attention_results-llama.csv $@.pdf
