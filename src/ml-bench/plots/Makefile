TRANSFORMER_RESULTS=../volta_transformer/results/
CONV_RESULTS=../volta_conv2d/results/

all: attention mlp resnet vgg resnet-vgg-common resnetEnd2End gptend2End

gptstreamk: plotGPTStreamK.py mlp_results-gpt3.csv attention_results-gpt3.csv
	python3 plotGPTStreamK.py mlp_results-gpt3.csv attention_results-gpt3.csv $@.pdf

mlp: mlp-gpt3 mlp-llama 

attention: attention-gpt3 attention-llama

attention-gpt3: plotGPT.py $(TRANSFORMER_RESULTS)/attention-gpt3.csv
	python3 plotGPT.py $(TRANSFORMER_RESULTS)/attention-gpt3.csv Figure-6-b.pdf

mlp-gpt3: plotGPT.py $(TRANSFORMER_RESULTS)/mlp-gpt3.csv
	python3 plotGPT.py $(TRANSFORMER_RESULTS)/mlp-gpt3.csv Figure-6-a.pdf

attention-llama: plotGPT.py $(TRANSFORMER_RESULTS)/attention-llama.csv
	python3 plotGPT.py $(TRANSFORMER_RESULTS)/attention-llama.csv Figure-6-d.pdf

mlp-llama: plotGPT.py $(TRANSFORMER_RESULTS)/mlp-llama.csv
	python3 plotGPT.py $(TRANSFORMER_RESULTS)/mlp-llama.csv Figure-6-c.pdf

resnet-vgg-common: plotResnet.py $(CONV_RESULTS)/resnet.csv
	python3 plotResnet.py $(CONV_RESULTS)/resnet.csv Figure-7-a.pdf

resnet: plotResnet.py $(CONV_RESULTS)/resnet.csv
	python3 plotResnet.py $(CONV_RESULTS)/resnet.csv Figure-7-b.pdf

vgg: plotResnet.py $(CONV_RESULTS)/vgg.csv
	python3 plotResnet.py $(CONV_RESULTS)/vgg.csv Figure-7-c.pdf

resnetEnd2End: plotResnet.py plotResnetEnd2End.py $(CONV_RESULTS)/resnet.csv $(CONV_RESULTS)/vgg.csv
	python3 plotResnetEnd2End.py $(CONV_RESULTS)/resnet.csv Figure-8-b.pdf

gptend2End: plotGPT.py plotGPTEnd2End.py $(TRANSFORMER_RESULTS)/mlp-gpt3.csv $(TRANSFORMER_RESULTS)/attention-gpt3.csv $(TRANSFORMER_RESULTS)/mlp-llama.csv $(TRANSFORMER_RESULTS)/attention-llama.csv
	python3 plotGPTEnd2End.py $(TRANSFORMER_RESULTS)/mlp-gpt3.csv $(TRANSFORMER_RESULTS)/attention-gpt3.csv $(TRANSFORMER_RESULTS)/mlp-llama.csv $(TRANSFORMER_RESULTS)/attention-llama.csv $(TRANSFORMER_RESULTS)/allreduce_times-12288.csv $(TRANSFORMER_RESULTS)/allreduce_times-8192.csv Figure-8-a.pdf

clean:
	rm -rf *.pdf
