main.pdf: $(wildcard CSIM_Proposal/sections/*.tex)
	![[ "$(uname)" == "Darwin" ]] && cd CSIM_Proposal && pdflatex main.tex