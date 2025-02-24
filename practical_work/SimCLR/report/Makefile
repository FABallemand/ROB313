FILENAME=report

all:
	latexmk -pdf -bibtex -pdflatex="pdflatex -interaction=nonstopmode -shell-escape" -use-make -f $(FILENAME).tex
	
.PHONY: clean
clean:
	latexmk -CA
