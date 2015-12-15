# make file for seg doc

TARGET=dropc

all: ${TARGET}.dvi ${TARGET}.pdf 

${TARGET}.dvi: ${TARGET}.tex 
	latex ${TARGET}
	bibtex ${TARGET}
	#bibtex refs
	latex ${TARGET}
	latex ${TARGET}

${TARGET}.pdf: ${TARGET}.dvi
	dvipdf ${TARGET}.dvi

clean:
	rm -f ${TARGET}.aux
	rm -f ${TARGET}.log
	rm -f ${TARGET}.dvi
	rm -f ${TARGET}.pdf
	rm -f ${TARGET}.out
