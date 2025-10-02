# .latexmkrc
$pdf_mode = 1;                 # generate PDF directly
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
$lualatex = 'lualatex -interaction=nonstopmode -synctex=1 %O %S';
$xelatex  = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';

$bibtex = 'bibtex %O %S';
$biber  = 'biber %O %S';

# Clean up build artifacts
@generated_exts = qw(aux bbl blg fdb_latexmk fls log out synctex.gz toc lof lot run.xml);
