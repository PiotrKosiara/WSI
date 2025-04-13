@echo off
echo Symulacja: minmax vs minmax > wyniki_minmax_minmax.txt
for /L %%i in (1,1,5) do (
    echo Gra %%i >> wyniki_minmax_minmax.txt
    py ttt.py minmax minmax >> wyniki_minmax_minmax.txt
    echo ------------------------- >> wyniki_minmax_minmax.txt
)