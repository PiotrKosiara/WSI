@echo off
echo Symulacja: random vs minmax > wyniki_random_minmax.txt
for /L %%i in (1,1,100) do (
    echo Gra %%i >> wyniki_random_minmax.txt
    py ttt.py random minmax >> wyniki_random_minmax.txt
    echo ------------------------- >> wyniki_random_minmax.txt
)
