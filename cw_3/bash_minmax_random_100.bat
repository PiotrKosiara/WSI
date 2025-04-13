@echo off
echo Symulacja: minmax vs random > wyniki_minmax_random.txt
for /L %%i in (1,1,100) do (
    echo Gra %%i >> wyniki_minmax_random.txt
    py ttt.py minmax random >> wyniki_minmax_random.txt
    echo ------------------------- >> wyniki_minmax_random.txt
)