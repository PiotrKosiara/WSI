@echo off
echo Symulacja: random vs random > wyniki_random_random.txt
for /L %%i in (1,1,100) do (
    echo Gra %%i >> wyniki_random_random.txt
    py ttt.py random random >> wyniki_random_random.txt
    echo ------------------------- >> wyniki_random_random.txt
)
