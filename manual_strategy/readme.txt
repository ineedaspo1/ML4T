
# Instructions to generate my results

1. 'util.py' and folder 'data' must be copied to same dir where we locate all .py codes, and create 'plots' folder.
   cp ../util.py .
   ln -s ../data # making sym link to avoid 'PYTHONPATH=../:.'
   mkdir plots
   
2. Run indicators.py. Generates 3 figs and saves inside 'plots' dir
       python2.7 indicators.py
   
3. Run manual strategy code. Generates 2 figs and writes stat results to screen
       python2.7 ManualStrategy.py 

4. Run best possible strategy code. Generates 1 fig and writes stat results to screen
       python2.7 BetPossibleStrategy.py
