# MyCPD
My change point detection


## Data description table 
  |              | What  | How | Volume | 
  |:----:        |  :----:    |:----:    |:----:        |
  | agotsshiftmean1-5.mat | 1 dim shift mean time series | agots         | 
  | agotsshiftvar1-5.mat | 2 dim variance shift time series   | agots        |
  | extreme1-5.mat | 1 dim time series contains extreme noise  | agots  |
  | shiftcorr1-5 | 2 dim shift correlation time series   | rupture   | 
  | shiftmean1-5.py | 1 dim shift mean time series   | rupture   | 
  | shiftlinear1-5.py | 1 dim linear model shift   | rupture  |
  | singledimshiftfreq1-5.py | 1 dim time series with shift frequence   | rupture  |

- agots: https://github.com/KDD-OpenSource/agots
- rupture: https://github.com/deepcharles/ruptures
- ？？？need to undersatand the difference of how to generate the baseline time series between agots and ruptures, better resutls is gained on agots.

