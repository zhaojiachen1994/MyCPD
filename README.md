# MyCPD
My change point detection


## Data description table 
  |              | What  | How | Volume | loss function | update weights | 
  |:----:        |  :----:    |:----:    |:----:        |:----:    |:----:     |
  | agotsshiftmean1-5.mat | 1 dim meanshift time series | agots        | X        | X             |   X      |
  | agotsshiftvar1-5.mat | 2 dim variance shift time series   | agots        | X        | X             |   X      |
  | extreme.mat | 1 dim time series contains extreme  | tensor.backward()   | X        | X             |   X      |
  |NNwithnnmodel.py | tensor   | tensor.backward()   | nn.Sequential() | nn.MSELoss()  |   X      |
  |NNwithoptim.py | tensor   | tensor.backward()   | nn.Sequential() | nn.MSELoss()  |   optim.Adam() |
  |NNwithcustommodel.py | tensor   | tensor.backward()   | customized class | nn.MSELoss()  |   optim.Adam() |
  
/-[] need to undersatand the difference between  

