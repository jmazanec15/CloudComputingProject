# CloudComputingProject
Final Project for Cloud Computing. This is a simple model implementation of AlphaZero. It uses HTCondor to parallelize the process of the neural net playing games vs itself.

To run:
- Make sure you are runnning this on machine `crcfe02.crc.nd.edu`
- Make sure that the most up to date cctools has been installed on your machine. If not, run:
```
cd $HOME
wget http://ccl.cse.nd.edu/software/files/cctools-7.0.4-source.tar.gz
tar xvzf cctools-7.0.4-source.tar.gz
cd cctools-7.0.4-source
./configure --prefix $HOME/cctools --tcp-low-port 9000
make
make install
cd $HOME
```
- Make sure that you have pip installed and add `export PATH=/afs/crc.nd.edu/user/$USERNAME[0]$/$USERNAME$/.local/bin:$PATH` to your PATH
- Use `python -m pip install ___ --user` to install necessary libraries such as `numpy`, `keras`, and `tensorflow`
- Run the commands found in the file `Path.txt` in the terminal
- Use the command `condor_submit_workers crcfe02.crc.nd.edu PORT NUM_WORKERS` to submit `NUM_WORKERS` workers to HTCondor
- Ensure that the `PORT` matches the port number specified in the `workers()` function in `main.py`
- Set parameters in `main.py`
- Run `./main.py`


** Need to have script.h in a directory called cloud I guess.