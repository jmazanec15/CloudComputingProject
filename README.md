# CloudComputingProject
Final Project for Cloud Computing. It will use AWS and other frameworks to train an AlphaZero model to play connect4.

To run:
- Make sure you are runnning this on machine `crcfe02.crc.nd.edu`
- Use `python -m install ___ --user` to install necessary libraries such as `numpy`, `keras`, and `tensorflow`
- Run the commands found in the file `Path.txt` in the terminal
- Use the command `condor_submit_workers crcfe02.crc.nd.edu PORT NUM_WORKERS` to submit `NUM_WORKERS` workers to HTCondor
- Ensure that the `PORT` matches the port number specified in the `workers()` function in `main.py`
- Set parameters in `main.py`
- Run `./main.py`
