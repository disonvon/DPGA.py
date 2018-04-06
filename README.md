This repository contains the code that produces the numeric section in On the Use of TensorFlow Computation Graphs in combination with Distributed Optimization to Solve Large-Scale Convex Problems 


=== Dependencies ===

Python 3.5

mpi4py (Version == 3.0)(Windows):
The mpi4py documentation and installation instructions can be found at:

http://mpi4py.scipy.org/

TensorFlow-gpu (Windows):
The TensorFlow-gpu documentation and installation instructions can be found at:

https://www.tensorflow.org/install/install_windows

Some other libraries/packages needed are: NumPy, scipy


=== How to generate multiple processes on a single (multi-core/cpu) host ===
Run it with 

mpirun -np N ./some-program

where the number after "-np " is the numer of parallel MPI processes to be started.


===   Set up VM (Virtual Machine) instance on Google Cloud Engine ===

Step 1: Visit Google Cloud Platform and click Compute Engine with a Google account;

Step 2: Click CREATE INSTANCE button (every new user will have $ 300 free trial for one year);

Step 3: Customize what kind of machine you need (Name, Machine Zone, #cores/cpus, memory, gpus, Boot disk (Linux, Centos, Windows) and storage size);

Step 4: Connect to your VM instance via SSH/RDP with your dynamic External IP, User ID, password;

Step 5: Set up the libraries/packages/software you need;

Step 5: Test your code in prompt command window;

PS: You can edit your instance size (#cpu/memory/gpu) whnever you need 
