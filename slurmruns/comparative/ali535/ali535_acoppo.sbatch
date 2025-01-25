#! /bin/bash
##SBATCH --account=priority-johnsheppard      	# specify the account to use if using a priority partition
#SBATCH --partition=nextgen               	# queue partition to run the job in
#SBATCH --cpus-per-task=10              		# number of cores to allocate
#SBATCH --nodes=1                       	# number of nodes to allocate
#SBATCH --mem=32G                        	# amount of Memory allocated
#SBATCH --time=1-00:00:00               	# maximum job run time in days-hours:minutes:secconds
#SBATCH --job-name=ali535_acoppo	        # job name
#SBATCH --output=ali535_acoppo.out         	# standard output from job
#SBATCH --error=ali535_acoppo.err	         	# standard error from job
#SBATCH --mail-user=willjardee@gmail.com     	# enter your email to recieve email notifications
#SBATCH --mail-type=ALL

source ~/.bashrc
module load Anaconda3
source activate $HOME/sci
cd $HOME/dev/pgaco/src/pgaco/experiments/comparative/ali535

echo "running acoppo"
python ACOPPO.py

