#! /bin/bash
#SBATCH --account=priority-johnsheppard      	# specify the account to use if using a priority partition
#SBATCH --partition=priority               	# queue partition to run the job in
#SBATCH --cpus-per-task=16              		# number of cores to allocate
#SBATCH --nodes=1                       	# number of nodes to allocate
#SBATCH --mem=32G                        	# amount of Memory allocated
#SBATCH --time=2-00:00:00               	# maximum job run time in days-hours:minutes:secconds
#SBATCH --job-name=tuning_ali_mmaco	        # job name
#SBATCH --output=tuning_ali_mmaco.out         	# standard output from job
#SBATCH --error=tuning_ali_mmaco.err	        # standard error from job
#SBATCH --mail-user=willjardee@gmail.com     	# enter your email to recieve email notifications
#SBATCH --mail-type=ALL 

source ~/.bashrc
module load Anaconda3
source activate $HOME/sci
cd $HOME/dev/pgaco/src/pgaco/tuning

echo "running minmaxaco"
python tuning_minmaxaco.py

