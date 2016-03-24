# This file is configured by CMake automatically as DartConfiguration.tcl
# If you choose not to use CMake, this file may be hand configured, by
# filling in the required variables.


# Configuration directories and files
SourceDirectory: /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9/nifty_reg
BuildDirectory: /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9_linux_x64r

# Where to place the cost data store
CostDataFile: 

# Site is something like machine.domain, i.e. pragmatic.crd
Site: rothhr-Precision-T7610

# Build name is osname-revision-compiler, i.e. Linux-2.4.2-2smp-c++
BuildName: Linux-g++-4.4

# Submission information
IsCDash: TRUE
CDashVersion: 
QueryCDashVersion: 
DropSite: cmicdev.cs.ucl.ac.uk
DropLocation: /cdash/submit.php?project=NiftyReg
DropSiteUser: 
DropSitePassword: 
DropSiteMode: 
DropMethod: http
TriggerSite: 
ScpCommand: /usr/bin/scp

# Dashboard start time
NightlyStartTime: 22:00:00 GMT

# Commands for the build/test/submit cycle
ConfigureCommand: "/usr/bin/cmake" "/home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9/nifty_reg"
MakeCommand: /usr/bin/make -i
DefaultCTestConfigurationType: Release

# CVS options
# Default is "-d -P -A"
CVSCommand: /usr/bin/cvs
CVSUpdateOptions: -d -A -P

# Subversion options
SVNCommand: /usr/bin/svn
SVNOptions: 
SVNUpdateOptions: 

# Git options
GITCommand: /usr/bin/git
GITUpdateOptions: 
GITUpdateCustom: 

# Generic update command
UpdateCommand: 
UpdateOptions: 
UpdateType: 

# Compiler info
Compiler: /usr/bin/g++-4.4

# Dynamic analysis (MemCheck)
PurifyCommand: 
ValgrindCommand: 
ValgrindCommandOptions: 
MemoryCheckCommand: MEMORYCHECK_COMMAND-NOTFOUND
MemoryCheckCommandOptions: 
MemoryCheckSuppressionFile: 

# Coverage
CoverageCommand: /usr/bin/gcov
CoverageExtraFlags: -l

# Cluster commands
SlurmBatchCommand: SLURM_SBATCH_COMMAND-NOTFOUND
SlurmRunCommand: SLURM_SRUN_COMMAND-NOTFOUND

# Testing options
# TimeOut is the amount of time in seconds to wait for processes
# to complete during testing.  After TimeOut seconds, the
# process will be summarily terminated.
# Currently set to 25 minutes
TimeOut: 1500

UseLaunchers: 
CurlOptions: 
# warning, if you add new options here that have to do with submit,
# you have to update cmCTestSubmitCommand.cxx

# For CTest submissions that timeout, these options
# specify behavior for retrying the submission
CTestSubmitRetryDelay: 5
CTestSubmitRetryCount: 3
