# Check SLURM Job Status
1. Run `squeue -u $USER` to list all active jobs
2. For any FAILED or COMPLETED jobs, check logs in the project's slurm_logs directory (NOT ~/)
3. Report: job ID, status, runtime, and any errors found in logs
4. If jobs failed, check for OOM, disk quota, or missing dependency errors specifically
