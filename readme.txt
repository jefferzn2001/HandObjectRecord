Example commands (using test_wei_02 parent directory):

1. Record data (creates subdirectories with format V####PERSON####SEQ########):
   python record.py --data_path data/test_wei_02 --person 1 --version 1

2. Extract SAM features (slow, run once, processes all matching subdirectories):
   python sam_extract_features.py --data_path data/test_wei_02

3. Fast blue detection and mask generation (fast, can run multiple times, processes all matching subdirectories):
   python sam_detect_blue.py --data_path data/test_wei_02

4. Run FoundationPose for object pose estimation (processes all matching subdirectories by default):
   cd FoundationPose
   conda activate foundationpose
   python FP.py --data_path data/test_wei_02 --camera zed --mesh cup
   
   # Process only highest numbered subdirectory:
   python FP.py --data_path data/test_wei_02 --camera zed --mesh cup --latest
   
   # Process specific person and version:
   python FP.py --data_path data/test_wei_02 --person 1 --version 1 --camera zed --mesh cup

5. Run HaMeR for hand keypoints (bash script to loop through all directories):
   # Example bash script to process all directories:
   #!/bin/bash
   DATA_PATH="/mnt/aloque_scratch/jefferzn/ManipDepthRecord/data/test_wei_02"
   HAMER_DIR="/mnt/aloque_scratch/jefferzn/ManipDepthRecord/record/hamer"
   CONDA_ENV="/mnt/aloque_scratch/jefferzn/miniconda3/envs/hamer"
   
   # Activate conda environment
   source ${CONDA_ENV}/bin/activate
   cd ${HAMER_DIR}
   conda activate hammer
   # Loop through all V####PERSON####SEQ######## directories
   for seq_dir in ${DATA_PATH}/V*PERSON*SEQ*; do
       if [ -d "$seq_dir" ] && [[ ! "$seq_dir" =~ _TEMP$ ]]; then
           echo "Processing: $(basename $seq_dir)"
           
           # Run preprocess.py
           ${CONDA_ENV}/bin/python ${HAMER_DIR}/preprocess.py --data_path $seq_dir
           
           # Run save_video_processed.py
           ${CONDA_ENV}/bin/python ${HAMER_DIR}/save_video_processed.py --data_path $seq_dir
       fi
   done
   
   # For SLURM cluster (example):
   srun --partition=aloque-compute --qos=al-high-2gpu --job-name=gdev \
        --nodes=1 --gres=gpu:l40s:1 --time=24:00:00 --pty bash
   # Then run the loop script above

Notes:
- Recording requires both --person ID (1-9999) and --version ID (1-9999)
- Directory naming format: V####PERSON####SEQ########
  * V: 4-digit version ID (e.g., 0001, 0002) - prevents conflicts when code is modified
  * PERSON: 4-digit person ID (e.g., 0001, 0002)
  * SEQ: 8-digit sequence number (e.g., 00000001, 00000002)
  * Example: V0001PERSON0001SEQ00000001, V0001PERSON0001SEQ00000002, V0002PERSON0001SEQ00000001
- When person and version match, SEQ numbers increment from the maximum existing number
- Processing scripts (sam_extract_features, sam_detect_blue, FoundationPose) process all matching subdirectories by default
- FoundationPose: use --latest to process only highest numbered subdirectory
- HaMeR: use bash script to loop through all directories (see step 5)
- Each person's sequences are managed independently per version
