Example commands (using test_wei_02 directory):

1. Record data:
python record.py --name test_wei_02

2. Extract SAM features (slow, run once):
python sam_extract_features.py --data_path data/test_wei_02

3. Fast blue detection and mask generation (fast, can run multiple times):
python sam_detect_blue.py --data_path data/test_wei_02
