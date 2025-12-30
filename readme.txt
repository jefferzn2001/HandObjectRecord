Example commands (using test_wei_02 parent directory):

1. Record data (creates subdirectories with format V####PERSON####SEQ########):
   python record.py --data_path data/test_wei_02 --person 1 --version 1

2. Extract SAM features (slow, run once, processes highest numbered subdirectory):
   python sam_extract_features.py --data_path data/test_wei_02

3. Fast blue detection and mask generation (fast, can run multiple times, processes highest numbered subdirectory):
   python sam_detect_blue.py --data_path data/test_wei_02

Notes:
- Recording requires both --person ID (1-9999) and --version ID (1-9999)
- Directory naming format: V####PERSON####SEQ########
  * V: 4-digit version ID (e.g., 0001, 0002) - prevents conflicts when code is modified
  * PERSON: 4-digit person ID (e.g., 0001, 0002)
  * SEQ: 8-digit sequence number (e.g., 00000001, 00000002)
  * Example: V0001PERSON0001SEQ00000001, V0001PERSON0001SEQ00000002, V0002PERSON0001SEQ00000001
- When person and version match, SEQ numbers increment from the maximum existing number
- Processing scripts automatically use the highest numbered subdirectory for the given parent path
- Each person's sequences are managed independently per version
