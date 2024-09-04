# raman_prediction
predict mineral chemical composition from the Raman signal

# Data Source: RRUFF database (https://rruff.info/)
- Chemistry data: https://rruff.info/zipped_data_files/chemistry/Microprobe_Data.zip
- Raman data: https://rruff.info/zipped_data_files/raman/
  - LR-Raman (do not use)
  - excellent_oriented (https://rruff.info/zipped_data_files/raman/excellent_oriented.zip)
  - excellent_unoriented (https://rruff.info/zipped_data_files/raman/excellent_unoriented.zip)
  - fair_oriented (https://rruff.info/zipped_data_files/raman/fair_oriented.zip)
  - fair_unoriented (https://rruff.info/zipped_data_files/raman/fair_unoriented.zip)
  - ignore_unoriented (do not use)
  - poor_oriented (do not use)
  - poor_unoriented (do not use)
  - unrated_oriented (do not use)
  - unrated_unoriented (do not use)
 - consider using all non LR-Raman data with a method for excluding high noise spectrum

# Running with Anaconda
- conda create -n "env_name" python=3.11
- conda activate env_name
- pip install -r requirements.txt
- python run_me_first.py (extracting RRUFF Raman and Chemistry data)
- run other files (e.g., "python extract_chemistry.py")
- conda deactivate

# .gitignore
- does not track "chemistry_data" directory containing Microprobe_Data
- does not track "raman_data" directory containing all extracted Raman spectrum
