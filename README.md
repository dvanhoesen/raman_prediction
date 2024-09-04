# raman_prediction
predict mineral chemical composition from the Raman signal

# Data Source: RRUFF database ()
- Chemistry data: https://rruff.info/zipped_data_files/chemistry/Microprobe_Data.zip
- Raman data: https://rruff.info/zipped_data_files/raman/

# Running with Anaconda
- conda create -n "env_name" python=3.11
- conda activate env_name
- pip install -r requirements.txt
- run files (e.g., "python extract_chemistry.py")
- conda deactivate

# .gitignore
- does not track "Microprobe_Data" directory
- add raman data directory when available
