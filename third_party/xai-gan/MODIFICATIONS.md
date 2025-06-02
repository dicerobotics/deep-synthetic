This file documents all changes made to the original repository (https://github.com/vin-nag/XAI_GAN.git) within the third_party/xai-gan directory. These changes diverge from the upstream version and are tracked here for transparency, reproducibility, and ease of merging or re-syncing in the future.

## **Notes**
- Do not pull directly from the upstream repository without reviewing these changes.
- Any planned updates from upstream should be merged manually with conflict resolution and new entries added here.

## **Format**  
Each entry follows:

    - Date
    - Commit (optional)
    - Author
    - Description
    - Files/Sections Changed
    - Reason

## **Modification Log**

**Date**: [2025-05-29]  
**Commit (optional)**:  
**Author**: [Arshad MA]  
**Description**: Removed `.git` `.gitignore` `.gitmodules` and Added `__init__.py `   
**Files/Sections Changed**: `.git/`, `.gitignore`, `__init__.py`    
**Reason**: Our project uses this repo as a python package.  


**Date**: [2025-06-02]  
**Commit (optional)**:     
**Author**: [Arshad MA]  
**Description**: Commented/Uncommented some portion of `experiment_enum.py`  
**Files/Sections Changed**: `experiment_enum.py`  
**Reason**: Modified to include/exclude experiments as per need. Further changes to include/exclude experiments won't be logged in this file `MODIFICATIONS.md`.  


**Date**: [2025-06-02]  
**Commit**: 5e07ba06702ef000b0eb41b0d8731eaeebd4a771      
**Author**: [Arshad MA]  
**Description**: Changed data storage directory  
**Files/Sections Changed**: `src/get_data.py`  
**Reason**: To make data handling consistent with the project.  

