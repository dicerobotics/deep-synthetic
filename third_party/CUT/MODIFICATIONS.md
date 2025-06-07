This file documents all changes made to the original repository (https://github.com/taesungp/contrastive-unpaired-translation) within the third_party/CUT directory. These changes diverge from the upstream version and are tracked here for transparency, reproducibility, and ease of merging or re-syncing in the future.

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
**Commit**:  
**Author**: [Arshad MA]   
**Description**: Removed .git .gitignore .gitmodules and Added `__init__.py`.    
**Files/Sections Changed**: `.git/`, `.gitmodule`, `.gitignore`, `__init__.py`.   
**Reason**: Our project uses this repo as a package.  

**Date**: [2025-06-05]  
**Commit**:  
**Author**: [Arshad MA]   
**Description**: Included pre-trained models    
**Files/Sections Changed**: checkpoints/.   
**Reason**: We decided to fine-tune the pretrained models instead of traning from scratch.  

**Date**: [2025-06-07]  
**Commit**:  
**Author**: [Arshad MA]   
**Description**: Included scripts to run models    
**Files/Sections Changed**: run_mwir_real2cut.sh, run_mwir_sym2cut.sh   
**Reason**: Easy reference to run the CUT model.  