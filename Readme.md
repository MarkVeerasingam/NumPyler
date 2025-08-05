# Matrix llvmlite Project

This project is designed to help you write LLVM IR and use **llvmlite** as a JIT engine.

You will be building a hardware acclerated Python compiler for matrix operations. (i.e. the computational backbone of AI/ML)

It serves as a first step towards hardware acceleration by working directly with low-level memory references based on the NumPy array structure.  
(Inspired by Stephen Diehl’s post on MLIR memory: [https://www.stephendiehl.com/posts/mlir_memory/](https://www.stephendiehl.com/posts/mlir_memory/))

Please Read.

This will be a big step up from the LLVM we have been writing but I have no doubt you guys are up to the challenge :) 

to run the python file, execute the run.py - main project entry point.
---

## Goals

- Get hands-on experience with Python and LLVM IR.
- Create new LLVM files such as:
  - `vector_add.ll`
  - `vector_sub.ll`
  - `vector_div.ll`
- Update your engine.py in accordance 
    - You might realise the project structure is not to your liking and it gets messy in one file. I encourage you to seperate out files!
- If you can get to the above step, you have done outstanding work!, If you want to extend functionality, somde ideas below...
    - numpy_array_layout.py is 1d, you would need to expand it be 2d and 3d arrays/matrices
    - Right now the program is hardcoded to the vector vallues assigned in engine.py, you could extend to take user inputs?
---

## Why this project?

This is intended as an independent and exploratory project that can serve as poster material for EuroLLVM submissions or inspire early research topics such as:

- Compiler construction
- Hardware acceleration
- Designing your own programming language

It’s a great way to step into LLVM, tensors, and matrix computations, and could form the basis of an FYP

---

## Further thoughts if you want to expand 

This project is flexible enough for you to extend individually on the side, reinforcing the importance of thinking about your FYP or research topic early. You can develop this alongside your main internship work. 
This project can evolve into it's own programming language or move to a more AI Acceleration approach.

You'll soon find that LLVM for tensor and matrix operations can get complicated and a pain in the ass. If so, I highly recommend looking into **MLIR** - which inspired this project - for more elegant and scalable solutions:  
- [https://www.stephendiehl.com/posts/mlir_introduction/](https://www.stephendiehl.com/posts/mlir_introduction/)

- [https://www.stephendiehl.com/posts/mlir_memory/](https://www.stephendiehl.com/posts/mlir_memory/)

Feel free to explore MLIR if interested, but for now, this project should keep you well engaged while I’m away.

If won't be of major help for you, for this project but if you do go down that route, it is something I personally have done before
- [https://github.com/MarkVeerasingam/matrices-mlir/tree/main](https://github.com/MarkVeerasingam/matrices-mlir/tree/main)
# NumPyler
# NumPyler
# NumPyler
