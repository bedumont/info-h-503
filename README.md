# Welcome to INFO-H-503

- **INFOH503-CUDA GPGPU programming-*.pptx**:  Exercises related theory slides.
- The exercises are on numerated folder. You can do the exercises at the rhythm you want, but, I will answer only questions on the current subject of the day.
- Those exercises are very practical and here to prepare you for the project.

# Project
- The project is big, so please do it in time.
- In the project you will be evaluated on the acceleration you obtained and which cuda concepts you used.

# Exercises

## Seance 01
- Visual Studio Installation for everyone
    - If you have a laptop, install VS2017
- CUDA installation for everyone
    - If you have a laptop, install CUDA for your GPU and NSIGHT
- Slides introduction : Read the slides and ask questions
- Creating a first Visual Studio CUDA Project (please use VS2017)
- NSIGHT for viewing the GPU Specs
    - Look at the document NSIGHT_START.pdf to see how to use NSIGHT
	- If NSIGHT do not works in your computer, you have to change the solution to be a VS2015 solution
	- You will also need to install VS2015 dependencies with the visual studio installer, ask me.
- Analysis of the default generated kernel by Visual Studio
    - Start a new CUDA project and try to understand the code
- Analysis of the results of the **gridBlockThreads** kernel
    - Open The 00-* folder, look and execute the code.

## Seance 02
- **VectorAdd** Kernel analysis and experimentations
- **ScalarProduct** Try to do yourself a new kernel END-TO-END for a scalar product.
    - Use shared memory (sync)
    - Use multiple blocks (sync)
    - optional: Dyadic Sum (reduction)

- **MatrixMultiplication** Write your code for a matrix multiplication
    - CPU version
    - GPU version (simple)
    - GPU version with shared memory

## Seance 03
- Finish Seance 02
- **Transpose**: Understand the transpose code
- Add image support for the transpose and try it
    - Look at the folder **ImagesHelper**
- Debugging:
    - Learn how to use printf
    - Learn how to use the CUDA debugger (breakpoints, stepping, etc.)

- Analysis with NSIGHT: **Application tracing** and **Profile CUDA Application**
    - **VectorAdd**
    - **Transpose**

## Seance 04

- **Convolution** Write a kernel for a stencil 1D
- **Convolution** Write a kernel for a convolution 2D
    - You have to play with images! Look at the folder **ImagesHelper**

## Seance 05 & more

- **Questions**
- **Project**
