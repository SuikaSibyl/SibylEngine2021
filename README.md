# SibylEngine

Personal game engine, which I would work on for a long time.



**Post Scripts**

*The Design and Basic Codes are heavily based on The Cherno's Hazel 2D Engine.
Thanks for his great tutorials, which really taught me to learn a lot.*



## Modules

- **Event System**
  - Uniform API for independent realizations of Windows HWND & GLFW
- **Graphic API**
  - Uniform API for independent realizations of DX12 & OpenGL
  - Computer Shader is supported
  - (Shading language is not uniformed yet...)
- **Graphic System**
  - Material
  - Camera
  - Light (directional, point)
  - Post Processing
- **Graphic Pipeline**

- **Editor**
  - Based on Dear ImGUI
  - Scene / Material / Component Editing
- **Physics**
  - Self collision detection (based on CUDA)



## Algorithms

### Graphic

- **Temporal Anti Aliasing** - [An Excursion in Temporal Supersampling (nvidia.cn)](https://developer.download.nvidia.cn/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf)



### Parallel Computing

- **CUDA BVH construction** - *Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees (2012)*

- **Independent Traversal Collision Detection** - [Thinking Parallel, Part II: Tree Traversal on the GPU | NVIDIA Developer Blog](https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/)



## References

- **Engine Tutorial Hazel** (very recommend):  [TheCherno/Hazel: Hazel Engine (github.com)](https://github.com/TheCherno/Hazel)

