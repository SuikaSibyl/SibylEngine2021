# SIByL Engine

Personal game engine, which I plan to work on for a long time in coming years.

The engine is currently far from real production, and mass architecture change might happen in the future.



***Post Scripts***

*The Design and Basic Codes are heavily based on The Cherno's Hazel 2D Engine.
Thanks for his great tutorials, which really taught me to learn a lot.*

**Demo**
![Hair Demo](https://i.loli.net/2021/11/24/I2h6FwdKbmUZeE4.png)

## Plans
**Recent Update**
- [x] Post Process Pipe: SSAO
- [x] Shadow System ( directed )

**Recent Plans**

- [ ] New Shader System
- [ ] New graphic abstract layer

**Future Plans**

- [ ] Path Tracer
- [ ] PBR probe system
- [ ] Other GI system...



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
  - ScriptableRenderPipeline
    - Original designed API
    - Implementation of basic pipes:
      - Draw Pipes
        - EarlyZ
        - LitForward
      - Post Processing Pipes
        - ACES
        - Bloom
        - FXAA
        - Sharpen
        - TAA
        - Vignette
- **ECS System**
  - Based on *entt* : https://github.com/skypjack/entt
  - Implemented Components:
    - Tag
    - Camera
    - Transform
    - Light
    - Self-Collision Detector
    - Mesh Filter
    - Mesh Renderer (multi-pass supported!)
    - Sprite Renderer
- **Editor**
  - Based on *Dear ImGUI*: [ocornut/imgui: Dear ImGui](https://github.com/ocornut/imgui)
  - Scene / Material / Component Editing
- **Physics**
  - Self Collision Detection (based on CUDA)
    - Performance: 1.2 million triangles flag testcase 0.06s (on RTX 2070S)



## Algorithms

### Graphic

- **Temporal Anti Aliasing** - [An Excursion in Temporal Supersampling (nvidia.cn)](https://developer.download.nvidia.cn/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf)



### Parallel Computing

- **CUDA LBVH construction** - *Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees (2012)*

- **Independent Traversal Collision Detection** - [Thinking Parallel, Part II: Tree Traversal on the GPU | NVIDIA Developer Blog](https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/)



## References

- **Engine Tutorial Hazel** (very recommend):  [TheCherno/Hazel: Hazel Engine (github.com)](https://github.com/TheCherno/Hazel)

