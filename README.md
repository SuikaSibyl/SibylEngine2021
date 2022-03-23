# SIByL Engine 2022

Welcome! In August 2022, I get my first try of writing a toy game engine, that is  *SIByL Engine (2021)*. It is really interesting to build such a system and I am kind of like it. However, I am keep going on  my trip along Compute Graphics and soon find it laborious to keep on developing the 2021 version.

I guess the main reasons initially come from the design of multi-backends and the OpenGL-oriented interfaces. To be specific, I tried to support both OpenGL & DirectX12 and try to make the abstract interface somewhat closer to OpenGL ones. It is not that bad until lots of shader codes come in, which means I somewhat need a coherent supporting for both GLSL & HLSL and that is really tedious. I guess there are great works like SLANG or simply SPIRV-Cross could solve the problem, but beyond shader codes I also need to support each Render Interface feature twice. I am also unsatisfied with some design of 2021 version caused by naive design out of unfamiliarity of the game engine system.

It is time to embrace a 2022 version, I guess. In SIByL Engine, I will try out some new ideas:

- C++ 20 features, especially Module.
- Vulkan backend (Multi-backend structure is retained, but no recent plan for supporting another API)

## 1. System

The system is based on **Visual Studio 2022**, **Windows 11**, with **UTF-8 print enabled**.

The project is primarily arranged by **module** (C++ 20 new feature), and it is poorly supported by IntelliSense  now. I wish a better support later.

Codes are divided into three parts: Runtime, Application, Editor.

- **Runtime** is the core part of the engine, including all the codes need for an application.
- **Editor** is now entirely separated from Runtime, supporting GUI & Editor parts which would be dismissed in a Release Version
- **Application** is the part for actual *game-play* logics, it is also the only module built as an executable.

## 2. Runtime

**Core** is namely a core part of the engine. The main contribution is defining the structure of an application and also some utilities. An application is consists of several layers, a simple version could be: Window Layer - RHI Layer - ImGui Layer. Each layer is an unit of updating & event handling.

**RHI** is temporarily a light-weight wrap of Vulkan. Multi-backends structure is retained, but I have no recent plan for supporting another API.

**RDG** is an ad-hoc implementation of "Render Graph". **No fully support for all the features** mentioned by Frost or later works. The RDG is designed for static pipeline, resource management and automatic command submit & barrier generate. More details in "/doc/ad-hoc RenderGraph.md".

**GFX** is a set of commonly used graphics things, including GFXLayer, Postprocessing (as RDG proxy unit).

## References

### Library Based

- **GLFW** | zlib License [An OpenGL library | GLFW](https://www.glfw.org/)
- **GLM** | Modified MIT License [g-truc/glm: OpenGL Mathematics (GLM)](https://github.com/g-truc/glm)
- **Dear ImGui** | MIT License [ocornut/imgui: Dear ImGui: Bloat-free Graphical User interface for C++ with minimal dependencies](https://github.com/ocornut/imgui)
- **Entt** | MIT License [skypjack/entt: Gaming meets modern C++ - a fast and reliable entity component system (ECS) and much more](https://github.com/skypjack/entt)
- **BLAKE2b** | CC0 1.0 Universal / OpenSSL license / Apache 2.0  [BLAKE2 — fast secure hashing](https://github.com/BLAKE2/)
- **spdlog**
- **stb**
- **Slang** | MIT License [shader-slang/slang: Making it easier to work with shaders](https://github.com/shader-slang/slang)
- **vulkan**
- **yaml**

### Algorithm Referenced

- ...

### Tutorials Referenced (also recommend!)

- Engine Tutorial Hazel : [TheCherno/Hazel: Hazel Engine (github.com)](https://github.com/TheCherno/Hazel)

