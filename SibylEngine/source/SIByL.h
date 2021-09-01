#pragma once

// For Bind By Application
#include "SIByLpch.h"
#include "Sibyl/Core/Application.h"
#include "Sibyl/Core/Layer.h"
#include "Sibyl/Core/Log.h"
#include "Sibyl/Core/Input.h"
#include "Sibyl/Core/KeyCodes.h"
#include "Sibyl/Core/MouseButtonCodes.h"
#include "Sibyl/ImGui/ImGuiLayer.h"

// EnterPoint
#include "Sibyl/Core/Instrumental.h"

// ImGui
#include "imgui.h"

// Scene
#include "Sibyl/Scene/Scene.h"
#include "Sibyl/Components/Components.h"

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Render Basic
#include "Sibyl/Renderer/Renderer.h"
#include "Sibyl/Graphic/AbstractAPI/Middle/Shader.h"
#include "Sibyl/Graphic/Geometry/TriangleMesh.h"
#include "Sibyl/Graphic/AbstractAPI/Middle/Texture.h"
#include "Sibyl/Renderer/Renderer2D.h"
#include "Sibyl/Graphic/AbstractAPI/Middle/FrameBuffer.h"

// Components
#include <Sibyl/Graphic/Core/Camera.h>
#include <Sibyl/Components/ViewCameraController.h>