#include "application.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/ext/vector_float4.hpp"
#include "graphics_internal.hpp"
#include "vulkan/vulkan_core.h"

#include <imgui.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace application {

struct Vertex {
  float position[3];
  float normal[3];
};

struct SceneUniform {
  glm::mat4 viewProjection;
  glm::vec4 cameraPosition;
  glm::vec4 ambient;
  glm::vec4 sunDirection;
  glm::vec4 sunColor;
  glm::ivec4 counts;
};

struct PointLight {
  glm::vec4 position;
  glm::vec4 color;
};

struct SpotLight {
  glm::vec4 position;
  glm::vec4 direction;
  glm::vec4 color;
  glm::vec4 cone;
};

struct ModelUniform {
  glm::mat4 model;
  glm::mat4 normalMatrix;
  glm::vec4 albedo;
  glm::vec4 specular;
};

struct RenderObject {
  glm::vec3 position;
  glm::vec3 rotationDeg;
  glm::vec3 scale;
  bool animated;

  glm::vec3 albedo;
  glm::vec3 specularColor;
  float shininess;

  VkBuffer uniformBuffer = VK_NULL_HANDLE;
  VmaAllocation uniformBufferAllocation = nullptr;
  void* uniformBufferMemory = nullptr;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

std::array<RenderObject, 4> renderObjects{};

std::vector<Vertex> vertices;
std::vector<uint32_t> indices;

// Камера
bool perspective;
float fovDeg;
float orthoSize;
glm::vec3 cameraPosition;
float cameraYaw;
float cameraPitch;

// Освещение
glm::vec3 ambientColor;
float ambientIntensity;
glm::vec3 sunDirection;
glm::vec3 sunColor;
float sunIntensity;

std::vector<PointLight> pointLights;
std::vector<SpotLight> spotLights;
uint32_t maxLights;

int selectedObject;

// Анимация
bool playing;
float animTime;
float animSpeed;
float orbitRadius;
float verticalAmplitude;
glm::vec3 spinDegPerSec;

VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout sceneSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout modelSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet sceneDescriptorSet = VK_NULL_HANDLE;

VkShaderModule vertexShader = VK_NULL_HANDLE;
VkShaderModule fragmentShader = VK_NULL_HANDLE;
VkPipeline graphicsPipeline = VK_NULL_HANDLE;

VkBuffer vertexBuffer = VK_NULL_HANDLE;
VmaAllocation vertexBufferAllocation = nullptr;
VkBuffer indexBuffer = VK_NULL_HANDLE;
VmaAllocation indexBufferAllocation = nullptr;

VkBuffer sceneBuffer = VK_NULL_HANDLE;
VmaAllocation sceneBufferAllocation = nullptr;
void* sceneBufferMemory = nullptr;

VkBuffer pointLightBuffer = VK_NULL_HANDLE;
VmaAllocation spotLightBufferAllocation = nullptr;
void* spotLightBufferMemory = nullptr;

// Читает скомпилированный шейдер .spv
VkShaderModule loadShaderModule(const char path[]) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return VK_NULL_HANDLE;
  }

  const size_t size = file.tellg();

  std::vector<uint32_t> buffer(size / sizeof(uint32_t));

  file.seekg(0);
  file.read(reinterpret_cast<char*>(buffer.data()), size);
  file.close();

  VkShaderModuleCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = buffer.data(),
  };

  VkShaderModule result = VK_NULL_HANDLE;
  vkCreateShaderModule(graphics::internal::context.device, &info, nullptr,
                       &result);
  return result;
}

// Создает буфер в памяти, доступной процессору (для вершин и индексов, которые не меняются)
bool createBuffer(const void* data, size_t size, VkBufferUsageFlags usage,
                  VkBuffer& buffer, VmaAllocation& allocation) {
  auto& context = graphics::internal::context;

  VkBufferCreateInfo bufferInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };

  VmaAllocationCreateInfo allocationInfo{
      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO,
  };

  if (vmaCreateBuffer(context.allocator, &bufferInfo, &allocationInfo, &buffer,
                      &allocation, nullptr) != VK_SUCCESS) {
    return false;
  }

  void* mapped = nullptr;
  if (vmaMapMemory(context.allocator, allocation, &mapped) != VK_SUCCESS) {
    return false;
  }
  memcpy(mapped, data, size);
  vmaFlushAllocation(context.allocator, allocation, 0, VK_WHOLE_SIZE);
  vmaUnmapMemory(context.allocator, allocation);
  return true;
}

// Память этого буфера остается отображенной
bool createMappedBuffer(size_t size, VkBufferUsageFlags usage, VkBuffer& buffer,
                        VmaAllocation& allocation, void*& memory) {
  auto& context = graphics::internal::context;

  VkBufferCreateInfo bufferInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };

  VmaAllocationCreateInfo allocationInfo{
      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO,
  };

  if (vmaCreateBuffer(context.allocator, &bufferInfo, &allocationInfo, &buffer,
                      &allocation, nullptr) != VK_SUCCESS) {
    return false;
  }

  return vmaMapMemory(context.allocator, allocation, &memory) == VK_SUCCESS;
}

// Строит параллелепипед
void generateBoxGeometry() {
  vertices.clear();
  indices.clear();

  vertices.reserve(6 * 4);
  indices.reserve(6 * 6);

  constexpr float halfSize[3] = {0.75f, 0.5f, 0, 4f};

  constexpr float cornerU[4] = {-1.0f, 1.0f, 1.0f, -1.0f};
  constexpr float cornerV[4] = {-1.0f, -1.0f, 1.0f, 1.0f};

  for (int axis = 0; axis < 3; ++axis) {
    const int u = (axis + 1) % 3;
    const int v = (axis + 2) % 3;

    for (int side = 0; side < 2; ++side) {
      const float sign = (side == 0) ? -1.0f : 1.0f;
      const uint32_t first = static_cast<uint32_t>(vertices.size());

      for (int corner = 0; corner < 4; ++corner) {
        Vertex vertex{};

        vertex.position[axis] = sign * halfSize[axis];
        vertex.position[u] = cornerU[corner] * halfSize[u];
        vertex.position[v] = cornerV[corner] * halfSize[v];

        vertex.normal[axis] = sign;

        vertices.push_back(vertex);
      }

      indices.insert(indices.end(), {
                                        first,
                                        first + 1,
                                        first + 2,
                                        first + 2,
                                        first + 3,
                                    });
    }
  }
}

bool initialize() {
  auto& context = graphics::internal::context;

  perspective = true;
  fovDeg = 60.0f;
  orthoSize = 4.0f;
  cameraPosition = glm::vec3(0.0f, 2.5f, 8.0f);
  cameraYaw = 0.0f;
  cameraPitch = -15.0f;

  ambientColor = glm::vec3(1.0f);
  ambientIntensity = 0.1f;
  sunDirection = glm::vec3(-0.3f, -1.0f, -0.4f);
  sunColor = glm::vec3(1.0f);
  sunIntensity = 0.5f;

  maxLights = 16;

  pointLights.push_back({
      .position = glm::vec4(0.0f, 2.5f, 3.0f, 1.0f),
      .color = glm::vec4(1.0f, 0.8f, 0.6f, 15.0f),
  });

  spotLights.push_back({
      .position = glm::vec4(3.0f, 4.0f, 2.0f, 1.0f),
      .direction = glm::vec4(-0.5f, -1.0f, -0.3f, 0.0f),
      .color = glm::vec4(0.6f, 0.8f, 1.0f, 60.0f),
      .cone = glm::vec4(15.0f, 25.0f, 0.0f, 0.0f),
  });

  selectedObject = 0;

  playing = true;
  animTime = 0.0f;
  animSpeed = 1.0f;
  orbitRadius = 2.0f;
  verticalAmplitude = 0.8f;
  spinDegPerSec = glm::vec3(30.0f, 45.0f, 15.0f);

  const glm::vec3 boxColors[3] = {
      glm::vec3(0.8f, 0.3f, 0.3f),
      glm::vec3(0.3f, 0.8f, 0.3f),
      glm::vec3(0.3f, 0.3f, 0.8f),
  };

  // Три параллелепипеда в ряд по оси Х
  for (size_t i = 0; i < 3; ++i) {
    RenderObject& object = renderObjects[i];

    object.position = glm::vec3(-2.0f + 2.0f * float(i), 0.0f, 0.0f);
    object.rotationDeg = glm::vec3(20.0f, 30.0f, 0.0f);
    object.scale = glm::vec3(0.7f);
    object.animated = true;

    object.albedo = boxColors[i];
    object.specularColor = glm::vec3(1.0f);
    object.shininess = 32.0f;
  }

  // Пол (плоский параллелепипед)
  RenderObject& floor = renderObjects[3];

  floor.position = glm::vec3(0.0f, -1.5f, 0.0f);
  floor.rotationDeg = glm::vec3(0.0f);
  floor.scale = glm::vec3(10.0f, 0.1f, 15.0f);
  floor.animated = false;

  floor.albedo = glm::vec3(0.6f);
  floor.specularColor = glm::vec3(0.2f);
  floor.shininess = 8.0f;

  generateBoxGeometry();

  if (!createBuffer(vertices.data(), vertices.size() * sizeof(Vertex),
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer,
                    vertexBufferAllocation)) {
    std::cerr << "Не удалось создать vertex buffer\n";
    return false;
  }

  if (!createBuffer(indices.data(), indices.size() * sizeof(uint32_t),
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer,
                    indexBufferAllocation)) {
    std::cerr << "Не удалось создать index buffer\n";
    return false;
  }

  // Эти данные меняются каждый кадр, все время работы память отображенная
  VkBufferCreateInfo uniformBufferInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(UniformBufferObject),
      .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE};

  VmaAllocationCreateInfo uniformAllocationInfo{
      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO};

  for (RenderObject& object : renderObjects) {
    if (vmaCreateBuffer(context.allocator, &uniformBufferInfo,
                        &uniformAllocationInfo, &object.uniformBuffer,
                        &object.uniformBufferAllocation,
                        nullptr) != VK_SUCCESS) {
      std::cerr << "Не удалось создать Uniform buffer \n";
      return false;
    }

    if (vmaMapMemory(context.allocator, object.uniformBufferAllocation,
                     &object.uniformBufferMemory) != VK_SUCCESS) {
      std::cerr << "Не удалось отобразить uniform buffer в память\n";
      return false;
    }
  }

  // Какие ресурсы ожидает шейдер (uniform buffer)
  VkDescriptorSetLayoutBinding uniformLayoutBinding{
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT};

  VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &uniformLayoutBinding,
  };

  if (vkCreateDescriptorSetLayout(context.device, &descriptorLayoutInfo,
                                  nullptr,
                                  &descriptorSetLayout) != VK_SUCCESS) {
    std::cerr << "Не удалось создать descriptor set layout \n";
    return false;
  }

  // Выделяем descriptor sets из пула
  VkDescriptorPoolSize poolSize{
      .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = static_cast<uint32_t>(renderObjects.size()),
  };

  VkDescriptorPoolCreateInfo descriptorPoolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = static_cast<uint32_t>(renderObjects.size()),
      .poolSizeCount = 1,
      .pPoolSizes = &poolSize,
  };

  if (vkCreateDescriptorPool(context.device, &descriptorPoolInfo, nullptr,
                             &descriptorPool) != VK_SUCCESS) {
    std::cerr << "Не удалось создать descriptor pool \n";
    return false;
  }

  // Выделяем set и привязываем к uniform buffer объекта
  for (RenderObject& object : renderObjects) {
    VkDescriptorSetAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptorSetLayout,
    };

    if (vkAllocateDescriptorSets(context.device, &allocateInfo,
                                 &object.descriptorSet) != VK_SUCCESS) {
      std::cerr << "Не удалось выделить descriptor set\n";
      return false;
    }

    VkDescriptorBufferInfo bufferInfo{.buffer = object.uniformBuffer,
                                      .offset = 0,
                                      .range = sizeof(UniformBufferObject)};

    VkWriteDescriptorSet descriptorWrite{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = object.descriptorSet,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &bufferInfo};

    vkUpdateDescriptorSets(context.device, 1, &descriptorWrite, 0, nullptr);
  }

  // Набор ресурсов для pipeline
  VkPipelineLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &descriptorSetLayout};

  if (vkCreatePipelineLayout(context.device, &layoutInfo, nullptr,
                             &pipelineLayout) != VK_SUCCESS) {
    std::cerr << "Не удалось создать pipeline layout \n";
    return false;
  }

  vertexShader = loadShaderModule("shaders/basic.vert.spv");
  fragmentShader = loadShaderModule("shaders/basic.frag.spv");

  if (vertexShader == VK_NULL_HANDLE || fragmentShader == VK_NULL_HANDLE) {
    std::cerr << "Не удалось загрузить шейдеры \n";
    return false;
  }

  // Распределение по стадиям конвейера
  VkPipelineShaderStageCreateInfo shaderStages[2]{};

  shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  shaderStages[0].module = vertexShader;
  shaderStages[0].pName = "main";

  shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  shaderStages[1].module = fragmentShader;
  shaderStages[1].pName = "main";

  // Формат вершин
  VkVertexInputBindingDescription vertexBinding{
      .binding = 0,
      .stride = sizeof(Vertex),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
  };

  VkVertexInputAttributeDescription vertexAttributes[2]{
      {.location = 0,
       .binding = 0,
       .format = VK_FORMAT_R32G32B32_SFLOAT,
       .offset = offsetof(Vertex, position)},
      {.location = 1,
       .binding = 0,
       .format = VK_FORMAT_R32G32B32_SFLOAT,
       .offset = offsetof(Vertex, color)},
  };

  VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount = 1;
  vertexInputInfo.pVertexBindingDescriptions = &vertexBinding;
  vertexInputInfo.vertexAttributeDescriptionCount = 2;
  vertexInputInfo.pVertexAttributeDescriptions = vertexAttributes;

  // Отдельный треугольник каждые 3 индекса
  VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo{};
  inputAssemblyInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  // viewport + scissor (значения задаются динамически)
  VkPipelineViewportStateCreateInfo viewportInfo{};
  viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportInfo.viewportCount = 1;
  viewportInfo.scissorCount = 1;

  // Растеризация
  VkPipelineRasterizationStateCreateInfo rasterInfo{};
  rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
  rasterInfo.cullMode = VK_CULL_MODE_NONE;
  rasterInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterInfo.lineWidth = 1.0f;

  // Тест глубины
  VkPipelineDepthStencilStateCreateInfo depthInfo{};
  depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthInfo.depthTestEnable = VK_TRUE;
  depthInfo.depthWriteEnable = VK_TRUE;
  depthInfo.depthCompareOp = VK_COMPARE_OP_LESS;

  // Сглаживание
  VkPipelineMultisampleStateCreateInfo sampleInfo{};
  sampleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  sampleInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  // Смешивание цветов (выключено)
  VkPipelineColorBlendAttachmentState colorAttachment{};
  colorAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorAttachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo blendInfo{};
  blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  blendInfo.attachmentCount = 1;
  blendInfo.pAttachments = &colorAttachment;

  const VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                          VK_DYNAMIC_STATE_SCISSOR};

  VkPipelineDynamicStateCreateInfo dynamicInfo{};
  dynamicInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicInfo.dynamicStateCount = 2;
  dynamicInfo.pDynamicStates = dynamicStates;

  // Сборка графического pipeline
  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = shaderStages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
  pipelineInfo.pViewportState = &viewportInfo;
  pipelineInfo.pRasterizationState = &rasterInfo;
  pipelineInfo.pDepthStencilState = &depthInfo;
  pipelineInfo.pMultisampleState = &sampleInfo;
  pipelineInfo.pColorBlendState = &blendInfo;
  pipelineInfo.pDynamicState = &dynamicInfo;
  pipelineInfo.layout = pipelineLayout;
  pipelineInfo.renderPass = context.render_pass;
  pipelineInfo.subpass = 0;

  if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
                                &pipelineInfo, nullptr,
                                &graphicsPipeline) != VK_SUCCESS) {
    std::cerr << "Не удалось создать графический pipeline \n";
    return false;
  }

  return true;
}

void shutdown() {
  auto& context = graphics::internal::context;
  vkQueueWaitIdle(context.graphics_queue);  // ждем освобождения

  vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);

  for (RenderObject& object : renderObjects) {
    vmaUnmapMemory(context.allocator, object.uniformBufferAllocation);
    vmaDestroyBuffer(context.allocator, object.uniformBuffer,
                     object.uniformBufferAllocation);
  }

  vmaDestroyBuffer(context.allocator, indexBuffer, indexBufferAllocation);
  vmaDestroyBuffer(context.allocator, vertexBuffer, vertexBufferAllocation);

  vkDestroyPipeline(context.device, graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(context.device, descriptorSetLayout, nullptr);

  vkDestroyShaderModule(context.device, vertexShader, nullptr);
  vkDestroyShaderModule(context.device, fragmentShader, nullptr);
}

// логика кадра и интерфейс (каждый кадр до render)
void update([[maybe_unused]] double time) {
  if (playing) {
    animTime += ImGui::GetIO().DeltaTime * animSpeed;
  }

  ImGui::Begin("Parallelepipeds");

  // Проекция
  ImGui::Text("Projection");
  if (ImGui::RadioButton("Perspective", perspective))
    perspective = true;
  ImGui::SameLine();
  if (ImGui::RadioButton("Orthographic", !perspective))
    perspective = false;
  if (perspective) {
    ImGui::SliderFloat("FOV (deg)", &fovDeg, 20.0f, 120.0f);
  } else {
    ImGui::SliderFloat("Ortho size", &orthoSize, 0.5f, 10.0f);
  }
  ImGui::SliderFloat("Camera distance", &cameraDistance, 2.0f, 30.0f);

  // Трансформация
  ImGui::Separator();
  ImGui::SliderInt("Object", &selectedObject, 0, int(renderObjects.size() - 1));

  RenderObject& selected = renderObjects[selectedObject];
  ImGui::DragFloat3("Position", selected.position, 0.01f);
  ImGui::DragFloat3("Rotation (deg)", selected.rotationDeg, 0.5f);
  ImGui::DragFloat3("Scale", selected.scale, 0.01f, 0.1f, 5.0f);
  ImGui::ColorEdit3("Tint", selected.tint);

  // Анимация
  ImGui::Separator();
  if (ImGui::Button(playing ? "Pause" : "Play"))
    playing = !playing;
  ImGui::SliderFloat("Speed", &animSpeed, 0.0f, 5.0f);
  ImGui::SliderFloat("Orbit radius", &orbitRadius, 0.0f, 5.0f);
  ImGui::SliderFloat("Vertical amplitude", &verticalAmplitude, 0.0f, 3.0f);
  ImGui::SliderFloat3("Spin (deg/s)", spinDegPerSec, -180.0f, 180.0f);

  ImGui::End();
}

// Запись команд для видеокарты (каждый кадр)
void render(const graphics::internal::FrameData& fd) {
  auto& context = graphics::internal::context;

  constexpr float degToRad = 3.14159265f / 180.0f;

  const float width = float(context.swapchain_extent.width);
  const float height = float(context.swapchain_extent.height);

  for (size_t i = 0; i < renderObjects.size(); ++i) {
    RenderObject& object = renderObjects[i];
    // У каждого объекта свое время (иначе синхронно)
    const float t = animTime + 2.0f * float(i);

    UniformBufferObject ubo{};

    // Траектория:  x = R * cos(t),  y = A * sin(2t),  z = R * sin(t)
    // Окружность в плоскости XZ радиусом R, по высоте колебания с амплитудой
    // A и удвоенной частотой
    ubo.position[0] = object.position[0] + orbitRadius * std::cos(t);
    ubo.position[1] =
        object.position[1] + verticalAmplitude * std::sin(2.0f * t);
    ubo.position[2] = object.position[2] + orbitRadius * std::sin(t);

    for (int axis = 0; axis < 3; ++axis) {
      // Угол = начальный из интерфейса + скорость вращения * время
      ubo.rotation[axis] =
          (object.rotationDeg[axis] + spinDegPerSec[axis] * t) * degToRad;
      ubo.scale[axis] = object.scale[axis];
      ubo.tint[axis] = object.tint[axis];
    }

    ubo.camera[0] = perspective ? 1.0f : 0.0f;
    ubo.camera[1] = perspective ? fovDeg * degToRad : orthoSize;
    ubo.camera[2] = cameraDistance;
    ubo.camera[3] = width / height;

    // Копируем в видимую память буфера и сбрасываем кэш процессора
    memcpy(object.uniformBufferMemory, &ubo, sizeof(ubo));
    vmaFlushAllocation(context.allocator, object.uniformBufferAllocation, 0,
                       sizeof(ubo));
  }

  // Начало записи в command buffer (graphics.cpp отправляет в vkQueueSubmit)
  vkResetCommandBuffer(fd.command_buffer, 0);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  vkBeginCommandBuffer(fd.command_buffer, &beginInfo);

  // Цвет и глубина (1 - самая дальняя) очистки
  VkClearValue clearValues[2]{};
  clearValues[0].color = {{0.01f, 0.01f, 0.01f, 1.0f}};
  clearValues[1].depthStencil = {1.0f, 0};

  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = context.render_pass;
  renderPassInfo.framebuffer = fd.framebuffer;
  renderPassInfo.renderArea.extent = context.swapchain_extent;
  renderPassInfo.clearValueCount = 2;
  renderPassInfo.pClearValues = clearValues;

  // Основная отрисовка
  // Очистка картинки и depth buffer
  vkCmdBeginRenderPass(fd.command_buffer, &renderPassInfo,
                       VK_SUBPASS_CONTENTS_INLINE);

  const VkViewport viewport{.x = 0,
                            .y = 0,
                            .width = width,
                            .height = height,
                            .minDepth = 0.0f,
                            .maxDepth = 1.0f};

  const VkRect2D scissor{
      .extent = context.swapchain_extent,
  };

  vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
  vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

  vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    graphicsPipeline);

  // Геометрия для всех объектов общая => привязка один раз
  const VkDeviceSize vertexBufferOffset = 0;
  vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertexBuffer,
                         &vertexBufferOffset);
  vkCmdBindIndexBuffer(fd.command_buffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

  // Для каждого объекта выбираем его descriptor set и рисуем ту же геометрию (шейдер сам выбирает данные из текущего set)
  for (const RenderObject& object : renderObjects) {
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 0, 1, &object.descriptorSet, 0,
                            nullptr);
    vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(indices.size()),
                     1, 0, 0, 0);
  }

  vkCmdEndRenderPass(fd.command_buffer);
  vkEndCommandBuffer(fd.command_buffer);
}

}  // namespace application