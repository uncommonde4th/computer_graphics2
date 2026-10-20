#include "application.hpp"
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

namespace application {

struct Vertex {
  float position[3];
  float color[3];
};

struct UniformBufferObject {
  float position[4];
  float rotation[4];
  float scale[4];
  float tint[4];
  float camera[4];
};

struct RenderObject {
  float position[3];
  float rotationDeg[3];
  float scale[3];
  float tint[3];

  VkBuffer uniformBuffer = VK_NULL_HANDLE;
  VmaAllocation uniformBufferAllocation = nullptr;
  void* uniformBufferMemory = nullptr;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

std::array<RenderObject, 3> renderObjects{};

std::vector<Vertex> vertices;
std::vector<uint32_t> indices;

bool perspective;
float fovDeg;
float orthoSize;
float cameraDistance;

int selectedObject;

bool playing;
float animTime;
float animSpeed;
float orbitRadius;
float verticalAmplitude;
float spinDegPerSec[3];

VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkShaderModule vertexShader = VK_NULL_HANDLE;
VkShaderModule fragmentShader = VK_NULL_HANDLE;
VkPipeline graphicsPipeline = VK_NULL_HANDLE;

VkBuffer vertexBuffer = VK_NULL_HANDLE;
VmaAllocation vertexBufferAllocation = nullptr;
VkBuffer indexBuffer = VK_NULL_HANDLE;
VmaAllocation indexBufferAllocation = nullptr;

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

// Строит параллелепипед
void generateBoxGeometry() {
  vertices.clear();
  indices.clear();

  vertices.reserve(8 + 2);
  indices.reserve(4 * 12);

  constexpr uint32_t corners = 4;
  constexpr float halfX = 0.75f;
  constexpr float halfY = 0.5f;
  constexpr float halfZ = 0.4f;

  for (uint32_t i = 0; i < corners; ++i) {
    const float x = (i == 1 || i == 2) ? halfX : -halfX;
    const float z = (i >= 2) ? halfZ : -halfZ;

    vertices.push_back({
        .position = {x, -halfY, z},
        .color = {0.6f + 0.4f * x / halfX, 0.2f, 0.6f + 0.4f * z / halfZ},
    });

    vertices.push_back({
        .position = {x, halfY, z},
        .color = {0.6f + 0.4f * x / halfX, 1.0f, 0.6f + 0.4f * z / halfZ},
    });
  }

  vertices.push_back({
      .position = {0.0f, -halfY, 0.0f},
      .color = {0.6f, 0.2f, 0.6f},
  });

  vertices.push_back({
      .position = {0.0f, halfY, 0.0f},
      .color = {0.6f, 1.0f, 0.6f},
  });

  const uint32_t bottomCenter = corners * 2;
  const uint32_t topCenter = bottomCenter + 1;

  for (uint32_t i = 0; i < corners; ++i) {
    const uint32_t next = (i + 1) % corners;

    const uint32_t bottom = i * 2;
    const uint32_t top = bottom + 1;

    const uint32_t nextBottom = next * 2;
    const uint32_t nextTop = nextBottom + 1;

    indices.insert(indices.end(), {
                                      bottom,
                                      top,
                                      nextBottom,
                                      nextBottom,
                                      top,
                                      nextTop,
                                      bottomCenter,
                                      bottom,
                                      nextBottom,
                                      topCenter,
                                      nextTop,
                                      top,
                                  });
  }
}

bool initialize() {
  auto& context = graphics::internal::context;

  perspective = true;
  fovDeg = 60.0f;
  orthoSize = 3.0f;
  cameraDistance = 7.0f;

  selectedObject = 0;

  playing = true;
  animTime = 0.0f;
  animSpeed = 1.0f;
  orbitRadius = 2.0f;
  verticalAmplitude = 0.8f;
  spinDegPerSec[0] = 30.0f;
  spinDegPerSec[1] = 45.0f;
  spinDegPerSec[2] = 15.0f;

  for (size_t i = 0; i < renderObjects.size(); ++i) {
    RenderObject& object = renderObjects[i];

    object.position[0] = -2.0f + 2.0f * float(i);
    object.position[1] = 0.0f;
    object.position[2] = 0.0f;

    object.rotationDeg[0] = 20.0f;
    object.rotationDeg[1] = 30.0f;
    object.rotationDeg[2] = 0.0f;

    object.scale[0] = object.scale[1] = object.scale[2] = 0.7f;
    object.tint[0] = object.tint[1] = object.tint[2] = 1.0f;
  }

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