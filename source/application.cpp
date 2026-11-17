#include "application.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/ext/vector_float4.hpp"
#include "glm/trigonometric.hpp"
#include "graphics_internal.hpp"

#include <imgui.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace application {

struct Vertex {
  float position[3];
  float normal[3];
  float texCoord[2];
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

// Элемент динамического uniform буфера (одна модель)
struct ModelUniform {
  glm::mat4 model;
  glm::mat4 normalMatrix;
  glm::vec4 albedo;
  glm::vec4 specular;
};

// Push константы включают нетривиальное сэмплирование
struct PushConstants {
  uint32_t effect;
  float time;
};

// Текстура и её материал свой набор дескрипторов
// (динамический uniform буфер + изображение с сэмплером)
struct Texture {
  std::string name;
  VkImage image = VK_NULL_HANDLE;
  VmaAllocation allocation = nullptr;
  VkImageView view = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

struct RenderObject {
  glm::vec3 position;
  glm::vec3 rotationDeg;
  glm::vec3 scale;
  bool animated;

  int textureIndex;
  bool nonTrivialSampling;

  glm::vec3 specularColor;
  float shininess;
};

std::array<RenderObject, 4> renderObjects{};

std::vector<Texture> textures;

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
bool showLightVisualizations = true;
VkBuffer coneVertexBuffer = VK_NULL_HANDLE;
VmaAllocation coneVertexBufferAllocation = nullptr;
void* coneVertexBufferMemory = nullptr;

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
VkDescriptorSetLayout materialSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet sceneDescriptorSet = VK_NULL_HANDLE;

VkShaderModule vertexShader = VK_NULL_HANDLE;
VkShaderModule fragmentShader = VK_NULL_HANDLE;
VkPipeline graphicsPipeline = VK_NULL_HANDLE;
VkPipeline linePipeline = VK_NULL_HANDLE;

// Вершины и индексы лежат в VRAM
VkBuffer vertexBuffer = VK_NULL_HANDLE;
VmaAllocation vertexBufferAllocation = nullptr;
VkBuffer indexBuffer = VK_NULL_HANDLE;
VmaAllocation indexBufferAllocation = nullptr;

VkBuffer sceneBuffer = VK_NULL_HANDLE;
VmaAllocation sceneBufferAllocation = nullptr;
void* sceneBufferMemory = nullptr;

VkBuffer pointLightBuffer = VK_NULL_HANDLE;
VmaAllocation pointLightBufferAllocation = nullptr;
void* pointLightBufferMemory = nullptr;

VkBuffer spotLightBuffer = VK_NULL_HANDLE;
VmaAllocation spotLightBufferAllocation = nullptr;
void* spotLightBufferMemory = nullptr;

// Один динамический uniform буфер на все модели
VkBuffer modelBuffer = VK_NULL_HANDLE;
VmaAllocation modelBufferAllocation = nullptr;
void* modelBufferMemory = nullptr;
size_t modelStride = 0;

VkSampler sampler = VK_NULL_HANDLE;
VkCommandPool transferCommandPool = VK_NULL_HANDLE;

size_t pointMarkerIndex(size_t i) {
  return renderObjects.size() + i;
}
size_t spotMarkerIndex(size_t i) {
  return renderObjects.size() + size_t(maxLights) + i;
}
size_t coneIndex() {
  return renderObjects.size() + size_t(maxLights) * 2;
}

// Записывает данные модели в её элемент динамического буфера
void writeModel(size_t index, const ModelUniform& ubo) {
  memcpy(static_cast<char*>(modelBufferMemory) + index * modelStride, &ubo,
         sizeof(ubo));
}

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

// Создает staging буфер в памяти, доступной процессору, и копирует туда данные
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

// Командный буфер для разовых операций (копирование в VRAM)
VkCommandBuffer beginSingleTimeCommands() {
  auto& context = graphics::internal::context;

  VkCommandBufferAllocateInfo allocateInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = transferCommandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };

  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(context.device, &allocateInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  return commandBuffer;
}

// Отправляет команды и ждет их выполнения
void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
  auto& context = graphics::internal::context;

  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &commandBuffer,
  };

  vkQueueSubmit(context.graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(context.graphics_queue);

  vkFreeCommandBuffers(context.device, transferCommandPool, 1, &commandBuffer);
}

// Создает буфер в VRAM данные сначала попадают в staging буфер, затем копируются командой GPU
bool createDeviceBuffer(const void* data, size_t size, VkBufferUsageFlags usage,
                        VkBuffer& buffer, VmaAllocation& allocation) {
  auto& context = graphics::internal::context;

  VkBuffer stagingBuffer = VK_NULL_HANDLE;
  VmaAllocation stagingAllocation = nullptr;

  if (!createBuffer(data, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingBuffer,
                    stagingAllocation)) {
    return false;
  }

  VkBufferCreateInfo bufferInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };

  VmaAllocationCreateInfo allocationInfo{
      .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
  };

  if (vmaCreateBuffer(context.allocator, &bufferInfo, &allocationInfo, &buffer,
                      &allocation, nullptr) != VK_SUCCESS) {
    vmaDestroyBuffer(context.allocator, stagingBuffer, stagingAllocation);
    return false;
  }

  VkCommandBuffer commandBuffer = beginSingleTimeCommands();

  VkBufferCopy copyRegion{
      .srcOffset = 0,
      .dstOffset = 0,
      .size = size,
  };
  vkCmdCopyBuffer(commandBuffer, stagingBuffer, buffer, 1, &copyRegion);

  endSingleTimeCommands(commandBuffer);

  vmaDestroyBuffer(context.allocator, stagingBuffer, stagingAllocation);
  return true;
}

// Загружает изображение с диска и создает из него текстуру в VRAM
bool loadTexture(const std::string& path, Texture& texture) {
  auto& context = graphics::internal::context;

  int width = 0;
  int height = 0;
  int channels = 0;
  stbi_uc* pixels =
      stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
  if (pixels == nullptr) {
    std::cerr << "Не удалось загрузить изображение " << path << "\n";
    return false;
  }

  const size_t imageSize = size_t(width) * size_t(height) * 4;

  VkBuffer stagingBuffer = VK_NULL_HANDLE;
  VmaAllocation stagingAllocation = nullptr;
  const bool staged =
      createBuffer(pixels, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   stagingBuffer, stagingAllocation);
  stbi_image_free(pixels);
  if (!staged) {
    return false;
  }

  // Изображение с цветом (sRGB), память в VRAM
  VkImageCreateInfo imageInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_SRGB,
      .extent = {uint32_t(width), uint32_t(height), 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };

  VmaAllocationCreateInfo allocationInfo{
      .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
  };

  if (vmaCreateImage(context.allocator, &imageInfo, &allocationInfo,
                     &texture.image, &texture.allocation,
                     nullptr) != VK_SUCCESS) {
    vmaDestroyBuffer(context.allocator, stagingBuffer, stagingAllocation);
    std::cerr << "Не удалось создать изображение для текстуры " << path << "\n";
    return false;
  }

  VkImageSubresourceRange subresourceRange{
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .baseMipLevel = 0,
      .levelCount = 1,
      .baseArrayLayer = 0,
      .layerCount = 1,
  };

  VkCommandBuffer commandBuffer = beginSingleTimeCommands();

  // готовим изображение к копированию
  VkImageMemoryBarrier toTransfer{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = 0,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = texture.image,
      .subresourceRange = subresourceRange,
  };

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &toTransfer);

  VkBufferImageCopy copyRegion{
      .bufferOffset = 0,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageOffset = {0, 0, 0},
      .imageExtent = {uint32_t(width), uint32_t(height), 1},
  };

  vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, texture.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

  // изображение можно читать из шейдера
  VkImageMemoryBarrier toShaderRead{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = texture.image,
      .subresourceRange = subresourceRange,
  };

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &toShaderRead);

  endSingleTimeCommands(commandBuffer);

  vmaDestroyBuffer(context.allocator, stagingBuffer, stagingAllocation);

  VkImageViewCreateInfo viewInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = texture.image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_SRGB,
      .subresourceRange = subresourceRange,
  };

  if (vkCreateImageView(context.device, &viewInfo, nullptr, &texture.view) !=
      VK_SUCCESS) {
    std::cerr << "Не удалось создать image view для текстуры " << path << "\n";
    return false;
  }

  texture.name = std::filesystem::path(path).filename().string();
  return true;
}

// Строит параллелепипед
void generateBoxGeometry() {
  vertices.clear();
  indices.clear();

  vertices.reserve(6 * 4);
  indices.reserve(6 * 6);

  constexpr float halfSize[3] = {0.75f, 0.5f, 0.4f};

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

        // Текстурные координаты: каждая грань целиком покрывается текстурой
        vertex.texCoord[0] = (cornerU[corner] + 1.0f) * 0.5f;
        vertex.texCoord[1] = (cornerV[corner] + 1.0f) * 0.5f;

        vertices.push_back(vertex);
      }

      indices.insert(indices.end(), {
                                        first,
                                        first + 1,
                                        first + 2,
                                        first,
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
  showLightVisualizations = true;

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

  // Три параллелепипеда в ряд по оси Х
  for (size_t i = 0; i < 3; ++i) {
    RenderObject& object = renderObjects[i];

    object.position = glm::vec3(-2.0f + 2.0f * float(i), 0.0f, 0.0f);
    object.rotationDeg = glm::vec3(20.0f, 30.0f, 0.0f);
    object.scale = glm::vec3(0.7f);
    object.animated = true;

    object.textureIndex = int(i);
    object.nonTrivialSampling = false;

    object.specularColor = glm::vec3(1.0f);
    object.shininess = 32.0f;
  }

  // Нетривиальное сэмплирование включено у среднего параллелепипеда
  renderObjects[1].nonTrivialSampling = true;

  // Пол (плоский параллелепипед)
  RenderObject& floor = renderObjects[3];

  floor.position = glm::vec3(0.0f, -1.5f, 0.0f);
  floor.rotationDeg = glm::vec3(0.0f);
  floor.scale = glm::vec3(10.0f, 0.1f, 15.0f);
  floor.animated = false;

  floor.textureIndex = 3;
  floor.nonTrivialSampling = false;

  floor.specularColor = glm::vec3(0.2f);
  floor.shininess = 8.0f;

  generateBoxGeometry();

  // Command pool для разовых команд копирования
  VkCommandPoolCreateInfo commandPoolInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = context.graphics_queue_index,
  };

  if (vkCreateCommandPool(context.device, &commandPoolInfo, nullptr,
                          &transferCommandPool) != VK_SUCCESS) {
    std::cerr << "Не удалось создать command pool для копирования\n";
    return false;
  }

  if (!createDeviceBuffer(vertices.data(), vertices.size() * sizeof(Vertex),
                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer,
                          vertexBufferAllocation)) {
    std::cerr << "Не удалось создать vertex buffer\n";
    return false;
  }

  if (!createDeviceBuffer(indices.data(), indices.size() * sizeof(uint32_t),
                          VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer,
                          indexBufferAllocation)) {
    std::cerr << "Не удалось создать index buffer\n";
    return false;
  }

  if (!createMappedBuffer(sizeof(SceneUniform),
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sceneBuffer,
                          sceneBufferAllocation, sceneBufferMemory)) {
    std::cerr << "Не удалось создать uniform buffer сцены\n";
    return false;
  }

  if (!createMappedBuffer(maxLights * sizeof(PointLight),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, pointLightBuffer,
                          pointLightBufferAllocation, pointLightBufferMemory)) {
    std::cerr << "Не удалось создать storage buffer точечных источников\n";
    return false;
  }

  if (!createMappedBuffer(maxLights * sizeof(SpotLight),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spotLightBuffer,
                          spotLightBufferAllocation, spotLightBufferMemory)) {
    std::cerr << "Не удалось создать storage buffer прожекторных источников\n";
    return false;
  }

  // Динамический uniform буфер
  VkPhysicalDeviceProperties deviceProperties{};
  vkGetPhysicalDeviceProperties(context.physical_device, &deviceProperties);
  const size_t alignment =
      size_t(deviceProperties.limits.minUniformBufferOffsetAlignment);
  modelStride = (sizeof(ModelUniform) + alignment - 1) / alignment * alignment;

  if (!createMappedBuffer((coneIndex() + 1) * modelStride,
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, modelBuffer,
                          modelBufferAllocation, modelBufferMemory)) {
    std::cerr << "Не удалось создать динамический uniform buffer\n";
    return false;
  }

  // Для каждого конуса: 24 отрезка окружности и 4 образующие
  constexpr size_t coneVerticesPerLight = 56;
  if (!createMappedBuffer(
          size_t(maxLights) * coneVerticesPerLight * sizeof(Vertex),
          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, coneVertexBuffer,
          coneVertexBufferAllocation, coneVertexBufferMemory)) {
    std::cerr << "Не удалось создать vertex buffer конусов\n";
    return false;
  }

  // Общий элемент конусов
  ModelUniform coneUniform{};
  coneUniform.model = glm::mat4(1.0f);
  coneUniform.normalMatrix = glm::mat4(1.0f);
  coneUniform.albedo = glm::vec4(1.0f);
  coneUniform.specular = glm::vec4(0.0f);
  writeModel(coneIndex(), coneUniform);

  // Текстуры
  const std::vector<std::string> texturePaths = {
      "textures/sharingan.png",
      "textures/grass.png",
      "textures/sheep.png",
  };

  textures.resize(texturePaths.size());
  for (size_t i = 0; i < texturePaths.size(); ++i) {
    if (!loadTexture(texturePaths[i], textures[i])) {
      return false;
    }
  }

  for (RenderObject& object : renderObjects) {
    object.textureIndex %= int(textures.size());
  }

  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.maxLod = 1.0f;
  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

  if (vkCreateSampler(context.device, &samplerInfo, nullptr, &sampler) !=
      VK_SUCCESS) {
    std::cerr << "Не удалось создать sampler\n";
    return false;
  }

  VkDescriptorSetLayoutBinding sceneBindings[3]{
      {.binding = 0,
       .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
      {
          .binding = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      {
          .binding = 2,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
  };

  VkDescriptorSetLayoutCreateInfo sceneLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3,
      .pBindings = sceneBindings,
  };

  if (vkCreateDescriptorSetLayout(context.device, &sceneLayoutInfo, nullptr,
                                  &sceneSetLayout) != VK_SUCCESS) {
    std::cerr << "Не удалось создать descriptor set layout сцены\n";
    return false;
  }

  // Набор материала: динамический uniform буфер моделей + текстура
  VkDescriptorSetLayoutBinding materialBindings[2]{
      {
          .binding = 0,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
          .descriptorCount = 1,
          .stageFlags =
              VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      {
          .binding = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
  };

  VkDescriptorSetLayoutCreateInfo materialLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = materialBindings,
  };

  if (vkCreateDescriptorSetLayout(context.device, &materialLayoutInfo, nullptr,
                                  &materialSetLayout) != VK_SUCCESS) {
    std::cerr << "Не удалось создать descriptor set layout материала \n";
    return false;
  }

  const uint32_t textureCount = static_cast<uint32_t>(textures.size());

  VkDescriptorPoolSize poolSizes[4]{
      {
          .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          .descriptorCount = 1,
      },
      {
          .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 2,
      },
      {
          .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
          .descriptorCount = textureCount,
      },
      {
          .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = textureCount,
      },
  };

  VkDescriptorPoolCreateInfo descriptorPoolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1 + textureCount,
      .poolSizeCount = 4,
      .pPoolSizes = poolSizes,
  };

  if (vkCreateDescriptorPool(context.device, &descriptorPoolInfo, nullptr,
                             &descriptorPool) != VK_SUCCESS) {
    std::cerr << "Не удалось создать descriptor pool\n";
    return false;
  }

  // Descriptor set сцены
  VkDescriptorSetAllocateInfo sceneAllocateInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &sceneSetLayout,
  };

  if (vkAllocateDescriptorSets(context.device, &sceneAllocateInfo,
                               &sceneDescriptorSet) != VK_SUCCESS) {
    std::cerr << "Не удалось выделить descriptor set сцены\n";
    return false;
  }

  VkDescriptorBufferInfo sceneBufferInfo{
      .buffer = sceneBuffer,
      .offset = 0,
      .range = sizeof(SceneUniform),
  };

  VkDescriptorBufferInfo pointLightBufferInfo{
      .buffer = pointLightBuffer,
      .offset = 0,
      .range = VK_WHOLE_SIZE,
  };

  VkDescriptorBufferInfo spotLightBufferInfo{
      .buffer = spotLightBuffer,
      .offset = 0,
      .range = VK_WHOLE_SIZE,
  };

  VkWriteDescriptorSet sceneWrites[3]{
      {
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = sceneDescriptorSet,
          .dstBinding = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          .pBufferInfo = &sceneBufferInfo,
      },
      {
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = sceneDescriptorSet,
          .dstBinding = 1,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &pointLightBufferInfo,
      },
      {
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = sceneDescriptorSet,
          .dstBinding = 2,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &spotLightBufferInfo,
      },
  };

  vkUpdateDescriptorSets(context.device, 3, sceneWrites, 0, nullptr);

  // Descriptor set каждого материала, один и тот же динамический буфер, своя текстура
  for (Texture& texture : textures) {
    VkDescriptorSetAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &materialSetLayout,
    };

    if (vkAllocateDescriptorSets(context.device, &allocateInfo,
                                 &texture.descriptorSet) != VK_SUCCESS) {
      std::cerr << "Не удалось выделить descriptor set материала\n";
      return false;
    }

    // range - размер одного элемента, сдвиг добавляется при привязке набора
    VkDescriptorBufferInfo modelBufferInfo{
        .buffer = modelBuffer,
        .offset = 0,
        .range = sizeof(ModelUniform),
    };

    VkDescriptorImageInfo imageInfo{
        .sampler = sampler,
        .imageView = texture.view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    VkWriteDescriptorSet materialWrites[2]{
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = texture.descriptorSet,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .pBufferInfo = &modelBufferInfo,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = texture.descriptorSet,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &imageInfo,
        },
    };

    vkUpdateDescriptorSets(context.device, 2, materialWrites, 0, nullptr);
  }

  VkDescriptorSetLayout setLayouts[2] = {sceneSetLayout, materialSetLayout};

  // Push константы, нужны только фрагментному шейдеру
  VkPushConstantRange pushRange{
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      .offset = 0,
      .size = sizeof(PushConstants),
  };

  // Набор ресурсов для pipeline
  VkPipelineLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 2,
      .pSetLayouts = setLayouts,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushRange,
  };

  if (vkCreatePipelineLayout(context.device, &layoutInfo, nullptr,
                             &pipelineLayout) != VK_SUCCESS) {
    std::cerr << "Не удалось создать pipeline layout \n";
    return false;
  }

  // Шейдеры
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

  VkVertexInputAttributeDescription vertexAttributes[3]{
      {.location = 0,
       .binding = 0,
       .format = VK_FORMAT_R32G32B32_SFLOAT,
       .offset = offsetof(Vertex, position)},
      {.location = 1,
       .binding = 0,
       .format = VK_FORMAT_R32G32B32_SFLOAT,
       .offset = offsetof(Vertex, normal)},
      {.location = 2,
       .binding = 0,
       .format = VK_FORMAT_R32G32_SFLOAT,
       .offset = offsetof(Vertex, texCoord)},
  };

  VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount = 1;
  vertexInputInfo.pVertexBindingDescriptions = &vertexBinding;
  vertexInputInfo.vertexAttributeDescriptionCount = 3;
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
  dynamicInfo.dynamicStateCount =
      sizeof(dynamicStates) / sizeof(dynamicStates[0]);
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

  VkPipelineInputAssemblyStateCreateInfo lineAssemblyInfo = inputAssemblyInfo;
  lineAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

  VkGraphicsPipelineCreateInfo linePipelineInfo = pipelineInfo;
  linePipelineInfo.pInputAssemblyState = &lineAssemblyInfo;

  if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
                                &linePipelineInfo, nullptr,
                                &linePipeline) != VK_SUCCESS) {
    std::cerr << "Не удалось создать pipeline проволочных конусов\n";
    return false;
  }

  return true;
}

void shutdown() {
  auto& context = graphics::internal::context;
  vkQueueWaitIdle(context.graphics_queue);  // ждем освобождения

  vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);

  vkDestroySampler(context.device, sampler, nullptr);

  for (Texture& texture : textures) {
    vkDestroyImageView(context.device, texture.view, nullptr);
    vmaDestroyImage(context.allocator, texture.image, texture.allocation);
  }

  vmaUnmapMemory(context.allocator, modelBufferAllocation);
  vmaDestroyBuffer(context.allocator, modelBuffer, modelBufferAllocation);

  vmaUnmapMemory(context.allocator, coneVertexBufferAllocation);
  vmaDestroyBuffer(context.allocator, coneVertexBuffer,
                   coneVertexBufferAllocation);

  vmaUnmapMemory(context.allocator, spotLightBufferAllocation);
  vmaDestroyBuffer(context.allocator, spotLightBuffer,
                   spotLightBufferAllocation);

  vmaUnmapMemory(context.allocator, pointLightBufferAllocation);
  vmaDestroyBuffer(context.allocator, pointLightBuffer,
                   pointLightBufferAllocation);

  vmaUnmapMemory(context.allocator, sceneBufferAllocation);
  vmaDestroyBuffer(context.allocator, sceneBuffer, sceneBufferAllocation);

  vmaDestroyBuffer(context.allocator, indexBuffer, indexBufferAllocation);
  vmaDestroyBuffer(context.allocator, vertexBuffer, vertexBufferAllocation);

  vkDestroyPipeline(context.device, linePipeline, nullptr);
  vkDestroyPipeline(context.device, graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(context.device, materialSetLayout, nullptr);
  vkDestroyDescriptorSetLayout(context.device, sceneSetLayout, nullptr);

  vkDestroyShaderModule(context.device, vertexShader, nullptr);
  vkDestroyShaderModule(context.device, fragmentShader, nullptr);

  vkDestroyCommandPool(context.device, transferCommandPool, nullptr);
}

// логика кадра и интерфейс (каждый кадр до render)
void update([[maybe_unused]] double time) {
  if (playing) {
    animTime += ImGui::GetIO().DeltaTime * animSpeed;
  }

  ImGui::Begin("Camera");

  // Проекция
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

  ImGui::DragFloat3("Position", glm::value_ptr(cameraPosition), 0.05f);
  ImGui::SliderFloat("Yaw", &cameraYaw, -180.0f, 180.0f);
  ImGui::SliderFloat("Pitch", &cameraPitch, -89.0f, 89.0f);

  ImGui::End();

  // Освещение

  ImGui::Begin("Lighting");

  ImGui::Text("Ambient light");
  ImGui::ColorEdit3("Ambient color", glm::value_ptr(ambientColor));
  ImGui::SliderFloat("Ambient intensity", &ambientIntensity, 0.0f, 1.0f);

  ImGui::Separator();
  ImGui::Text("Directional light");
  ImGui::DragFloat3("Sun direction", glm::value_ptr(sunDirection), 0.01f);
  ImGui::ColorEdit3("Sun color", glm::value_ptr(sunColor));
  ImGui::SliderFloat("Sun intensity", &sunIntensity, 0.0f, 5.0f);
  ImGui::Separator();
  ImGui::Checkbox("Show light visualizations", &showLightVisualizations);

  // Точечные источники. Удаление откладываем до конца цикла, чтобы не менять вектор во время обхода.
  ImGui::Separator();
  ImGui::Text("Point lights (%d)", int(pointLights.size()));
  ImGui::PushID("point");

  int removePoint = -1;
  for (size_t i = 0; i < pointLights.size(); ++i) {
    PointLight& light = pointLights[i];

    ImGui::PushID(int(i));
    ImGui::Text("Point light %d", int(i));
    ImGui::DragFloat3("Position", glm::value_ptr(light.position), 0.05f);
    ImGui::ColorEdit3("Color", glm::value_ptr(light.color));
    ImGui::SliderFloat("Intensity", &light.color.w, 0.0f, 100.0f);
    if (ImGui::Button("Remove"))
      removePoint = int(i);
    ImGui::PopID();
  }

  if (removePoint >= 0) {
    pointLights.erase(pointLights.begin() + removePoint);
  }

  if (ImGui::Button("Add point light") && pointLights.size() < maxLights) {
    pointLights.push_back({
        .position = glm::vec4(0.0f, 2.0f, 0.0f, 1.0f),
        .color = glm::vec4(1.0f, 1.0f, 1.0f, 15.0f),
    });
  }

  ImGui::PopID();

  // Прожекторные источники
  ImGui::Separator();
  ImGui::Text("Spot lights (%d)", int(spotLights.size()));
  ImGui::PushID("spot");

  int removeSpot = -1;
  for (size_t i = 0; i < spotLights.size(); ++i) {
    SpotLight& light = spotLights[i];

    ImGui::PushID(int(i));
    ImGui::Text("Spot light %d", int(i));
    ImGui::DragFloat3("Position", glm::value_ptr(light.position), 0.05f);
    ImGui::DragFloat3("Direction", glm::value_ptr(light.direction), 0.01f);
    ImGui::ColorEdit3("Color", glm::value_ptr(light.color));
    ImGui::SliderFloat("Intensity", &light.color.w, 0.0f, 200.0f);
    ImGui::SliderFloat("Inner angle", &light.cone.x, 0.0f, 90.0f);
    ImGui::SliderFloat("Outer angle", &light.cone.y, 0.0f, 90.0f);
    if (ImGui::Button("Remove"))
      removeSpot = int(i);
    ImGui::PopID();

    // Внутренний угол не может быть больше внешнего
    if (light.cone.x > light.cone.y)
      light.cone.x = light.cone.y;
  }

  if (removeSpot >= 0) {
    spotLights.erase(spotLights.begin() + removeSpot);
  }

  if (ImGui::Button("Add spot light") && spotLights.size() < maxLights) {
    spotLights.push_back({
        .position = glm::vec4(0.0f, 4.0f, 0.0f, 1.0f),
        .direction = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f),
        .color = glm::vec4(1.0f, 1.0f, 1.0f, 60.0f),
        .cone = glm::vec4(15.0f, 25.0f, 0.0f, 0.0f),
    });
  }

  ImGui::PopID();

  ImGui::End();

  // Объекты

  ImGui::Begin("Objects");

  ImGui::SliderInt("Object", &selectedObject, 0, int(renderObjects.size() - 1));

  RenderObject& selected = renderObjects[selectedObject];

  ImGui::DragFloat3("Position", glm::value_ptr(selected.position), 0.01f);
  ImGui::DragFloat3("Rotation (deg)", glm::value_ptr(selected.rotationDeg),
                    0.5f);
  ImGui::DragFloat3("Scale", glm::value_ptr(selected.scale), 0.01f, 0.1f,
                    20.0f);

  ImGui::Separator();
  ImGui::Text("Material");

  // Выбор текстуры из выпадающего списка
  if (ImGui::BeginCombo("Texture",
                        textures[selected.textureIndex].name.c_str())) {
    for (int i = 0; i < int(textures.size()); ++i) {
      const bool isSelected = (selected.textureIndex == i);
      if (ImGui::Selectable(textures[i].name.c_str(), isSelected))
        selected.textureIndex = i;
      if (isSelected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }

  ImGui::Checkbox("Non-trivial sampling", &selected.nonTrivialSampling);
  ImGui::ColorEdit3("Specular color", glm::value_ptr(selected.specularColor));
  ImGui::SliderFloat("Shininess", &selected.shininess, 1.0f, 256.0f);

  // Анимация (для всех подвижных объектов)
  ImGui::Separator();
  if (ImGui::Button(playing ? "Pause" : "Play"))
    playing = !playing;
  ImGui::SliderFloat("Speed", &animSpeed, 0.0f, 5.0f);
  ImGui::SliderFloat("Orbit radius", &orbitRadius, 0.0f, 5.0f);
  ImGui::SliderFloat("Vertical amplitude", &verticalAmplitude, 0.0f, 3.0f);
  ImGui::SliderFloat3("Spin (deg/s)", glm::value_ptr(spinDegPerSec), -180.0f,
                      180.0f);

  ImGui::End();
}

// Запись команд для видеокарты (каждый кадр)
void render(const graphics::internal::FrameData& fd) {
  auto& context = graphics::internal::context;

  const float width = float(context.swapchain_extent.width);
  const float height = float(context.swapchain_extent.height);
  const float aspect = width / height;

  glm::mat4 view = glm::rotate(glm::mat4(1.0f), glm::radians(-cameraPitch),
                               glm::vec3(1.0f, 0.0f, 0.0f));
  view =
      glm::rotate(view, glm::radians(-cameraYaw), glm::vec3(0.0f, 1.0f, 0.0f));
  view = glm::translate(view, -cameraPosition);

  glm::mat4 projection;
  if (perspective) {
    projection = glm::perspective(glm::radians(fovDeg), aspect, 0.1f, 100.0f);

  } else {
    const float halfWidth = orthoSize * aspect;
    projection =
        glm::ortho(-halfWidth, halfWidth, -orthoSize, orthoSize, 0.1f, 100.0f);
  }

  projection[1][1] *= -1.0f;

  SceneUniform scene{};
  scene.viewProjection = projection * view;
  scene.cameraPosition = glm::vec4(cameraPosition, 1.0f);
  scene.ambient = glm::vec4(ambientColor, ambientIntensity);
  scene.sunDirection = glm::vec4(sunDirection, 0.0f);
  scene.sunColor = glm::vec4(sunColor, sunIntensity);
  scene.counts =
      glm::ivec4(int(pointLights.size()), int(spotLights.size()), 0, 0);

  memcpy(sceneBufferMemory, &scene, sizeof(scene));
  vmaFlushAllocation(context.allocator, sceneBufferAllocation, 0,
                     VK_WHOLE_SIZE);

  // Источники света копируем в storage буферы
  if (!pointLights.empty()) {
    memcpy(pointLightBufferMemory, pointLights.data(),
           pointLights.size() * sizeof(PointLight));
    vmaFlushAllocation(context.allocator, pointLightBufferAllocation, 0,
                       VK_WHOLE_SIZE);
  }

  if (!spotLights.empty()) {
    memcpy(spotLightBufferMemory, spotLights.data(),
           spotLights.size() * sizeof(SpotLight));
    vmaFlushAllocation(context.allocator, spotLightBufferAllocation, 0,
                       VK_WHOLE_SIZE);
  }

  // Данные каждого объекта: матрица трансформации и материал

  for (size_t i = 0; i < renderObjects.size(); ++i) {
    RenderObject& object = renderObjects[i];

    glm::vec3 position = object.position;
    glm::vec3 rotation = object.rotationDeg;

    if (object.animated) {
      const float t = animTime + 2.0f * float(i);

      position += glm::vec3(orbitRadius * std::cos(t),
                            verticalAmplitude * std::sin(2.0f * t),
                            orbitRadius * std::sin(t));
      rotation += spinDegPerSec * t;
    }

    glm::mat4 model(1.0f);
    model = glm::translate(model, position);
    model = glm::rotate(model, glm::radians(rotation.z),
                        glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::rotate(model, glm::radians(rotation.y),
                        glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rotation.x),
                        glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::scale(model, object.scale);

    ModelUniform ubo{};
    ubo.model = model;
    ubo.normalMatrix = glm::transpose(glm::inverse(model));
    ubo.albedo = glm::vec4(1.0f);
    ubo.specular = glm::vec4(object.specularColor, object.shininess);

    writeModel(i, ubo);
  }

  // Подготовка цветных маркеров точечных и прожекторных источников.
  // У каждого маркера свой элемент динамического буфера, чтобы записи не
  // перезаписывали данные предыдущих draw-команд в одном command buffer.
  if (showLightVisualizations) {
    auto updateMarker = [&](size_t modelIndex, const glm::vec3& position,
                            const glm::vec3& color) {
      glm::mat4 model(1.0f);
      model = glm::translate(model, position);
      // Исходная геометрия имеет полуразмеры (0.75, 0.5, 0.4).
      // Компенсируем пропорции, чтобы маркер стал кубом со стороной 0.24.
      model = glm::scale(model,
                         glm::vec3(0.12f / 0.75f, 0.12f / 0.5f, 0.12f / 0.4f));

      ModelUniform markerUniform{};
      markerUniform.model = model;
      markerUniform.normalMatrix = glm::transpose(glm::inverse(model));
      markerUniform.albedo = glm::vec4(color, 1.0f);
      markerUniform.specular = glm::vec4(0.0f);  // unlit marker

      writeModel(modelIndex, markerUniform);
    };

    for (size_t i = 0; i < pointLights.size(); ++i) {
      updateMarker(pointMarkerIndex(i), glm::vec3(pointLights[i].position),
                   glm::vec3(pointLights[i].color));
    }

    for (size_t i = 0; i < spotLights.size(); ++i) {
      updateMarker(spotMarkerIndex(i), glm::vec3(spotLights[i].position),
                   glm::vec3(spotLights[i].color));
    }
  }

  // Все элементы динамического буфера записаны
  vmaFlushAllocation(context.allocator, modelBufferAllocation, 0,
                     VK_WHOLE_SIZE);

  // Проволочная геометрия внешних конусов прожекторов в мировых координатах.
  // Перестраивается каждый кадр, поэтому следует за позицией, направлением и углом.
  std::vector<Vertex> coneVertices;
  if (showLightVisualizations) {
    constexpr int segments = 24;
    constexpr float coneLength = 1.5f;
    coneVertices.reserve(spotLights.size() * 56);

    auto addLine = [&](const glm::vec3& a, const glm::vec3& b) {
      Vertex va{};
      Vertex vb{};
      va.position[0] = a.x;
      va.position[1] = a.y;
      va.position[2] = a.z;
      vb.position[0] = b.x;
      vb.position[1] = b.y;
      vb.position[2] = b.z;
      coneVertices.push_back(va);
      coneVertices.push_back(vb);
    };

    for (const SpotLight& light : spotLights) {
      glm::vec3 direction(light.direction);
      if (glm::length(direction) < 0.0001f)
        direction = glm::vec3(0.0f, -1.0f, 0.0f);
      else
        direction = glm::normalize(direction);

      const glm::vec3 helper =
          std::abs(glm::dot(direction, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.95f
              ? glm::vec3(1.0f, 0.0f, 0.0f)
              : glm::vec3(0.0f, 1.0f, 0.0f);
      const glm::vec3 right = glm::normalize(glm::cross(direction, helper));
      const glm::vec3 up = glm::normalize(glm::cross(right, direction));

      const glm::vec3 apex(light.position);
      const glm::vec3 center = apex + direction * coneLength;
      const float outerAngle = glm::clamp(light.cone.y, 0.1f, 89.0f);
      const float radius = std::tan(glm::radians(outerAngle)) * coneLength;

      std::array<glm::vec3, segments> ring{};
      for (int j = 0; j < segments; ++j) {
        const float angle = 6.28318530718f * float(j) / float(segments);
        ring[size_t(j)] =
            center + radius * (right * std::cos(angle) + up * std::sin(angle));
      }

      for (int j = 0; j < segments; ++j)
        addLine(ring[size_t(j)], ring[size_t((j + 1) % segments)]);

      for (int j = 0; j < segments; j += segments / 4)
        addLine(apex, ring[size_t(j)]);
    }

    if (!coneVertices.empty()) {
      const size_t bytes = coneVertices.size() * sizeof(Vertex);
      memcpy(coneVertexBufferMemory, coneVertices.data(), bytes);
      vmaFlushAllocation(context.allocator, coneVertexBufferAllocation, 0,
                         bytes);
    }
  }

  // Очистка и заполнение черным

  vkResetCommandBuffer(fd.command_buffer, 0);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  vkBeginCommandBuffer(fd.command_buffer, &beginInfo);

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

  // Main

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

  const VkDeviceSize vertexBufferOffset = 0;
  vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertexBuffer,
                         &vertexBufferOffset);
  vkCmdBindIndexBuffer(fd.command_buffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

  // Set 0 (сцена) один на всех, привязываем один раз
  vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelineLayout, 0, 1, &sceneDescriptorSet, 0,
                          nullptr);

  // Set 1 (материал)
  auto bindModel = [&](size_t modelIndex, const VkDescriptorSet& materialSet,
                       uint32_t effect) {
    const uint32_t dynamicOffset = uint32_t(modelIndex * modelStride);
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 1, 1, &materialSet, 1,
                            &dynamicOffset);

    PushConstants pushConstants{effect, animTime};
    vkCmdPushConstants(fd.command_buffer, pipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pushConstants),
                       &pushConstants);
  };

  for (size_t i = 0; i < renderObjects.size(); ++i) {
    const RenderObject& object = renderObjects[i];

    bindModel(i, textures[object.textureIndex].descriptorSet,
              object.nonTrivialSampling ? 1 : 0);
    vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(indices.size()),
                     1, 0, 0, 0);
  }

  if (showLightVisualizations) {
    // Кубы точечных источников.
    for (size_t i = 0; i < pointLights.size(); ++i) {
      bindModel(pointMarkerIndex(i), textures[0].descriptorSet, 0);
      vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(indices.size()),
                       1, 0, 0, 0);
    }

    // Кубы прожекторных источников.
    for (size_t i = 0; i < spotLights.size(); ++i) {
      bindModel(spotMarkerIndex(i), textures[0].descriptorSet, 0);
      vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(indices.size()),
                       1, 0, 0, 0);
    }

    // Белые проволочные конусы прожекторов.
    if (!coneVertices.empty()) {
      vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        linePipeline);
      const VkDeviceSize lineOffset = 0;
      vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &coneVertexBuffer,
                             &lineOffset);

      bindModel(coneIndex(), textures[0].descriptorSet, 0);
      vkCmdDraw(fd.command_buffer, static_cast<uint32_t>(coneVertices.size()),
                1, 0, 0);

      // Восстанавливаем обычный pipeline и геометрию для последующих команд.
      vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        graphicsPipeline);
      vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertexBuffer,
                             &vertexBufferOffset);
      vkCmdBindIndexBuffer(fd.command_buffer, indexBuffer, 0,
                           VK_INDEX_TYPE_UINT32);
    }
  }

  vkCmdEndRenderPass(fd.command_buffer);
  vkEndCommandBuffer(fd.command_buffer);
}

}  // namespace application