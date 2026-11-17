#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

// Set 0 - данные сцены (в вершинном шейдере нужна только viewProjection)
layout(set = 0, binding = 0) uniform SceneData {
  mat4 viewProjection;
  vec4 cameraPosition;
  vec4 ambient;
  vec4 sunDirection;
  vec4 sunColor;
  ivec4 counts;
}
scene;

// Set 1, binding 0 - динамический uniform буфер, из него читается один элемент,
// сдвиг задаётся при vkCmdBindDescriptorSets
layout(set = 1, binding = 0) uniform ModelData {
  mat4 model;
  mat4 normalMatrix;
  vec4 albedo;
  vec4 specular;
}
object;

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;

void main() {
  vec4 worldPosition = object.model * vec4(inPosition, 1.0);

  fragPosition = worldPosition.xyz;
  fragNormal = mat3(object.normalMatrix) * inNormal;
  fragTexCoord = inTexCoord;

  gl_Position = scene.viewProjection * worldPosition;
}
