#version 450

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

// Set 0 - данные сцены
layout(set = 0, binding = 0) uniform SceneData {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 ambient;
    vec4 sunDirection;
    vec4 sunColor;
    ivec4 counts;
} scene;

struct PointLight {
    vec4 position;
    vec4 color;
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color;
    vec4 cone;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight data[];
} pointLights;

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight data[];
} spotLights;

// Set 1 - данные модели
layout(set = 1, binding = 0) uniform ModelData {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specular;
} object;

// Модель Блинна-Фонга для одного источника.
vec3 blinnPhong(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 radiance) {
    // Рассеянная составляющая (закон Ламберта)
    float diffuse = max(dot(normal, lightDir), 0.0);

    // Бликовая составляющая
    float specular = 0.0;
    if (diffuse > 0.0) {
        vec3 halfDir = normalize(lightDir + viewDir);
        specular = pow(max(dot(normal, halfDir), 0.0), object.specular.w);
    }

    return radiance * (object.albedo.rgb * diffuse + object.specular.rgb * specular);
}

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 viewDir = normalize(scene.cameraPosition.xyz - fragPosition);

    // Рассеянный свет (ambient)
    vec3 color = scene.ambient.rgb * scene.ambient.w * object.albedo.rgb;

    // Направленный источник: затухания нет, направление везде одинаковое
    vec3 sunDir = normalize(-scene.sunDirection.xyz);
    color += blinnPhong(normal, viewDir, sunDir, scene.sunColor.rgb * scene.sunColor.w);

    // Точечные источники
    for (int i = 0; i < scene.counts.x; ++i) {
        PointLight light = pointLights.data[i];

        vec3 toLight = light.position.xyz - fragPosition;
        float dist2 = max(dot(toLight, toLight), 0.0001);
        vec3 lightDir = toLight * inversesqrt(dist2);

        // Затухание по закону обратных квадратов
        float attenuation = 1.0 / dist2;

        color += blinnPhong(normal, viewDir, lightDir, light.color.rgb * light.color.w * attenuation);
    }

    // Прожекторные источники
    for (int i = 0; i < scene.counts.y; ++i) {
        SpotLight light = spotLights.data[i];

        vec3 toLight = light.position.xyz - fragPosition;
        float dist2 = max(dot(toLight, toLight), 0.0001);
        vec3 lightDir = toLight * inversesqrt(dist2);

        float attenuation = 1.0 / dist2;

        // Косинус угла между осью прожектора и направлением от источника к точке
        float cosAngle = dot(-lightDir, normalize(light.direction.xyz));
        float cosInner = cos(radians(light.cone.x));
        float cosOuter = cos(radians(light.cone.y));

        // 1 внутри внутреннего конуса, 0 за внешним, между ними плавный переход (мягкий край)
        float edge = clamp((cosAngle - cosOuter) / max(cosInner - cosOuter, 0.0001), 0.0, 1.0);

        color += blinnPhong(normal, viewDir, lightDir, light.color.rgb * light.color.w * attenuation * edge);
    }

    outColor = vec4(color, 1.0);
}
