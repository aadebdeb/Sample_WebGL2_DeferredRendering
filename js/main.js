(function() {

  function addVertex3(vertices, vi, x, y, z) {
    vertices[vi++] = x;
    vertices[vi++] = y;
    vertices[vi++] = z;
    return vi;
  };
  
  function addTriangle(indices, i, v0, v1, v2) {
    indices[i++] = v0;
    indices[i++] = v1;
    indices[i++] = v2;
    return i;
  };
  
  function addQuad(indices, i, v00, v10, v01, v11) {
    indices[i] = v00;
    indices[i + 1] = indices[i + 5] = v10;
    indices[i + 2] = indices[i + 4] = v01;
    indices[i + 3] = v11;
    return i + 6;
  };
  
  function createSphere(radius, thetaSegment, phiSegment) {
    const vertexNum = 2 + (thetaSegment - 1) * phiSegment;
    const indexNum = phiSegment * 6 + (thetaSegment - 2) * phiSegment * 6;
    const indices = new Int16Array(indexNum);
    const positions = new Float32Array(3 * vertexNum);
    const normals = new Float32Array(3 * vertexNum);
  
    const thetaStep = Math.PI / thetaSegment;
    const phiStep = 2.0 * Math.PI / phiSegment;
  
    // setup positions & normals
    let posCount = 0;
    let normalCount = 0;
    posCount = addVertex3(positions, posCount, 0, -radius, 0);
    normalCount = addVertex3(normals, normalCount, 0, -1, 0);
    for (let hi = 1; hi < thetaSegment; hi++) {
      const theta = Math.PI - hi * thetaStep;
      const sinT = Math.sin(theta);
      const cosT = Math.cos(theta);
      for (let pi = 0; pi < phiSegment; pi++) {
        const phi = pi * phiStep;
        const sinP = Math.sin(-phi);
        const cosP = Math.cos(-phi);
        const p = new Vector3(
          radius * sinT * cosP,
          radius * cosT,
          radius * sinT * sinP
        );
        posCount = addVertex3(positions, posCount, p.x, p.y, p.z);
        const np = Vector3.norm(p);
        normalCount = addVertex3(normals, normalCount, np.x, np.y, np.z);
      }
    }
    posCount = addVertex3(positions, posCount, 0, radius, 0);
    normalCount = addVertex3(normals, normalCount, 0, 1, 0);
  
    // setup indices
    let indexCount = 0;
    for (let pi = 0; pi < phiSegment; pi++) {
      indexCount = addTriangle(indices, indexCount, 0, pi !== phiSegment - 1 ? pi + 2 : 1, pi + 1);
    }
    for (let hi = 0; hi < thetaSegment - 2; hi++) {
      const hj = hi + 1;
      for (let pi = 0; pi < phiSegment; pi++) {
        const pj = pi !== phiSegment - 1 ? pi + 1 : 0;
        indexCount = addQuad(indices, indexCount, 
          pi + hi * phiSegment + 1,
          pj + hi * phiSegment + 1,
          pi + hj * phiSegment + 1,
          pj + hj * phiSegment + 1
        );
      }
    }
    for (let pi = 0; pi < phiSegment; pi++) {
      indexCount = addTriangle(indices, indexCount,
        vertexNum - 1,
        pi + (thetaSegment - 2) * phiSegment + 1,
        (pi !== phiSegment - 1 ? pi + 1 : 0) + (thetaSegment - 2) * phiSegment + 1
      );
    }
  
    return {
      indices: indices,
      positions: positions,
      normals: normals,
    };
  }

  function createShader(gl, source, type) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader) + source);
    }
    return shader;
  }

  function createProgramFromSource(gl, vertexShaderSource, fragmentShaderSource) {
    const program = gl.createProgram();
    gl.attachShader(program, createShader(gl, vertexShaderSource, gl.VERTEX_SHADER));
    gl.attachShader(program, createShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program));
    }
    return program;
  }

  function createVbo(gl, array, usage) {
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, array, usage !== undefined ? usage : gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return vbo;
  }
  
  function createIbo(gl, array) {
    const ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, array, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    return ibo;
  }

  function getUniformLocations(gl, program, keys) {
    const locations = {};
    keys.forEach(key => {
        locations[key] = gl.getUniformLocation(program, key);
    });
    return locations;
  }
  
  function setUniformTexture(gl, index, texture, location) {
    gl.activeTexture(gl.TEXTURE0 + index);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(location, index);
  }

  function createTexture(gl, sizeX, sizeY, internalFormat, format, type) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, sizeX, sizeY, 0, format, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return texture;
  }

  const createGBuffer = (sizeX, sizeY) => {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const positionTexture = createTexture(gl, sizeX, sizeY, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, positionTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, positionTexture, 0);
    const normalTexture = createTexture(gl, sizeX, sizeY, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, normalTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, normalTexture, 0);
    const colorTexture = createTexture(gl, sizeX, sizeY, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, colorTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, colorTexture, 0);
    const depthTexture = createTexture(gl, sizeX, sizeY, gl.DEPTH_COMPONENT32F, gl.DEPTH_COMPONENT, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, depthTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depthTexture, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      positionTexture: positionTexture,
      normalTexture: normalTexture,
      colorTexture: colorTexture,
      depthTexture: depthTexture
    };
  };

  function hsvToRgb(h, s, v) {
    h = h / 60;
    const c = v * s;
    const x = c * (1.0 - Math.abs(h % 2 - 1));
    let r, g, b;
    if (h < 1.0) {
      [r, g, b] = [c, x, 0];
    } else if (h < 2.0) {
      [r, g, b] = [x, c, 0];
    } else if (h < 3.0) {
      [r, g, b] = [0, c, x];
    } else if (h < 4.0) {
      [r, g, b] = [0, x, c];
    } else if (h < 5.0) {
      [r, g, b] = [x, 0, c];
    } else {
      [r, g, b] = [c, 0, x];
    }
    const m = v - c;
    [r, g, b] = [r + m, g + m, b + m];
    return [r, g, b];
  }

  const GEOMETRY_PASS_VERTEX_SHADER_SOURCE =
`#version 300 es

layout (location = 0) in vec3 i_position;
layout (location = 1) in vec3 i_normal;

out vec3 v_position;
out vec3 v_normal;

uniform mat4 u_modelMatrix;
uniform mat4 u_normalMatrix;
uniform mat4 u_mvpMatrix;

void main(void) {
  vec4 position = vec4(i_position, 1.0);
  v_position = (u_modelMatrix * position).xyz;
  v_normal = (u_normalMatrix * vec4(i_normal, 0.0)).xyz;
  gl_Position = u_mvpMatrix * position;
}
`;

  const GEOMETRY_PASS_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in vec3 v_position;
in vec3 v_normal;

layout (location = 0) out vec3 o_position;
layout (location = 1) out vec3 o_normal;
layout (location = 2) out vec4 o_color;

uniform vec4 u_color;

void main(void) {
  o_position = v_position;
  o_normal = normalize(v_normal);
  o_color = u_color;
}
`;

  const FILL_VIEWPORT_VERTEX_SHADER_SOURCE =
`#version 300 es

layout (location = 0) in vec2 position;

out vec2 v_uv;

void main(void) {
  v_uv = position * 0.5 + 0.5;
  gl_Position = vec4(position, 0.0, 1.0);
}
`;

  const LIGHTING_PASS_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in vec2 v_uv;

out vec4 o_color;

uniform sampler2D u_positionTexture;
uniform sampler2D u_normalTexture;
uniform sampler2D u_colorTexture;
uniform vec3 u_lightDir;
uniform vec3 u_lightColor;
uniform vec3 u_cameraPos;

void main(void) {
  vec4 color = texture(u_colorTexture, v_uv);
  vec3 albedo = color.xyz;
  if (albedo == vec3(0.0)) {
    discard;
  }
  vec3 position = texture(u_positionTexture, v_uv).xyz;
  vec3 normal = texture(u_normalTexture, v_uv).xyz;
  float specIntensity = color.w;
  vec3 viewDir = normalize(u_cameraPos - position);

  vec3 lightDir = normalize(u_lightDir);
  vec3 reflectDir = reflect(-lightDir, normal);

  vec3 diffuse = albedo * u_lightColor * max(0.0, dot(lightDir, normal));
  vec3 specular =  albedo * pow(max(0.0, dot(viewDir, reflectDir)), specIntensity);
  o_color = vec4(diffuse + specular, 1.0);
}
`;

  const RENDER_POSITION_FRAGMENT_SHADRE_SOURCE =
`#version 300 es

precision highp float;

in vec2 v_uv;

out vec4 o_color;

uniform sampler2D u_positionTexture;

void main(void) {
  vec3 position = texture(u_positionTexture, v_uv).xyz * 0.1;
  o_color = vec4(position, 1.0);
}
`

  const RENDER_NORMAL_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in vec2 v_uv;

out vec4 o_color;

uniform sampler2D u_normalTexture;

void main(void) {
  vec3 normal = texture(u_normalTexture, v_uv).xyz;
  o_color = vec4(0.5 * normal + 0.5, 1.0);
}
`;

  const RENDER_ALBEDO_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in vec2 v_uv;

out vec4 o_color;

uniform sampler2D u_colorTexture;

void main(void) {
  vec3 albedo = texture(u_colorTexture, v_uv).xyz;
  o_color = vec4(albedo, 1.0);
}
`;

  const RENDER_SPECULAR_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in vec2 v_uv;

out vec4 o_color;

uniform sampler2D u_colorTexture;

void main(void) {
  float specular = texture(u_colorTexture, v_uv).w;
  o_color = vec4(vec3(specular / 32.0), 1.0);
}
`;

  const RENDER_DEPTH_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in vec2 v_uv;

out vec4 o_color;

uniform sampler2D u_depthTexture;
uniform float u_near;
uniform float u_far;

float convertToLinearDepth(float d, float near, float far) {
  return (2.0 * near) / (far + near - d * (far - near));
}

void main(void) {
  float depth = texture(u_depthTexture, v_uv).x;
  o_color = vec4(vec3(convertToLinearDepth(depth, u_near, u_far)), 1.0);
}
`;

  const FILL_VIEWPORT_POSITIONS = new Float32Array([
    -1.0, -1.0,
    1.0, -1.0,
    -1.0,  1.0,
    1.0,  1.0
  ]);

  const FILL_VIEWPORT_INDICES = new Int16Array([
    0, 1, 2,
    3, 2, 1
  ]);

  const stats = new Stats();
  document.body.appendChild(stats.dom);

  const parameters = {
    'render': 'lighting'
  };

  const gui = new dat.GUI();
  gui.add(parameters, 'render', ['lighting', 'pos/normal/albedo/spec', 'depth'])

  const canvas = document.getElementById('canvas');
  const resizeCanvas = () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  };
  resizeCanvas();

  const gl = canvas.getContext('webgl2');
  gl.getExtension('EXT_color_buffer_float');
  gl.enable(gl.CULL_FACE);

  const geometryPassProgram = createProgramFromSource(gl, GEOMETRY_PASS_VERTEX_SHADER_SOURCE, GEOMETRY_PASS_FRAGMENT_SHADER_SOURCE);
  const lightingPassProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, LIGHTING_PASS_FRAGMENT_SHADER_SOURCE);
  const renderPositionProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, RENDER_POSITION_FRAGMENT_SHADRE_SOURCE);
  const renderNormalProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, RENDER_NORMAL_FRAGMENT_SHADER_SOURCE);
  const renderAlbedoProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, RENDER_ALBEDO_FRAGMENT_SHADER_SOURCE);
  const renderSpecularProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, RENDER_SPECULAR_FRAGMENT_SHADER_SOURCE);
  const renderDepthProgram = createProgramFromSource(gl, FILL_VIEWPORT_VERTEX_SHADER_SOURCE, RENDER_DEPTH_FRAGMENT_SHADER_SOURCE);

  const geometryPassUniforms = getUniformLocations(gl, geometryPassProgram, ['u_modelMatrix', 'u_normalMatrix', 'u_mvpMatrix', 'u_color']);
  const lightingPassUniforms = getUniformLocations(gl, lightingPassProgram, ['u_positionTexture', 'u_normalTexture', 'u_colorTexture', 'u_lightDir', 'u_lightColor', 'u_cameraPos']);
  const renderPositionUniforms = getUniformLocations(gl, renderPositionProgram, ['u_positionTexture']);
  const renderNormalUniforms = getUniformLocations(gl, renderNormalProgram, ['u_normalTexture']);
  const renderAlbedoUniforms = getUniformLocations(gl, renderAlbedoProgram, ['u_colorTexture']);
  const renderSpecularUniforms = getUniformLocations(gl, renderSpecularProgram, ['u_colorTexture']);
  const renderDepthUniforms = getUniformLocations(gl, renderDepthProgram, ['u_depthTexture', 'u_near', 'u_far']);

  const sphere = createSphere(1.0, 16, 32);
  const sphereVao = gl.createVertexArray();
  gl.bindVertexArray(sphereVao);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, createIbo(gl, sphere.indices));
  [sphere.positions, sphere.normals].forEach((array, i) => {
    gl.bindBuffer(gl.ARRAY_BUFFER, createVbo(gl, array));
    gl.enableVertexAttribArray(i);
    gl.vertexAttribPointer(i, 3, gl.FLOAT, false, 0, 0);
  });
  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  const fillViewportVao = gl.createVertexArray();
  gl.bindVertexArray(fillViewportVao);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, createIbo(gl, FILL_VIEWPORT_INDICES));
  gl.bindBuffer(gl.ARRAY_BUFFER, createVbo(gl, FILL_VIEWPORT_POSITIONS));
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  const sphereInfos = [];
  for (let xi = -2; xi <= 2; xi++) {
    for (let yi = -2; yi <= 2; yi++) {
      for (let zi = -2; zi <= 2; zi++) {
        sphereInfos.push({
          position: new Vector3(xi * 10.0, yi * 10.0, zi * 10.0),
          radius: 2.0 + Math.random() * 3.0,
          color: hsvToRgb(Math.random() * 360, 1.0, 1.0),
          specular: Math.max(4.0, Math.random() * 32.0),
        });
      }
    }
  }

  let requestId = null;
  const reset = () => {
    if (requestId !== null) {
      cancelAnimationFrame(requestId);
    }

    const gBuffer = createGBuffer(canvas.width, canvas.height);
    const cameraNear = 0.01;
    const cameraFar = 300.0;
    let cameraPos;
    const updateCameraPos = (elapsedTime) => {
      cameraPos = new Vector3(
        100.0 * Math.cos(elapsedTime),
        50.0 * Math.cos(elapsedTime * 0.25),
        100.0 * Math.sin(elapsedTime)
      );
    };

    const proceedGeometryPass = () => {
      gl.enable(gl.DEPTH_TEST);
      gl.bindFramebuffer(gl.FRAMEBUFFER, gBuffer.framebuffer);
      gl.viewport(0.0, 0.0, canvas.width, canvas.height);
      gl.clearColor(0.0, 0.0, 0.0, 0.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      gl.useProgram(geometryPassProgram);
      const viewMatrix = Matrix4.inverse(Matrix4.lookAt(
        cameraPos,
        Vector3.zero,
        new Vector3(0.0, 1.0, 0.0)
      ));
      const projectionMatrix = Matrix4.perspective(canvas.width / canvas.height, 60, cameraNear, cameraFar);
      const vpMatrix = Matrix4.mul(viewMatrix, projectionMatrix);

      sphereInfos.forEach((sphereInfo) => {
        const modelMatrix = Matrix4.mul(
          Matrix4.scale(sphereInfo.radius, sphereInfo.radius, sphereInfo.radius),
          Matrix4.translate(sphereInfo.position.x, sphereInfo.position.y, sphereInfo.position.z)
        );
        const normalMatrix = Matrix4.transpose(Matrix4.inverse(modelMatrix));
        const mvpMatrix = Matrix4.mul(modelMatrix, vpMatrix);
  
        gl.uniformMatrix4fv(geometryPassUniforms['u_modelMatrix'], false, modelMatrix.elements);
        gl.uniformMatrix4fv(geometryPassUniforms['u_normalMatrix'], false, normalMatrix.elements);
        gl.uniformMatrix4fv(geometryPassUniforms['u_mvpMatrix'], false, mvpMatrix.elements);
        gl.uniform4f(geometryPassUniforms['u_color'], sphereInfo.color[0], sphereInfo.color[1], sphereInfo.color[2], sphereInfo.specular);

        gl.bindVertexArray(sphereVao);
  
        gl.drawElements(gl.TRIANGLES, sphere.indices.length, gl.UNSIGNED_SHORT, 0);
        gl.bindVertexArray(null);

      });

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.disable(gl.DEPTH_TEST);
    };

    const proceedLightingPass = () => {
      gl.clearColor(0.3, 0.3, 0.3, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(lightingPassProgram);
      setUniformTexture(gl, 0, gBuffer.positionTexture, lightingPassUniforms['u_positionTexture']);
      setUniformTexture(gl, 1, gBuffer.normalTexture, lightingPassUniforms['u_normalTexture']);
      setUniformTexture(gl, 2, gBuffer.colorTexture, lightingPassUniforms['u_colorTexture']);
      gl.uniform3f(lightingPassUniforms['u_lightDir'], 1.0, 1.0, 1.0);
      gl.uniform3f(lightingPassUniforms['u_lightColor'], 1.0, 1.0, 1.0);
      gl.uniform3f(lightingPassUniforms['u_cameraPos'], cameraPos.x, cameraPos.y, cameraPos.z);

      gl.bindVertexArray(fillViewportVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
    };

    const renderPositionBuffer = () => {
      gl.useProgram(renderPositionProgram);
      setUniformTexture(gl, 0, gBuffer.positionTexture, renderPositionUniforms['u_positionTexture']);
      gl.bindVertexArray(fillViewportVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
    };

    const renderNormalBuffer = () => {
      gl.useProgram(renderNormalProgram);
      setUniformTexture(gl, 0, gBuffer.normalTexture, renderNormalUniforms['u_normalTexture']);
      gl.bindVertexArray(fillViewportVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
    };

    const renderAlbedoBuffer = () => {
      gl.useProgram(renderAlbedoProgram);
      setUniformTexture(gl, 0, gBuffer.colorTexture, renderAlbedoUniforms['u_colorTexture']);
      gl.bindVertexArray(fillViewportVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
    };

    const renderSpecularBuffer = () => {
      gl.useProgram(renderSpecularProgram);
      setUniformTexture(gl, 0, gBuffer.colorTexture, renderSpecularUniforms['u_colorTexture']);
      gl.bindVertexArray(fillViewportVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
    };

    const renderGBuffer = () => {
      const halfW = canvas.width * 0.5;
      const halfH = canvas.height * 0.5;
      gl.viewport(0.0, halfH, halfW, halfH);
      renderPositionBuffer();
      gl.viewport(halfW, halfH, halfW, halfH);
      renderNormalBuffer();
      gl.viewport(0.0, 0.0, halfW, halfH);
      renderAlbedoBuffer();
      gl.viewport(halfW, 0.0, halfW, halfH);
      renderSpecularBuffer();
    };

    const renderDepthBuffer = () => {
      gl.viewport(0.0, 0.0, canvas.width, canvas.height);
      gl.useProgram(renderDepthProgram);
      setUniformTexture(gl, 0, gBuffer.depthTexture, renderDepthUniforms['u_depthTexture']);
      gl.uniform1f(renderDepthUniforms['u_near'], cameraNear);
      gl.uniform1f(renderDepthUniforms['u_far'], cameraFar);
      gl.bindVertexArray(fillViewportVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
    };

    const startTime = performance.now();
    const render = () => {
      stats.update();

      const currentTime = performance.now();
      const elapsedTime = (currentTime - startTime) * 0.001;
      updateCameraPos(elapsedTime);

      proceedGeometryPass();

      if (parameters.render === 'lighting') {
        proceedLightingPass();
      } else if (parameters.render === 'pos/normal/albedo/spec') {
        renderGBuffer();
      } else if (parameters.render === 'depth') {
        renderDepthBuffer();
      }

      requestAnimationFrame(render);
    };
    render();
  };

  window.addEventListener('resize', _ => {
    resizeCanvas();
    reset();
  });

  reset();

}());