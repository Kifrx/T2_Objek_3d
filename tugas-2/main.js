import { 
  mat4Identity, mat4Multiply, mat4Perspective, mat4LookAt, 
  normalMatrix, shadowMatrix, 
  M, stack, resetM, push, pop, translate, scale, rotateX, rotateY, rotateZ 
} from './matrix.js';

const canvas = document.getElementById('glcanvas');
const gl = canvas.getContext('webgl', { antialias: true, stencil: true });
if (!gl) {
  throw new Error("WebGL2 tidak tersedia.");
}

// =====================================================================
// 1. PENGATURAN SUASANA
// =====================================================================
const SKY_TOP    = '#5fa6ee';
const SKY_HORIZ  = '#bcf3cf';
const SUN_COLOR  = '#fff6e6';
const SUN_POS    = [14, 26, -4];
const HEMI_SKY   = '#dcefff';
const HEMI_GROUND= '#7a9158';
const AMBIENT    = 0.6;
const SUN_POWER  = 1.1;
const FOG_NEAR   = 200;
const FOG_FAR    = 4500;

function hexToRgb(hex) {
  const n = parseInt(hex.replace('#', ''), 16);
  return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
}
function toLinear(rgb) { return rgb.map(c => Math.pow(c, 2.2)); }
function linHex(hex) { return toLinear(hexToRgb(hex)); }

// =====================================================================
// 4. SHADER (GLSL)
// =====================================================================
const VERT_SRC = `
attribute vec3 aPos;
attribute vec3 aNormal;
attribute vec2 aUV;
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat3 uNormalMat;
varying vec3 vN;
varying vec3 vWorld;
varying vec2 vUV;
varying vec3 vDir;
varying float vDist;
void main() {
  vec4 w = uModel * vec4(aPos, 1.0);
  vec4 v = uView * w;
  vWorld = w.xyz;
  vN = uNormalMat * aNormal;
  vUV = aUV;
  vDir = aPos;
  vDist = length(v.xyz);
  gl_Position = uProj * v;
}`;

const FRAG_SRC = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
varying vec3 vN;
varying vec3 vWorld;
varying vec2 vUV;
varying vec3 vDir;
varying float vDist;

uniform float uMode;
uniform vec3 uColor;
uniform vec3 uEmissive;
uniform float uUseTex;
uniform sampler2D uTex;
uniform float uSpec;
uniform float uShin;
uniform float uMetal;

uniform vec3 uSunDir;
uniform vec3 uSunCol;
uniform vec3 uSkyCol;
uniform vec3 uGroundCol;
uniform float uAmb;
uniform vec3 uCamPos;
uniform vec3 uFogColor;
uniform float uFogNear;
uniform float uFogFar;
uniform vec3 uSkyTop;
uniform vec3 uSkyHor;

void main() {
  if (uMode > 1.5) {
    float t = clamp(normalize(vDir).y * 2.0, 0.0, 1.0);
    t = pow(t, 0.7);
    gl_FragColor = vec4(mix(uSkyHor, uSkyTop, t), 1.0);
    return;
  }

  float fog = clamp((vDist - uFogNear) / (uFogFar - uFogNear), 0.0, 1.0);

  if (uMode > 0.5) {
    gl_FragColor = vec4(0.02, 0.04, 0.08, 0.40 * (1.0 - fog));
    return;
  }

  vec3 N = normalize(vN);
  vec3 V = normalize(uCamPos - vWorld);
  vec3 L = normalize(uSunDir);

  vec3 albedo = uColor;
  if (uUseTex > 0.5) albedo *= pow(texture2D(uTex, vUV).rgb, vec3(2.2));

  float diff = max(dot(N, L), 0.0);
  vec3 hemi = mix(uGroundCol, uSkyCol, N.y * 0.5 + 0.5);
  vec3 col = albedo * (hemi * uAmb + uSunCol * diff);

  vec3 H = normalize(L + V);
  float sp = pow(max(dot(N, H), 0.0), uShin) * uSpec * step(0.001, diff);
  col += uSunCol * sp;

  vec3 R = reflect(-V, N);
  vec3 env = mix(uGroundCol, uSkyCol, clamp(R.y * 0.5 + 0.5, 0.0, 1.0));
  col = mix(col, env * (albedo * 0.6 + 0.4), uMetal * 0.4);

  col += uEmissive;
  col = mix(col, uFogColor, fog);
  col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
  gl_FragColor = vec4(col, 1.0);
}`;

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(s));
  }
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT_SRC));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG_SRC));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
  throw new Error(gl.getProgramInfoLog(prog));
}
gl.useProgram(prog);

const aPos = gl.getAttribLocation(prog, 'aPos');
const aNormal = gl.getAttribLocation(prog, 'aNormal');
const aUV = gl.getAttribLocation(prog, 'aUV');
gl.enableVertexAttribArray(aPos);
gl.enableVertexAttribArray(aNormal);
gl.enableVertexAttribArray(aUV);

const U = {};
['uModel','uView','uProj','uNormalMat','uMode','uColor','uEmissive','uUseTex','uTex',
 'uSpec','uShin','uMetal','uSunDir','uSunCol','uSkyCol','uGroundCol','uAmb','uCamPos',
 'uFogColor','uFogNear','uFogFar','uSkyTop','uSkyHor'
].forEach(n => U[n] = gl.getUniformLocation(prog, n));

// =====================================================================
// 5. PEMBUAT GEOMETRI
// =====================================================================
function makeGeometry(verts, indices) {
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);
  const ibo = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
  return { vbo, ibo, count: indices.length };
}

const boxCache = {};
function boxGeo(w, h, d) {
  const key = [w, h, d].join(',');
  if (boxCache[key]) return boxCache[key];
  const half = [w/2, h/2, d/2];
  const faces = [
    { n:[0,0,1],  u:[1,0,0],  v:[0,1,0] },
    { n:[0,0,-1], u:[-1,0,0], v:[0,1,0] },
    { n:[1,0,0],  u:[0,0,-1], v:[0,1,0] },
    { n:[-1,0,0], u:[0,0,1],  v:[0,1,0] },
    { n:[0,1,0],  u:[1,0,0],  v:[0,0,-1] },
    { n:[0,-1,0], u:[1,0,0],  v:[0,0,1] }
  ];
  const verts = [], idx = [];
  faces.forEach((f, fi) => {
    [[-1,-1,0,0],[1,-1,1,0],[1,1,1,1],[-1,1,0,1]].forEach(([s, t, uu, vv]) => {
      for (let k = 0; k < 3; k++) verts.push((f.n[k] + f.u[k]*s + f.v[k]*t) * half[k]);
      verts.push(f.n[0], f.n[1], f.n[2], uu, vv);
    });
    const b = fi * 4;
    idx.push(b, b+1, b+2, b, b+2, b+3);
  });
  return boxCache[key] = makeGeometry(verts, idx);
}

function planeGeo(w, l, repU, repV) {
  const x = w/2, z = l/2;
  const verts = [
    -x,0,-z, 0,1,0, 0,0,
     x,0,-z, 0,1,0, repU,0,
     x,0, z, 0,1,0, repU,repV,
    -x,0, z, 0,1,0, 0,repV
  ];
  return makeGeometry(verts, [0,2,1, 0,3,2]);
}

const cylCache = {};
function cylinderGeo(rTop, rBot, h, seg) {
  const key = [rTop, rBot, h, seg].join(',');
  if (cylCache[key]) return cylCache[key];
  const verts = [], idx = [];
  const ny = (rBot - rTop) / h;
  for (let i = 0; i <= seg; i++) {
    const a = i / seg * Math.PI * 2, c = Math.cos(a), s = Math.sin(a);
    const nl = Math.hypot(c, ny, s);
    verts.push(c*rTop, h/2, s*rTop, c/nl, ny/nl, s/nl, i/seg, 0);
    verts.push(c*rBot, -h/2, s*rBot, c/nl, ny/nl, s/nl, i/seg, 1);
  }
  for (let i = 0; i < seg; i++) {
    const a = i*2;
    idx.push(a, a+1, a+2, a+1, a+3, a+2);
  }
  [[h/2, rTop, 1], [-h/2, rBot, -1]].forEach(([y, r, ny2]) => {
    const center = verts.length / 8;
    verts.push(0, y, 0, 0, ny2, 0, .5, .5);
    for (let i = 0; i <= seg; i++) {
      const a = i / seg * Math.PI * 2;
      verts.push(Math.cos(a)*r, y, Math.sin(a)*r, 0, ny2, 0, .5, .5);
    }
    for (let i = 0; i < seg; i++) idx.push(center, center+1+i, center+2+i);
  });
  return cylCache[key] = makeGeometry(verts, idx);
}

function sphereGeo(lat, lon) {
  const verts = [], idx = [];
  for (let i = 0; i <= lat; i++) {
    const th = i * Math.PI / lat;
    for (let j = 0; j <= lon; j++) {
      const ph = j * Math.PI * 2 / lon;
      const x = Math.sin(th)*Math.cos(ph), y = Math.cos(th), z = Math.sin(th)*Math.sin(ph);
      verts.push(x, y, z, x, y, z, j/lon, i/lat);
    }
  }
  for (let i = 0; i < lat; i++) {
    for (let j = 0; j < lon; j++) {
      const a = i*(lon+1) + j, b = a + lon + 1;
      idx.push(a, b, a+1, b, b+1, a+1);
    }
  }
  return makeGeometry(verts, idx);
}

const GEO = {
  sphere: sphereGeo(10, 14),
  skyDome: sphereGeo(16, 24)
};


// =====================================================================
// 6. TEXTURE PROSEDURAL
// =====================================================================
function makePavingTexture() {
  const c = document.createElement('canvas');
  c.width = 512; c.height = 512;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#34383a';
  ctx.fillRect(0, 0, 512, 512);
  const cols = 16, rows = 32, bw = 512 / cols, bh = 512 / rows;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const shade = 92 + Math.random() * 38;
      ctx.fillStyle = `rgb(${shade},${shade + 2},${shade + 3})`;
      const off = (y % 2) * bw / 2;
      ctx.fillRect(x*bw + off + 1, y*bh + 1, bw - 2, bh - 2);
      ctx.fillRect(x*bw + off - 512 + 1, y*bh + 1, bw - 2, bh - 2);
    }
  }
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  const aniso = gl.getExtension('EXT_texture_filter_anisotropic');
  if (aniso) gl.texParameterf(gl.TEXTURE_2D, aniso.TEXTURE_MAX_ANISOTROPY_EXT, 8);
  return tex;
}
const pavingTex = makePavingTexture();

// =====================================================================
// 7. MATERIAL
// =====================================================================
function material(hex, o = {}) {
  return {
    color: o.tex ? [1, 1, 1] : linHex(hex),
    emissive: o.emissive ? linHex(o.emissive) : [0, 0, 0],
    tex: o.tex || null,
    spec: o.spec || 0,
    shin: o.shin || 32,
    metal: o.metal || 0,
    cast: o.cast !== false
  };
}

const MAT = {
  paving:  material('#ffffff', { tex: pavingTex, cast: false }),
  grass:   material('#5fae3a', { cast: false }),
  yellow:  material('#f2c418', { cast: false }),
  black:   material('#1d2123', { cast: false }),
  bark:    material('#6b5a4a'),
  leaf: [
    material('#3f9a2e'), material('#56b13a'),
    material('#2f8226'), material('#6bbd3e')
  ],
  pole:    material('#b9bdbf', { spec: .3, shin: 20 }),
  white:   material('#f4f4ed'),
  paint:   material('#bfc8ce', { metal: .72, spec: .9, shin: 70 }),
  darkPaint: material('#18232b', { metal: .45, spec: .5, shin: 50 }),
  glass:   material('#152d3d', { metal: .35, spec: 1, shin: 90 }),
  rubber:  material('#17191b'),
  rim:     material('#b7c0c5', { metal: .85, spec: .8, shin: 60 }),
  hub:     material('#555d62', { metal: .8, spec: .6, shin: 40 }),
  red:     material('#d71920', { emissive: '#330000' }),
  headLamp:material('#fff4ce', { emissive: '#665533' }),
  mBody:   material('#a31d22', { metal: .5, spec: .8, shin: 60 }),
  mDark:   material('#1b1e20'),
  jacket:  material('#2f5aa8'),
  pants:   material('#23272d'),
  helmet:  material('#f2f2ee', { metal: .2, spec: .9, shin: 80 })
};

// =====================================================================
// 8. RENDER MESH
// =====================================================================
const PASS_LIT = 0, PASS_SHADOW = 1;
let pass = PASS_LIT;
let lastMat = null;

const sunDir = (() => {
  const l = Math.hypot(...SUN_POS);
  return SUN_POS.map(v => v / l);
})();
const SHADOW_MAT = shadowMatrix(sunDir, 0.04);

function setMaterial(m) {
  if (m === lastMat) return;
  lastMat = m;
  gl.uniform3fv(U.uColor, m.color);
  gl.uniform3fv(U.uEmissive, m.emissive);
  gl.uniform1f(U.uSpec, m.spec);
  gl.uniform1f(U.uShin, m.shin);
  gl.uniform1f(U.uMetal, m.metal);
  if (m.tex) {
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, m.tex);
    gl.uniform1f(U.uUseTex, 1);
  } else {
    gl.uniform1f(U.uUseTex, 0);
  }
}

function drawMesh(geo, mat) {
  if (pass === PASS_SHADOW) {
    if (!mat.cast) return;
    gl.uniformMatrix4fv(U.uModel, false, new Float32Array(mat4Multiply(SHADOW_MAT, M)));
  } else {
    gl.uniformMatrix4fv(U.uModel, false, new Float32Array(M));
    gl.uniformMatrix3fv(U.uNormalMat, false, new Float32Array(normalMatrix(M)));
    setMaterial(mat);
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, geo.vbo);
  gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 32, 0);
  gl.vertexAttribPointer(aNormal, 3, gl.FLOAT, false, 32, 12);
  gl.vertexAttribPointer(aUV, 2, gl.FLOAT, false, 32, 24);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, geo.ibo);
  gl.drawElements(gl.TRIANGLES, geo.count, gl.UNSIGNED_SHORT, 0);
}

function box(w, h, d, mat, x, y, z, rotX = 0) {
  push();
  translate(x, y, z);
  if (rotX) rotateX(rotX);
  drawMesh(boxGeo(w, h, d), mat);
  pop();
}

// =====================================================================
// 9. LINGKUNGAN
// =====================================================================
const ROAD_LEN = 420, ROAD_Z = -190;
const roadGeo   = planeGeo(12,  ROAD_LEN, 3,   ROAD_LEN / 4);
const walkGeo   = planeGeo(2.4, ROAD_LEN, 1,   ROAD_LEN / 4);
const road2Geo  = planeGeo(9,   ROAD_LEN, 2.5, ROAD_LEN / 4);
const groundGeo = planeGeo(600, 600, 1, 1);

const trees = [];
function addTree(x, z, s, lean) {
  const crowns = [];
  for (let k = 0; k < 4; k++) {
    crowns.push({
      x: lean * 1.6 + (Math.random() - .5) * 2.6,
      y: 5.6 + Math.random() * 1.8,
      z: (Math.random() - .5) * 2.8,
      r: 1.5 + Math.random() * 1.1,
      m: Math.floor(Math.random() * 4)
    });
  }
  trees.push({ x, z, s, crowns });
}
for (let z = 8; z > -170; z -= 6) {
  addTree(-9.8 - Math.random() * 1.2, z,     1.1 + Math.random() * .4, 1);
  addTree( 8.3 + Math.random() * .6,  z - 2, 1.0 + Math.random() * .4, -.5);
  addTree(21 + Math.random() * 2,     z - 4, 1.2 + Math.random() * .5, -.3);
}

function drawTree(t) {
  push();
  translate(t.x, 0, t.z);
  scale(t.s, t.s, t.s);
  push();
  translate(0, 2.5, 0);
  drawMesh(cylinderGeo(.2, .32, 5, 8), MAT.bark);
  pop();
  for (const c of t.crowns) {
    push();
    translate(c.x, c.y, c.z);
    scale(c.r, c.r * .8, c.r);
    drawMesh(GEO.sphere, MAT.leaf[c.m]);
    pop();
  }
  pop();
}

function drawCurbs(x) {
  for (let z = 22, i = 0; z > -170; z -= 3, i++) {
    box(.4, .26, 2.9, i % 2 === 0 ? MAT.yellow : MAT.black, x, .12, z);
  }
}

function drawEnvironment() {
  push(); translate(0, -0.04, -100); drawMesh(groundGeo, MAT.grass); pop();
  push(); translate(0,    0,   ROAD_Z); drawMesh(roadGeo,  MAT.paving); pop();
  push(); translate(-7.3, .01, ROAD_Z); drawMesh(walkGeo,  MAT.paving); pop();
  push(); translate(15,   0,   ROAD_Z); drawMesh(road2Geo, MAT.paving); pop();

  drawCurbs(-6.2);
  drawCurbs(6.2);
  drawCurbs(10.4);

  for (let z = 22; z > -150; z -= 14) {
    push();
    translate(8.2, .3, z);
    scale(1.4, .72, 1.2);
    scale(.9, .9, .9);
    drawMesh(GEO.sphere, MAT.leaf[2]);
    pop();
  }
  for (let z = 150; z > -150; z -= 38) {
    push();
    translate(7.4, 4.5, z);
    drawMesh(cylinderGeo(.14, .2, 9, 10), MAT.pole);
    pop();
  }
  for (const t of trees) drawTree(t);
}

// =====================================================================
// 10. MOBIL
// =====================================================================
const car = { x: 0, z: 0, wheelAngle: 0 };
const CAR_START_Z = 40;

function drawWheel(x, z, angle) {
  push();
  translate(x, .48, z);
  rotateX(angle);
  rotateZ(Math.PI / 2);
  drawMesh(cylinderGeo(.43, .43, .24, 24), MAT.rubber);
  drawMesh(cylinderGeo(.23, .23, .255, 20), MAT.rim);
  drawMesh(cylinderGeo(.075, .075, .27, 16), MAT.hub);
  pop();
}

function drawCar() {
  push();
  translate(car.x, 0, car.z);

  box(2.05, .62, 4.15, MAT.paint, 0, .82, 0);
  box(2.08, .25, .18, MAT.darkPaint, 0, .55, 2.08);
  box(1.9, .12, .15, MAT.paint, 0, .43, 2.17);
  box(1.72, .78, 2.05, MAT.paint, 0, 1.48, .18);
  box(1.52, .56, .035, MAT.glass, 0, 1.55, 1.225, -.12);
  box(1.5, .55, .035, MAT.glass, 0, 1.56, -.86, .12);
  for (const side of [-1, 1]) {
    box(.035, .48, 1.55, MAT.glass, side * .875, 1.55, .16);
    box(.22, .14, .2, MAT.darkPaint, side * 1.08, 1.42, -.72);
  }
  for (const side of [-1, 1]) {
    box(.42, .22, .08, MAT.red, side * .72, 1.0, 2.105);
    box(.22, .08, .035, MAT.headLamp, side * .72, .91, 2.15);
  }
  box(.72, .22, .035, MAT.white, 0, .72, 2.18);
  box(.58, .08, .025, MAT.darkPaint, 0, .72, 2.205);

  drawWheel(-1.03, -1.35, car.wheelAngle);
  drawWheel( 1.03, -1.35, car.wheelAngle);
  drawWheel(-1.03,  1.35, car.wheelAngle);
  drawWheel( 1.03,  1.35, car.wheelAngle);
  pop();
}

// =====================================================================
// 11. MOTOR + PENGENDARA
// =====================================================================
const MOTOR_START_Z = -6;
const MOTOR_X = -2.6;
const MOTOR_SPEED = 6;
const motor = { z: MOTOR_START_Z, wheelAngle: 0, sway: 0 };

function drawMotorWheel(z, angle) {
  push();
  translate(0, .34, z);
  rotateX(angle);
  rotateZ(Math.PI / 2);
  drawMesh(cylinderGeo(.34, .34, .13, 20), MAT.rubber);
  drawMesh(cylinderGeo(.18, .18, .15, 14), MAT.rim);
  pop();
}

function drawMotor() {
  push();
  translate(MOTOR_X, 0, motor.z);
  rotateZ(motor.sway);

  box(.32, .42, 1.25, MAT.mBody, 0, .66, 0);
  box(.38, .12, .7, MAT.mDark, 0, .95, .22);
  box(.36, .16, .35, MAT.mBody, 0, .8, .78);
  box(.2, .1, .04, MAT.red, 0, .86, .97);
  box(.22, .12, .02, MAT.white, 0, .72, .98);
  box(.72, .05, .06, MAT.mDark, 0, 1.1, -.5);
  box(.08, .55, .08, MAT.mDark, 0, .75, -.58);

  drawMotorWheel(-.62, motor.wheelAngle);
  drawMotorWheel( .62, motor.wheelAngle);

  box(.46, .62, .3, MAT.jacket, 0, 1.42, .18, -.15);
  box(.16, .55, .16, MAT.jacket, -.3, 1.3, -.12, -1.0);
  box(.16, .55, .16, MAT.jacket,  .3, 1.3, -.12, -1.0);
  box(.2, .5, .22, MAT.pants, -.24, .98, .02, -.9);
  box(.2, .5, .22, MAT.pants,  .24, .98, .02, -.9);
  push();
  translate(0, 1.9, .1);
  scale(.2, .2, .2);
  drawMesh(GEO.sphere, MAT.helmet);
  pop();

  pop();
}

// =====================================================================
// 12. KAMERA
// =====================================================================
const CAM_POS = [-5.2, 1.6, 9];
const CAM_TARGET = [1.2, 1.3, -45];
const viewMat = mat4LookAt(CAM_POS, CAM_TARGET, [0, 1, 0]);
let projMat = mat4Identity();

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    gl.viewport(0, 0, w, h);
    projMat = mat4Perspective(60 * Math.PI / 180, w / h, 0.3, 320);
  }
}
window.addEventListener('resize', resize);

// =====================================================================
// 13. ANIMASI & KONTROL
// =====================================================================
let speed = 8;
let running = true;
const keys = {};

function resetScene() {
  car.x = 0;
  car.z = CAR_START_Z;
  car.wheelAngle = 0;
  speed = 8;
  motor.z = MOTOR_START_Z;
  motor.wheelAngle = 0;
}
resetScene();

const toggleBtn = document.getElementById('toggle');
function toggleRun() {
  running = !running;
  toggleBtn.textContent = running ? 'Jeda' : 'Lanjut';
}

window.addEventListener('keydown', (e) => {
  keys[e.key.toLowerCase()] = true;
  if (e.code === 'Space') { e.preventDefault(); toggleRun(); }
});
window.addEventListener('keyup', (e) => { keys[e.key.toLowerCase()] = false; });
toggleBtn.addEventListener('click', toggleRun);
document.getElementById('reset').addEventListener('click', resetScene);

function update(dt, time) {
  if (keys['w']) speed = Math.min(speed + 8 * dt, 22);
  if (keys['s']) speed = Math.max(speed - 8 * dt, 0);
  if (keys['a']) car.x = Math.max(car.x - 4 * dt, -3.5);
  if (keys['d']) car.x = Math.min(car.x + 4 * dt, 3.5);

  car.z -= speed * dt;
  motor.z -= MOTOR_SPEED * dt;

  car.wheelAngle -= (speed * dt) / .43;
  motor.wheelAngle -= (MOTOR_SPEED * dt) / .34;
  motor.sway = Math.sin(time * 1.3) * .015;

  if (car.z < -140) resetScene();
}

// =====================================================================
// 14. LOOP RENDER
// =====================================================================
function setGlobalUniforms() {
  gl.uniformMatrix4fv(U.uView, false, new Float32Array(viewMat));
  gl.uniformMatrix4fv(U.uProj, false, new Float32Array(projMat));
  gl.uniform3fv(U.uSunDir, sunDir);
  gl.uniform3fv(U.uSunCol, linHex(SUN_COLOR).map(v => v * SUN_POWER));
  gl.uniform3fv(U.uSkyCol, linHex(HEMI_SKY));
  gl.uniform3fv(U.uGroundCol, linHex(HEMI_GROUND));
  gl.uniform1f(U.uAmb, AMBIENT);
  gl.uniform3fv(U.uCamPos, CAM_POS);
  gl.uniform3fv(U.uFogColor, linHex(SKY_HORIZ));
  gl.uniform1f(U.uFogNear, FOG_NEAR);
  gl.uniform1f(U.uFogFar, FOG_FAR);
  gl.uniform3fv(U.uSkyTop, hexToRgb(SKY_TOP));
  gl.uniform3fv(U.uSkyHor, hexToRgb(SKY_HORIZ));
  gl.uniform1i(U.uTex, 0);
}

function drawWorld() {
  resetM(); // Perbaikan: ganti M = mat4Identity() dengan resetM()
  lastMat = null;
  drawEnvironment();
  drawMotor();
  drawCar();
}

function render() {
  resize();
  const c = hexToRgb(SKY_HORIZ);
  gl.clearColor(c[0], c[1], c[2], 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);
  gl.useProgram(prog);
  setGlobalUniforms();

  // 1) Langit
  gl.disable(gl.DEPTH_TEST);
  gl.depthMask(false);
  gl.uniform1f(U.uMode, 2);
  resetM(); // Perbaikan: ganti M = mat4Identity() dengan resetM()
  translate(CAM_POS[0], CAM_POS[1], CAM_POS[2]);
  scale(250, 250, 250);
  gl.uniformMatrix4fv(U.uModel, false, new Float32Array(M));
  gl.uniformMatrix3fv(U.uNormalMat, false, new Float32Array([1,0,0, 0,1,0, 0,0,1]));
  gl.bindBuffer(gl.ARRAY_BUFFER, GEO.skyDome.vbo);
  gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 32, 0);
  gl.vertexAttribPointer(aNormal, 3, gl.FLOAT, false, 32, 12);
  gl.vertexAttribPointer(aUV, 2, gl.FLOAT, false, 32, 24);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, GEO.skyDome.ibo);
  gl.drawElements(gl.TRIANGLES, GEO.skyDome.count, gl.UNSIGNED_SHORT, 0);

  // 2) Objek berlighting
  gl.enable(gl.DEPTH_TEST);
  gl.depthMask(true);
  gl.uniform1f(U.uMode, 0);
  pass = PASS_LIT;
  drawWorld();

  // 3) Bayangan
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.depthMask(false);
  gl.enable(gl.STENCIL_TEST);
  gl.stencilFunc(gl.EQUAL, 0, 0xFF);
  gl.stencilOp(gl.KEEP, gl.KEEP, gl.INCR);
  gl.uniform1f(U.uMode, 1);
  pass = PASS_SHADOW;
  drawWorld();
  gl.disable(gl.STENCIL_TEST);
  gl.disable(gl.BLEND);
  gl.depthMask(true);
}

let lastTime = performance.now();
function frame(now) {
  const dt = Math.min((now - lastTime) / 1000, .05);
  lastTime = now;
  if (running) update(dt, now / 1000);
  render();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
