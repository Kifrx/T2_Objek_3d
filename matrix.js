// Export fungsi matriks murni
export function mat4Identity() {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

export function mat4Multiply(a, b) {
  const o = new Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] =
        a[r] * b[c * 4] +
        a[4 + r] * b[c * 4 + 1] +
        a[8 + r] * b[c * 4 + 2] +
        a[12 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

export function mat4Perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) / (near - far), -1,
    0, 0, (2 * far * near) / (near - far), 0
  ];
}

export function mat4LookAt(eye, target, up) {
  let zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
  let l = Math.hypot(zx, zy, zz); zx /= l; zy /= l; zz /= l;
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
  l = Math.hypot(xx, xy, xz); xx /= l; xy /= l; xz /= l;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  return [
    xx, yx, zx, 0,
    xy, yy, zy, 0,
    xz, yz, zz, 0,
    -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
    -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
    -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
    1
  ];
}

export function normalMatrix(m) {
  const a = (i, j) => m[j * 4 + i];
  const c00 = a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1);
  const c01 = -(a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0));
  const c02 = a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0);
  const c10 = -(a(0, 1) * a(2, 2) - a(0, 2) * a(2, 1));
  const c11 = a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0);
  const c12 = -(a(0, 0) * a(2, 1) - a(0, 1) * a(2, 0));
  const c20 = a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1);
  const c21 = -(a(0, 0) * a(1, 2) - a(0, 2) * a(1, 0));
  const c22 = a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0);
  const det = a(0, 0) * c00 + a(0, 1) * c01 + a(0, 2) * c02;
  const id = 1 / (det || 1);
  return [
    c00 * id, c01 * id, c02 * id,
    c10 * id, c11 * id, c12 * id,
    c20 * id, c21 * id, c22 * id
  ];
}

export function shadowMatrix(L, h) {
  const k = 1 / L[1];
  return [
    1, 0, 0, 0,
    -L[0] * k, 0, -L[2] * k, 0,
    0, 0, 1, 0,
    L[0] * h * k, h, L[2] * h * k, 1
  ];
}

// MATRIX STACK — HIERARCHICAL MODELLING
export let M = mat4Identity();
export const stack = [];

export function resetM() {
  M = mat4Identity();
}

export function push() { stack.push(M.slice()); }
export function pop()  { M = stack.pop(); }

export function translate(x, y, z) {
  M[12] += M[0] * x + M[4] * y + M[8] * z;
  M[13] += M[1] * x + M[5] * y + M[9] * z;
  M[14] += M[2] * x + M[6] * y + M[10] * z;
  M[15] += M[3] * x + M[7] * y + M[11] * z;
}

export function scale(x, y, z) {
  for (let i = 0; i < 4; i++) {
    M[i] *= x;
    M[4 + i] *= y;
    M[8 + i] *= z;
  }
}

export function rotateX(a) {
  const c = Math.cos(a), s = Math.sin(a);
  for (let i = 0; i < 4; i++) {
    const m1 = M[4 + i], m2 = M[8 + i];
    M[4 + i] = m1 * c + m2 * s;
    M[8 + i] = m2 * c - m1 * s;
  }
}

export function rotateY(a) {
  const c = Math.cos(a), s = Math.sin(a);
  for (let i = 0; i < 4; i++) {
    const m0 = M[i], m2 = M[8 + i];
    M[i]     = m0 * c - m2 * s;
    M[8 + i] = m0 * s + m2 * c;
  }
}

export function rotateZ(a) {
  const c = Math.cos(a), s = Math.sin(a);
  for (let i = 0; i < 4; i++) {
    const m0 = M[i], m1 = M[4 + i];
    M[i]     = m0 * c + m1 * s;
    M[4 + i] = m1 * c - m0 * s;
  }
}