#!/usr/bin/env python3
"""Build a textured 3D head (GLB, glTF 2.0) from ONE photograph.

Pipeline
  1. procedural head: lathe profile + facial-feature displacements (nose, brows, lips, chin, cheekbones...)
  2. fit a weak-perspective camera (yaw/pitch/roll/scale/offset) so mesh landmarks land on the photo's landmarks
  3. bake a skin texture: project the photo onto the mesh; where the photo can't see (far cheek, hand, hair)
     use the mirrored side of the face, then a neutral skin tone
  4. add hair shell, neck, shoulders; write one GLB (< 2 MB)

Usage: python make_face_glb.py photo.png out/mercedes_sosa_face.glb
Needs: numpy, scipy, pillow
"""
import json, struct, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

PHOTO = sys.argv[1] if len(sys.argv) > 1 else "photo.png"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out/face.glb"
rng = np.random.default_rng(7)
ss = lambda e0, e1, x: np.clip((x - e0) / (e1 - e0), 0, 1) ** 2 * (3 - 2 * np.clip((x - e0) / (e1 - e0), 0, 1))

# ------------------------------------------------------------------ 1. head shape (millimetres, +Y up, +Z = face)
Y_MIN, Y_MAX = -123.0, 118.0
_rx = [(118, 0), (112, 26), (100, 50), (80, 66), (60, 72), (30, 74.5), (0, 74), (-20, 71), (-40, 66), (-60, 58),
       (-80, 49), (-100, 38), (-112, 27), (-120, 13), (-123, 0)]
_zf = [(118, -15), (105, 30), (90, 58), (70, 73), (40, 82), (15, 84), (0, 80), (-20, 78), (-45, 78), (-60, 77),
       (-76, 76), (-90, 72), (-105, 64), (-115, 52), (-121, 35), (-123, 30)]
_zb = [(118, -15), (112, -30), (100, -50), (80, -78), (60, -96), (40, -104), (20, -108), (0, -106), (-20, -104),
       (-40, -98), (-60, -85), (-76, -70), (-90, -50), (-105, -20), (-115, 5), (-121, 22), (-123, 30)]
mk = lambda pts: PchipInterpolator(*zip(*sorted(pts)))
RX, ZF, ZB = mk(_rx), mk(_zf), mk(_zb)

# (x, y, amplitude mm, sigma_x, sigma_y)   - amplitude is pushed along the surface normal on the front of the face
FEATURES = [
    (0, 14, 3.0, 9, 9), (0, -5, 6, 8, 14), (0, -22, 7.5, 8.5, 11), (0, -37, 10.5, 8.5, 8), (0, -46, 8, 9.5, 5.5),   # nose
    (-15, -40, 6, 6.5, 6), (15, -40, 6, 6.5, 6), (0, -52, -3, 14, 3.5),                                       # nostril wings, under-nose
    (-27, 17, 5.5, 17, 6.5), (27, 17, 5.5, 17, 6.5),                                                              # brow ridges
    (-31, 2, -6.5, 12, 8.5), (31, 2, -6.5, 12, 8.5),                                                              # eye sockets
    (-51, -14, 8, 19, 17), (51, -14, 8, 19, 17), (-44, -48, 5.5, 20, 20), (44, -48, 5.5, 20, 20),                 # cheekbones, cheeks
    (0, -63, 5, 15, 5), (0, -70, 8, 16, 4.6), (0, -84, 9.5, 15, 6), (0, -76.5, -3.6, 21, 1.7),                   # lips + mouth line
    (0, -95, -3.2, 14, 4), (0, -108, 13, 19, 12),                                                                 # chin
    (-66, 22, -4, 10, 18), (66, 22, -4, 10, 18), (0, 52, 3.5, 40, 22),                                            # temples, forehead
    (-28, -78, -3, 7, 8), (28, -78, -3, 7, 8),                                                                    # mouth-corner dimples
]


def base_point(theta, y):
    r, zf, zb = RX(y), ZF(y), ZB(y)
    zc, rz = 0.5 * (zf + zb), 0.5 * (zf - zb)
    return np.stack([r * np.sin(theta), y + 0 * theta, zc + rz * np.cos(theta)], -1)


def surf(theta, y, eps=1e-3):
    """Full skin surface: base lathe + features pushed along the base normal (front only)."""
    theta, y = np.broadcast_arrays(np.asarray(theta, float), np.asarray(y, float))
    B = base_point(theta, y)
    dt = base_point(theta + 1e-3, y) - base_point(theta - 1e-3, y)
    dy = base_point(theta, y + 0.05) - base_point(theta, y - 0.05)
    n = np.cross(dt, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    n = np.where((n[..., 2:3] < -0.0) & (np.abs(theta)[..., None] < 1.2), -n, n)
    if np.any(((n * np.stack([np.sin(theta), 0 * theta, np.cos(theta)], -1)).sum(-1) < 0)):
        sgn = np.sign((n * np.stack([np.sin(theta), 0 * theta, np.cos(theta)], -1)).sum(-1))
        n = n * np.where(sgn == 0, 1, sgn)[..., None]
    x, yy = B[..., 0], B[..., 1]
    D = np.zeros_like(x)
    for fx, fy, A, sx, sy in FEATURES:
        D += A * np.exp(-0.5 * (((x - fx) / sx) ** 2 + ((yy - fy) / sy) ** 2))
    wf = ss(0.15, 0.55, np.cos(theta)) * ss(-125, -112, yy) * 0 + ss(0.10, 0.60, np.cos(theta))
    P = B + n * (D * wf)[..., None]
    # lower face pushed forward (profile: lips/chin sit further forward than the sphere-like skull gives)
    P[..., 2] += 15.0 * ss(-25, -85, yy) * wf
    return P


# ------------------------------------------------------------------ 2. camera fit to the photo
img = Image.open(PHOTO).convert("L")
W, H = img.size
img_s = img.filter(ImageFilter.UnsharpMask(radius=1.6, percent=110, threshold=2))
A = np.asarray(img_s, float)


def rot(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def head_pt(x, y):
    """3D surface point of the mesh nearest to head-frame (x, y)."""
    th = math.asin(np.clip(x / float(RX(y)), -0.999, 0.999)) if x else 0.0
    return surf(th, y)


# landmark: (head x, head y, photo px, photo py, weight)
LM = [(-31, 0, 257, 147, 1), (31, 0, 327, 151, 1), (0, -45, 274, 205, 1.2), (0, -76, 285, 243, 1),
      (-25, -77, 254, 240, 1), (25, -77, 317, 246, 1), (0, 62, 297, 66, .5)]
L3 = np.array([head_pt(x, y) for x, y, *_ in LM])
L2 = np.array([[px, py] for *_, px, py, _w in LM])
LW = np.array([w for *_, w in LM])


def project(P, p):
    yaw, pitch, roll, s, cx, cy = p
    C = P @ rot(yaw, pitch, roll).T
    return np.stack([cx + s * C[..., 0], cy - s * C[..., 1]], -1), C


def resid(p):
    return ((project(L3, p)[0] - L2) * LW[:, None]).ravel()


p0 = [math.radians(-25), 0, math.radians(-3), 1.2, 290, 150]
lo = [math.radians(-50), math.radians(-20), math.radians(-12), .8, 150, 0]
hi = [math.radians(-5), math.radians(20), math.radians(8), 1.8, 450, 330]
fit = least_squares(resid, p0, bounds=(lo, hi))
CAM = fit.x
print("camera fit: yaw %.1f pitch %.1f roll %.1f scale %.3f cx %.1f cy %.1f" % (
    math.degrees(CAM[0]), math.degrees(CAM[1]), math.degrees(CAM[2]), *CAM[3:]))
print("landmark residuals px:", np.round(np.linalg.norm(project(L3, CAM)[0] - L2, axis=1), 1))

# ------------------------------------------------------------------ 3. bake skin texture (theta x y atlas)
TH_M = math.radians(100)
TS = 1024
th = np.linspace(-TH_M, TH_M, TS)[None, :].repeat(TS, 0)
yy = np.linspace(Y_MAX, Y_MIN, TS)[:, None].repeat(TS, 1)       # row 0 = top
# face skin polygon in photo pixels (excludes hair, hand, clothing); feathered
poly = [(252, 78), (275, 68), (300, 66), (322, 80), (348, 96), (370, 116), (388, 142), (398, 165), (401, 185), (398, 215),
        (390, 245), (372, 272), (345, 290), (318, 297), (293, 298), (268, 290), (255, 270), (249, 246), (243, 222),
        (236, 200), (233, 180), (233, 160), (238, 138), (245, 112)]
mask_img = Image.new("L", (W, H), 0)
ImageDraw.Draw(mask_img).polygon(poly, fill=255)
mask = np.asarray(mask_img.filter(ImageFilter.MinFilter(11)).filter(ImageFilter.GaussianBlur(4)), float) / 255
R = rot(*CAM[:3])


def sample(theta, y):
    P = surf(theta, y)
    e = 5e-4
    nT = (surf(theta + e, y) - surf(theta - e, y))
    nY = (surf(theta, y + 0.05) - surf(theta, y - 0.05))
    N = np.cross(nT, nY)
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12
    N = np.where((N * np.stack([np.sin(theta), 0 * theta, np.cos(theta)], -1)).sum(-1, keepdims=True) < 0, -N, N)
    uv, C = project(P, CAM)
    nz = (N @ R.T)[..., 2]
    xy = [uv[..., 1], uv[..., 0]]
    col = map_coordinates(A, xy, order=1, mode="nearest")
    m = map_coordinates(mask, xy, order=1, mode="constant")
    xh = np.abs(P[..., 0])
    lum_ok = np.where((y > 30) | ((xh > 40) & (y > -25)), ss(80, 125, col), 1.0)
    return col, m * ss(0.10, 0.42, nz) * lum_ok


cd, wd = sample(th, yy)
cm, wm = sample(-th, yy)
tot = wd + (1 - wd) * wm
fill = np.median(A[165:200, 330:380])
tex = (wd * cd + (1 - wd) * wm * cm + (1 - tot) * fill)
# soften the seam between photo / mirrored / fill, keep detail where the photo is direct
blur = np.asarray(Image.fromarray(tex.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5)), float)
k = ss(0.55, 0.95, wd)
tex = k * tex + (1 - k) * blur
# a little detail-preserving contrast to survive viewer shading
tex = np.clip((tex - 128) * 1.05 + 128, 0, 255)
tex_img = Image.fromarray(tex.astype(np.uint8)).convert("RGB")
tex_img = Image.merge("RGB", (tex_img.split()[0].point(lambda v: min(255, v * 1.03)),
                              tex_img.split()[1], tex_img.split()[2].point(lambda v: v * 0.96)))  # faint warm tint
tex_img.save("out/skin_tex.jpg", quality=86, optimize=True)

# hair texture: vertical strands
HW, HH = 512, 512
col = rng.normal(size=(HW,))
col = np.convolve(col, np.ones(3) / 3, "same")
streak = np.zeros((HH, HW))
for f, amp in ((1, 1.0), (0.37, 0.7), (0.11, 0.5)):
    c = np.convolve(rng.normal(size=HW + 8), np.ones(max(1, int(2 / f))) / max(1, int(2 / f)), "same")[4:HW + 4]
    vv = np.convolve(rng.normal(size=HH + 60), np.ones(30) / 30, "same")[30:HH + 30]
    streak += amp * (c[None, :] * (0.6 + 0.8 * (vv[:, None] - vv.min()) / (np.ptp(vv) + 1e-9)))
streak = (streak - streak.mean()) / streak.std()
sheen = np.exp(-((np.linspace(0, 1, HH)[:, None] - 0.18) / 0.07) ** 2) * 14
hair = np.clip(26 + 8 * streak + sheen, 8, 80)
Image.fromarray(hair.astype(np.uint8)).convert("RGB").save("out/hair_tex.jpg", quality=85, optimize=True)

# ------------------------------------------------------------------ 4. meshes
def grid_tris(nr, nc, wrap=True):
    t = []
    for i in range(nr - 1):
        for j in range(nc if wrap else nc - 1):
            a, b = i * nc + j, i * nc + (j + 1) % nc
            c, d = (i + 1) * nc + j, (i + 1) * nc + (j + 1) % nc
            t += [(a, c, b), (b, c, d)]
    return np.array(t, np.int64)


def normals(P, T):
    N = np.zeros_like(P)
    f = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    for k in range(3):
        np.add.at(N, T[:, k], f)
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)


def compact(P, N, UV, T):
    used = np.unique(T)
    remap = -np.ones(len(P), np.int64)
    remap[used] = np.arange(len(used))
    return P[used], N[used], UV[used], remap[T]


def hair_mask(alpha, y):
    th0 = math.radians(66)
    yh = 62 - 44 * (np.clip(alpha / th0, 0, 1) ** 2.2)
    w_top = ss(0, 14, y - yh)
    w_side = ss(0, 0.10, alpha - th0)
    return np.where(alpha < th0, w_top, np.maximum(w_top, w_side)) if False else np.maximum(w_top * (alpha < th0 + 0.02), w_side)


NC = 216
thetas = np.linspace(-math.pi, math.pi, NC, endpoint=False)
# rows: dense near top/bottom, skin
phi = np.linspace(0, math.pi, 112)
ys = 0.5 * (Y_MAX + Y_MIN) + 0.5 * (Y_MAX - Y_MIN) * np.cos(phi)
TH, YY = np.meshgrid(thetas, ys)
P = surf(TH, YY).reshape(-1, 3)
ALPHA = np.abs(((TH + math.pi) % (2 * math.pi)) - math.pi)   # angle from front, theta=0 front
Wh = hair_mask(np.abs(TH), YY)
T = grid_tris(len(ys), NC)
UV = np.stack([np.clip((TH.ravel() + TH_M) / (2 * TH_M), 0, 1), (Y_MAX - YY.ravel()) / (Y_MAX - Y_MIN)], 1)
# drop skin triangles fully buried under hair (saves size, nothing visible is lost)
wh = Wh.ravel()
keep = wh[T].min(1) < 0.9
Ts = T[keep]
Ns = normals(P, T)
skin = compact(P, Ns, UV, Ts)
print("skin verts", len(skin[0]), "tris", len(skin[3]))

# hair shell
hy = np.concatenate([ys[ys > -60], np.linspace(-60, -168, 30)[1:]])
HTH, HYY = np.meshgrid(thetas, hy)
HNC = NC
rx, zf, zb = RX(np.clip(HYY, Y_MIN, Y_MAX)), ZF(np.clip(HYY, Y_MIN, Y_MAX)), ZB(np.clip(HYY, Y_MIN, Y_MAX))
below = HYY < Y_MIN
rx = np.where(below, 0, rx)
zc, rz = 0.5 * (zf + zb), 0.5 * (zf - zb)
curt = ss(30, -95, HYY)
Rc_x = 80 + 10 * ss(-60, -168, HYY)
Rc_z, zcc = 72.0, -34.0
cx_ = rx + (np.maximum(Rc_x, rx) - rx) * curt
rz_ = rz + (np.maximum(Rc_z, rz) - rz) * curt
zc_ = zc + (zcc - zc) * curt
AL = np.abs(HTH)
w = hair_mask(AL, HYY)
bulge = 1 + 1.1 * np.exp(-((HYY - 10) / 50) ** 2) * ss(0.9, 1.4, AL)
top_vol = 1 + 0.6 * ss(55, 100, HYY)
wave = 1.3 * (np.sin(HTH * 19 + HYY * 0.045) + 0.7 * np.sin(HTH * 41 - HYY * 0.08))
thick = (5.0 * w * bulge * top_vol + 0.6) + wave * w
X = (cx_ + thick) * np.sin(HTH)
Z = zc_ + (rz_ + thick) * np.cos(HTH)
HP = np.stack([X, HYY, Z], -1).reshape(-1, 3)
HT = grid_tris(len(hy), HNC)
hw = w.ravel()
HT = HT[hw[HT].max(1) > 0.03]
HUV = np.stack([(HTH.ravel() + math.pi) / (2 * math.pi), (HYY.ravel() - (-170)) / 300], 1)
HN = normals(HP, grid_tris(len(hy), HNC))
hairm = compact(HP, HN, HUV, HT)
print("hair verts", len(hairm[0]), "tris", len(hairm[3]))


def lathe(prof, nc=48, z0=0.0):
    ysx, rxs, rzs = zip(*prof)
    th_ = np.linspace(-math.pi, math.pi, nc, endpoint=False)
    TT, YYY = np.meshgrid(th_, np.array(ysx))
    rxg, rzg = np.array(rxs)[:, None], np.array(rzs)[:, None]
    PP = np.stack([rxg * np.sin(TT), YYY, z0 + rzg * np.cos(TT)], -1).reshape(-1, 3)
    TR = grid_tris(len(ysx), nc)
    NN = normals(PP, TR)
    return PP, NN, np.zeros((len(PP), 2)), TR


neck = lathe([(-70, 40, 42), (-100, 42, 44), (-130, 45, 46), (-165, 50, 50)], 40, -32)
bust = lathe([(-150, 52, 52), (-165, 90, 62), (-185, 135, 78), (-210, 165, 90), (-235, 170, 92)], 56, -34)

# ------------------------------------------------------------------ 5. GLB writer
SC = 0.001  # mm -> m
OFF = np.array([0, 55.0, 10.0])
bin_ = bytearray()
views, accs = [], []


def add_view(data, target=None):
    while len(bin_) % 4:
        bin_.append(0)
    off = len(bin_)
    bin_.extend(data)
    v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
    if target:
        v["target"] = target
    views.append(v)
    return len(views) - 1


def add_acc(arr, ctype, comp, typ, target, normalized=False, minmax=False):
    bv = add_view(arr.tobytes(), target)
    a = {"bufferView": bv, "componentType": comp, "count": int(len(arr)), "type": typ}
    if normalized:
        a["normalized"] = True
    if minmax:
        a["min"], a["max"] = arr.min(0).tolist(), arr.max(0).tolist()
    accs.append(a)
    return len(accs) - 1


def prim(mesh, material):
    Pm, Nm, UVm, Tm = mesh
    Pm = ((Pm - OFF) * SC).astype(np.float32)
    Nm = Nm.astype(np.float32)
    at = {"POSITION": add_acc(Pm, None, 5126, "VEC3", 34962, minmax=True), "NORMAL": add_acc(Nm, None, 5126, "VEC3", 34962)}
    if material in (0, 1):
        u16 = np.clip(np.round(UVm * 65535), 0, 65535).astype(np.uint16)
        at["TEXCOORD_0"] = add_acc(u16, None, 5123, "VEC2", 34962, normalized=True)
    idx = Tm.astype(np.uint16).reshape(-1)
    return {"attributes": at, "indices": add_acc(idx, None, 5123, "SCALAR", 34963), "material": material, "mode": 4}


skin_jpg, hair_jpg = open("out/skin_tex.jpg", "rb").read(), open("out/hair_tex.jpg", "rb").read()
prims = {"head": prim(skin, 0), "hair": prim(hairm, 1), "neck": prim(neck, 2), "bust": prim(bust, 3)}
iv0, iv1 = add_view(skin_jpg), add_view(hair_jpg)
gltf = {
    "asset": {"version": "2.0", "generator": "make_face_glb.py", "copyright": "Procedural head textured from a public-domain-style source photograph for coursework"},
    "scene": 0, "scenes": [{"nodes": [0]}],
    "nodes": [{"name": "MercedesSosa_head", "children": [1, 2, 3, 4]},
              {"name": "face", "mesh": 0}, {"name": "hair", "mesh": 1}, {"name": "neck", "mesh": 2}, {"name": "shoulders", "mesh": 3}],
    "meshes": [{"name": "face", "primitives": [prims["head"]]}, {"name": "hair", "primitives": [prims["hair"]]},
               {"name": "neck", "primitives": [prims["neck"]]}, {"name": "shoulders", "primitives": [prims["bust"]]}],
    "materials": [
        {"name": "skin", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 0.72}},
        {"name": "hair", "doubleSided": True, "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}, "metallicFactor": 0.0, "roughnessFactor": 0.45}},
        {"name": "neck", "pbrMetallicRoughness": {"baseColorFactor": [0.15, 0.145, 0.14, 1], "metallicFactor": 0.0, "roughnessFactor": 0.8}},
        {"name": "garment", "doubleSided": True, "pbrMetallicRoughness": {"baseColorFactor": [0.04, 0.04, 0.045, 1], "metallicFactor": 0.0, "roughnessFactor": 0.9}}],
    "textures": [{"source": 0, "sampler": 0}, {"source": 1, "sampler": 1}],
    "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071},
                 {"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 33071}],
    "images": [{"bufferView": iv0, "mimeType": "image/jpeg"}, {"bufferView": iv1, "mimeType": "image/jpeg"}],
    "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": len(bin_)}]}
js = json.dumps(gltf, separators=(",", ":")).encode()
js += b" " * (-len(js) % 4)
while len(bin_) % 4:
    bin_.append(0)
glb = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_)) + struct.pack("<II", len(js), 0x4E4F534A) + js \
      + struct.pack("<II", len(bin_), 0x004E4942) + bytes(bin_)
open(OUT, "wb").write(glb)
print("wrote", OUT, len(glb), "bytes")

# keep what the preview/QA script needs
np.savez("out/_meshes.npz", cam=CAM, off=OFF, skin_P=skin[0], skin_N=skin[1], skin_UV=skin[2], skin_T=skin[3],
         hair_P=hairm[0], hair_N=hairm[1], hair_UV=hairm[2], hair_T=hairm[3],
         neck_P=neck[0], neck_N=neck[1], neck_T=neck[3], bust_P=bust[0], bust_N=bust[1], bust_T=bust[3])
